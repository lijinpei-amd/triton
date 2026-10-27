#include "triton/Analysis/BufferIndexAnalysis.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>
#include <optional>

namespace mlir {

/// A buffer index decomposed as `baseCoefficient * baseValue +
/// constantOffset` modulo `modulus`. The analysis records the modulus of
/// remsi and the pipeliner's select/cmpi idiom; `canonicalize` defaults a
/// missing modulus to the number of slots of the indexed descriptor.
struct BufferIndexExpr {
  Value baseValue;
  int64_t constantOffset = 0;
  std::optional<int64_t> modulus;
  Value indexedSource;
  // Multiplier of baseValue; 0 for constants, which have no base.
  int64_t baseCoefficient = 1;

  bool operator==(const BufferIndexExpr &) const = default;

  /// Sets a missing modulus to `numBuffers` and reduces the coefficient and
  /// the offset to [0, modulus), e.g. (3 - x) % 2 becomes (1 + x) % 2. A
  /// coefficient that reduces to 0 drops the base, so the expression compares
  /// like a constant.
  void canonicalize(int64_t numBuffers) {
    assert(numBuffers > 0);
    int64_t m = modulus.value_or(numBuffers);
    modulus = m;
    baseCoefficient = llvm::mod(baseCoefficient, m);
    constantOffset = llvm::mod(constantOffset, m);
    if (baseCoefficient == 0)
      baseValue = {};
  }

  bool isProvablyDifferentFrom(const BufferIndexExpr &other) const {
    if (baseValue != other.baseValue || indexedSource != other.indexedSource)
      return false;
    assert(modulus && other.modulus && "expected canonicalized expressions");
    // Proof. Membar compares slices only within one dynamic iteration, so
    // baseValue has a single runtime value b. Let v0 (this) and v1 (other) be
    // the i32 slot indices, with coefficients a0, a1, offsets c0, c1 and
    // moduli m0, m1, so that v_i == a_i * b + c_i (mod m_i). Every m_i is
    // positive, so m = gcd(m0, m1) >= 1 and d = gcd(a0 - a1, m) >= 1, also
    // when a0 == a1 because gcd(0, m) == m for m != 0.
    //   v0 == v1
    //   => a0 * b + c0 == a1 * b + c1 (mod m)
    //   => (a0 - a1) * b - (c1 - c0) == t * m for some integer t
    //   => d divides c1 - c0
    // Hence if d does not divide c1 - c0, then v0 != v1.
    // v_i == a_i * b + c_i (mod m_i) assumes that the i32 arithmetic folded
    // into a_i * b + c_i does not wrap, unless m_i divides 2^32. This covers
    // remsi operands and indices without a modulus, whose modulus is the
    // number of slots. These pairs are proven disjoint but select the same
    // slot when that arithmetic wraps:
    //   - (b + 4) % 3 vs b % 3 at b = 2^31 - 2.
    //   - b + INT_MIN vs (b + 2) % 3 in 3 buffers at b = INT_MIN.
    //   - ((b << 30) + 1) % 3 vs b % 3 at b = 3, 4 and 5. Folding muli and
    //     shli makes the gap reachable with small b.
    int64_t m = std::gcd(*modulus, *other.modulus);
    int64_t d = std::gcd(baseCoefficient - other.baseCoefficient, m);
    int64_t diff = other.constantOffset - constantOffset;
    return diff % d != 0;
  }
};

namespace {

enum class DFSState { Visiting, Visited };

/// A CFG is reducible iff every retreating edge in a depth-first traversal is
/// a backedge, i.e. its target dominates its source. Only blocks reachable
/// from the function entry matter to membar's forward dataflow analysis.
bool isReducibleCFG(Block *block, DenseMap<Block *, DFSState> &states,
                    DominanceInfo &dominanceInfo,
                    llvm::SmallPtrSetImpl<Block *> &loopHeaders) {
  states[block] = DFSState::Visiting;

  for (Block *successor : block->getTerminator()->getSuccessors()) {
    auto it = states.find(successor);
    if (it == states.end()) {
      if (!isReducibleCFG(successor, states, dominanceInfo, loopHeaders))
        return false;
      continue;
    }
    if (it->second == DFSState::Visiting) {
      if (!dominanceInfo.dominates(successor, block))
        return false;
      loopHeaders.insert(successor);
    }
  }

  states[block] = DFSState::Visited;
  return true;
}

bool isReducibleCFG(FunctionOpInterface funcOp, DominanceInfo &dominanceInfo,
                    llvm::SmallPtrSetImpl<Block *> &loopHeaders) {
  DenseMap<Block *, DFSState> states;
  return isReducibleCFG(&funcOp.getBlocks().front(), states, dominanceInfo,
                        loopHeaders);
}

std::optional<int64_t> getConstantIntValue(Value v) {
  APInt val;
  if (matchPattern(v, m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

BufferIndexExpr analyzeBufferIndex(Value indexValue);

bool isCFBlockArgProvablyBounded(BlockArgument blockArg, int64_t modulus,
                                 arith::SelectOp selectOp) {
  // For cf-form loops, the loop-carried value is a block argument. Each
  // incoming value must be either an in-range initial value or the matched
  // select that advances the counter.
  Block *header = blockArg.getOwner();
  unsigned argIdx = blockArg.getArgNumber();
  // Entry block arguments have no incoming operands to prove the bound.
  if (header->pred_begin() == header->pred_end())
    return false;

  for (auto predIt = header->pred_begin(), e = header->pred_end(); predIt != e;
       ++predIt) {
    Block *pred = *predIt;
    auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
    if (!branch)
      return false;

    auto operands = branch.getSuccessorOperands(predIt.getSuccessorIndex());
    Value incoming = operands[argIdx];

    if (incoming == selectOp.getResult())
      continue;

    auto c = getConstantIntValue(incoming);
    if (!c || *c < -1 || *c >= modulus)
      return false;
  }

  return true;
}

/// Verify that `base` is provably bounded by -1 <= base < N so that
///   select(cmpi sge/slt (addi(base, 1), N), ...)
/// truly equals (base + 1) % N. The select form returns 0 on the wrap arm
/// rather than (base + 1) - N; outside that range the two expressions
/// diverge and the match would be unsound.
///
/// Constants are checked directly. For cf-form loop-carried counters we prove
/// the bound inductively: incoming values to the loop-header block argument
/// must each be either an in-range constant or the matched select itself.
/// Given -1 <= base < N we have 0 <= base + 1 <= N, and the select maps N to 0
/// and otherwise returns base + 1, so the next value satisfies 0 <= next < N.
bool isBaseProvablyBounded(Value base, int64_t modulus,
                           arith::SelectOp selectOp) {
  assert(modulus > 0);

  if (auto c = getConstantIntValue(base))
    return *c >= -1 && *c < modulus;

  auto blockArg = dyn_cast<BlockArgument>(base);
  if (!blockArg)
    return false;

  return isCFBlockArgProvablyBounded(blockArg, modulus, selectOp);
}

/// Match the one-step modular wrap the pipeliner emits on its iter_arg:
///   select(cmpi sge (addi(base, 1), N), zero, addi(base, 1))
///   select(cmpi slt (addi(base, 1), N), addi(base, 1), zero)
/// Both equal (base + 1) % N only when -1 <= base < N; outside that range
/// the wrap arm would need to be (base + 1) - N, not 0, so accepting the
/// match unconditionally would be unsound. We require C == 1 and verify
/// the range assumption via isBaseProvablyBounded.
std::optional<BufferIndexExpr> matchModuloPattern(arith::SelectOp selectOp) {
  auto cmp = selectOp.getCondition().getDefiningOp<arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;

  Value wrapVal, noWrapVal;
  if (cmp.getPredicate() == arith::CmpIPredicate::sge) {
    wrapVal = selectOp.getTrueValue();
    noWrapVal = selectOp.getFalseValue();
  } else if (cmp.getPredicate() == arith::CmpIPredicate::slt) {
    noWrapVal = selectOp.getTrueValue();
    wrapVal = selectOp.getFalseValue();
  } else {
    return std::nullopt;
  }

  auto wrapConst = getConstantIntValue(wrapVal);
  if (!wrapConst || *wrapConst != 0)
    return std::nullopt;

  auto addOp = noWrapVal.getDefiningOp<arith::AddIOp>();
  if (!addOp || cmp.getLhs() != addOp.getResult())
    return std::nullopt;

  // Try constant on RHS then LHS (addi is commutative).
  std::optional<int64_t> c = getConstantIntValue(addOp.getRhs());
  Value base = addOp.getLhs();
  if (!c) {
    c = getConstantIntValue(addOp.getLhs());
    base = addOp.getRhs();
  }
  if (!c || *c != 1)
    return std::nullopt;

  // Modulus must be a positive compile-time constant.
  auto mod = getConstantIntValue(cmp.getRhs());
  if (!mod || *mod <= 0)
    return std::nullopt;

  // The (base + 1) % N rewrite is only valid for -1 <= base < N.
  if (!isBaseProvablyBounded(base, *mod, selectOp))
    return std::nullopt;

  auto baseExpr = analyzeBufferIndex(base);
  // Nested moduli ((x mod M) + 1) mod N don't reduce to (x + 1) mod N in
  // general; keep the full select as the expression root.
  if (baseExpr.modulus)
    return std::nullopt;
  BufferIndexExpr result = baseExpr;
  result.constantOffset += *c;
  result.modulus = *mod;
  return result;
}

// Folded coefficients and offsets stay within the signed range of the index
// type, which also keeps the int64_t arithmetic here from overflowing. The
// residue reasoning assumes that the folded arithmetic does not wrap; see
// BufferIndexExpr::isProvablyDifferentFrom.
BufferIndexExpr analyzeBufferIndex(Value indexValue) {
  if (auto c = getConstantIntValue(indexValue))
    return BufferIndexExpr{.constantOffset = *c, .baseCoefficient = 0};

  // memdesc_index takes an i32 index, and every recognized op preserves its
  // operand type.
  unsigned width = indexValue.getType().getIntOrFloatBitWidth();

  // Returns scale * x + constant.
  auto composeAffine = [&](Value nonConst, int64_t scale,
                           int64_t constant) -> BufferIndexExpr {
    auto expr = analyzeBufferIndex(nonConst);
    // (x mod N) * K + C is not represented as (base, offset, mod); keep the
    // full expression as the root.
    if (expr.modulus)
      return BufferIndexExpr{indexValue, 0};
    expr.baseCoefficient *= scale;
    expr.constantOffset = expr.constantOffset * scale + constant;
    if (!llvm::isIntN(width, expr.baseCoefficient) ||
        !llvm::isIntN(width, expr.constantOffset))
      return BufferIndexExpr{indexValue, 0};
    return expr;
  };

  if (auto addOp = indexValue.getDefiningOp<arith::AddIOp>()) {
    if (auto offset = getConstantIntValue(addOp.getRhs()))
      return composeAffine(addOp.getLhs(), 1, *offset);
    if (auto offset = getConstantIntValue(addOp.getLhs()))
      return composeAffine(addOp.getRhs(), 1, *offset);
  }

  if (auto subOp = indexValue.getDefiningOp<arith::SubIOp>()) {
    if (auto offset = getConstantIntValue(subOp.getRhs()))
      return composeAffine(subOp.getLhs(), 1, -*offset);
    if (auto offset = getConstantIntValue(subOp.getLhs()))
      return composeAffine(subOp.getRhs(), -1, *offset);
  }

  if (auto mulOp = indexValue.getDefiningOp<arith::MulIOp>()) {
    if (auto scale = getConstantIntValue(mulOp.getRhs()))
      return composeAffine(mulOp.getLhs(), *scale, 0);
    if (auto scale = getConstantIntValue(mulOp.getLhs()))
      return composeAffine(mulOp.getRhs(), *scale, 0);
  }

  // x << K equals x * 2^K modulo 2^width; K >= width yields poison. Take the
  // scale as a signed width-bit value, so x << 31 matches x * INT_MIN.
  if (auto shlOp = indexValue.getDefiningOp<arith::ShLIOp>()) {
    auto amount = getConstantIntValue(shlOp.getRhs());
    if (amount && *amount >= 0 && *amount < width)
      return composeAffine(shlOp.getLhs(),
                           APInt::getOneBitSet(width, *amount).getSExtValue(),
                           0);
  }

  if (auto selectOp = indexValue.getDefiningOp<arith::SelectOp>())
    if (auto result = matchModuloPattern(selectOp))
      return *result;

  // arith.remsi(x, N): strip the remainder and record N as the modulus.
  // N must be a positive compile-time constant.
  if (auto remOp = indexValue.getDefiningOp<arith::RemSIOp>()) {
    if (auto mod = getConstantIntValue(remOp.getRhs()); mod && *mod > 0) {
      auto result = analyzeBufferIndex(remOp.getLhs());
      // Nested modulus: keep the full remsi as the expression root.
      if (result.modulus)
        return BufferIndexExpr{indexValue, 0};
      result.modulus = *mod;
      return result;
    }
  }

  return BufferIndexExpr{indexValue, 0};
}

triton::gpu::MemDescIndexOp extractBufferIndex(Value value) {
  // MemDescIndexOp selects a whole slot of a multi-buffered allocation; its
  // index operand identifies the slot. MemDescViewTrait producers (trans,
  // reshape, reinterpret, subslice) are slot-preserving, so we can walk
  // through them to find the underlying MemDescIndexOp.
  Value v = value;
  while (auto *def = v.getDefiningOp()) {
    if (auto indexOp = dyn_cast<triton::gpu::MemDescIndexOp>(def))
      return indexOp;
    if (!def->hasTrait<OpTrait::MemDescViewTrait>())
      break;
    v = def->getOperand(0);
  }
  return {};
}

} // namespace

BufferIndexAnalysis::BufferIndexAnalysis(FunctionOpInterface funcOp)
    : dominanceInfo(funcOp),
      hasReducibleCFG(isReducibleCFG(funcOp, dominanceInfo, loopHeaders)) {}

BufferIndexAnalysis::~BufferIndexAnalysis() = default;

bool areBufferIndicesProvablyDifferent(const AllocationSlice &a,
                                       const AllocationSlice &b) {
  auto *aExpr = a.bufferIndexExpr;
  auto *bExpr = b.bufferIndexExpr;
  if (!aExpr || !bExpr)
    return false;
  return aExpr->isProvablyDifferentFrom(*bExpr);
}

const BufferIndexExpr *BufferIndexAnalysis::intern(BufferIndexExpr expr) {
  for (const auto &existing : expressions)
    if (*existing == expr)
      return existing.get();

  auto owned = std::make_unique<BufferIndexExpr>(expr);
  const BufferIndexExpr *result = owned.get();
  expressions.push_back(std::move(owned));
  return result;
}

AllocationSlice
BufferIndexAnalysis::makeSlice(Value value, Interval<size_t> allocationInterval,
                               Allocation::BufferId bufferId) {
  AllocationSlice slice(value, allocationInterval, bufferId);
  attachBufferIndex(slice, value);
  return slice;
}

void BufferIndexAnalysis::attachBufferIndex(AllocationSlice &slice,
                                            Value value) {
  if (!hasReducibleCFG)
    return;

  auto index = extractBufferIndex(value);
  if (!index)
    return;
  auto [it, inserted] = indexExpressions.try_emplace(index, nullptr);
  if (inserted)
    it->second = analyzeIndexOp(index);
  slice.bufferIndexExpr = it->second;
}

const BufferIndexExpr *
BufferIndexAnalysis::analyzeIndexOp(triton::gpu::MemDescIndexOp index) {
  // In-range slot indices lie in [0, numBuffers), so they are equal iff they
  // are congruent modulo numBuffers.
  int64_t numBuffers = index.getSrc().getType().getShape().front();
  if (numBuffers <= 0)
    return nullptr;
  auto expr = analyzeBufferIndex(index.getIndex());
  expr.indexedSource = index.getSrc();
  expr.canonicalize(numBuffers);
  return intern(std::move(expr));
}

bool BufferIndexAnalysis::isBackedgeSuccessor(Operation *terminator,
                                              Block *successor) const {
  if (isa<BranchOpInterface>(terminator))
    return !hasReducibleCFG ||
           !isa<FunctionOpInterface>(terminator->getParentOp()) ||
           dominanceInfo.dominates(successor, terminator->getBlock());
  // Region-to-region edges can start another dynamic iteration (scf.for and
  // scf.while). Conservatively invalidate on all such edges; exits to the
  // parent operation's continuation do not revisit the region's definitions.
  return isa<RegionBranchTerminatorOpInterface>(terminator) &&
         successor->getParentOp() == terminator->getParentOp();
}

bool BufferIndexAnalysis::entersLoop(Operation *terminator) const {
  if (!isa<BranchOpInterface>(terminator))
    return false;
  return llvm::any_of(terminator->getSuccessors(), [&](Block *successor) {
    return loopHeaders.contains(successor) &&
           !isBackedgeSuccessor(terminator, successor);
  });
}

} // namespace mlir
