#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::triton;

namespace {

using TensorCoords = SmallVector<std::pair<StringAttr, int32_t>>;
constexpr int kElementsPerPk8 = 8;

// Return the tensor coordinates of `elementIdx` for lane/warp/block zero.
static TensorCoords getElementCoords(const LinearLayout &layout,
                                     unsigned elementIdx, MLIRContext *ctx) {
  StringAttr kElement = StringAttr::get(ctx, "register");
  SmallVector<std::pair<StringAttr, int32_t>> layoutInputs;
  for (StringAttr inDim : layout.getInDimNames())
    layoutInputs.push_back({inDim, inDim == kElement ? elementIdx : 0});
  return layout.apply(layoutInputs);
}

// Find the local element index that owns `tensorCoords`.
static std::optional<int> getLocalElement(const LinearLayout &layout,
                                          const TensorCoords &tensorCoords,
                                          MLIRContext *ctx) {
  StringAttr kElement = StringAttr::get(ctx, "register");
  std::optional<int> elementIdx;
  for (auto [inDim, value] : layout.pseudoinvert().apply(tensorCoords)) {
    if (inDim == kElement)
      elementIdx = value;
    else if (value != 0)
      return std::nullopt;
  }
  return elementIdx;
}

// Emit a single `cvt.scale.pk8` ROCDL op producing 8 elements.
template <typename ConvertOp>
static Value createPk8(RewriterBase &rewriter, Location loc, Type resultType,
                       Value src, Value scale, int32_t scaleSel) {
  return ConvertOp::create(rewriter, loc, resultType, src, scale, scaleSel)
      .getRes();
}

using Pk8Emitter = Value (*)(RewriterBase &, Location, Type, Value, Value,
                             int32_t);

static Pk8Emitter selectPk8Emitter(bool fromFp4, bool isE4M3,
                                   Type elementType) {
  if (fromFp4) {
    if (elementType.isF16())
      return &createPk8<ROCDL::CvtPkScalePk8F16Fp4Op>;
    if (elementType.isBF16())
      return &createPk8<ROCDL::CvtPkScalePk8Bf16Fp4Op>;
    return &createPk8<ROCDL::CvtPkScalePk8F32Fp4Op>;
  }
  if (isE4M3) {
    if (elementType.isF16())
      return &createPk8<ROCDL::CvtPkScalePk8F16Fp8Op>;
    if (elementType.isBF16())
      return &createPk8<ROCDL::CvtPkScalePk8Bf16Fp8Op>;
    return &createPk8<ROCDL::CvtPkScalePk8F32Fp8Op>;
  }
  if (elementType.isF16())
    return &createPk8<ROCDL::CvtPkScalePk8F16Bf8Op>;
  if (elementType.isBF16())
    return &createPk8<ROCDL::CvtPkScalePk8Bf16Bf8Op>;
  return &createPk8<ROCDL::CvtPkScalePk8F32Bf8Op>;
}

struct CvtScalePkOpPattern
    : ConvertOpToLLVMPattern<amdgpu::CvtScalePkOp> {

  CvtScalePkOpPattern(const LLVMTypeConverter &converter,
                       const AMD::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(amdgpu::CvtScalePkOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (!targetInfo.supportsCvtPkScalePk8())
      return rewriter.notifyMatchFailure(
          op, "amdgpu.cvt_scale_pk requires gfx1250 (cvt.scale.pk8)");

    Type elementType = op.getType().getElementType();
    Type valElementType = op.getVal().getType().getElementType();
    bool fromFp4 = valElementType.isInteger(8);
    bool isE4M3 = isa<Float8E4M3FNType>(valElementType);

    auto valValues =
        unpackUniqueTensorElements(loc, adaptor.getVal(), rewriter);
    auto scaleValues =
        unpackUniqueTensorElements(loc, adaptor.getScale(), rewriter);

    int32_t axis = op.getAxis();
    int32_t packAxis = op.getPackAxis().value_or(axis);

    SmallVector<int64_t> valShape(op.getVal().getType().getShape());
    if (fromFp4)
      valShape[packAxis] *= 2;
    int64_t valK = valShape[axis];
    int64_t scaleK = op.getScale().getType().getShape()[axis];
    if (valK % scaleK != 0)
      return rewriter.notifyMatchFailure(op, "invalid scale/output axis sizes");
    int kFactor = static_cast<int>(valK / scaleK);
    SmallVector<int32_t> opSels;
    opSels.reserve(op.getScaleSel().size());
    for (Attribute scaleSelAttr : op.getScaleSel()) {
      auto scaleSel = cast<amdgpu::CvtScalePkScaleSelAttr>(scaleSelAttr);
      std::optional<int32_t> opSel = op.getOpSel(scaleSel);
      if (!opSel)
        return rewriter.notifyMatchFailure(
            op, "scale byte routing is not supported by the input type");
      opSels.push_back(*opSel);
    }
    if (opSels.empty())
      return rewriter.notifyMatchFailure(
          op, "scale_sel must contain at least one selection");
    unsigned scaleBits = op.getScale().getType().getElementTypeBitWidth();

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type resultType = vec_ty(elementType, kElementsPerPk8);
    auto scaleToI32 = [&](Value scale) -> Value {
      if (scaleBits < 32)
        return b.zext(i32_ty, scale);
      if (scaleBits > 32)
        return b.trunc(i32_ty, scale);
      return scale;
    };

    auto *ctx = rewriter.getContext();
    StringAttr kElement = StringAttr::get(ctx, "register");
    auto outDims = standardOutDimNames(ctx, op.getType().getRank());
    StringAttr axisDim = outDims[axis];

    // Work in the logical element layout. For fp4, insert a minor element bit
    // along pack_axis: elements 2*r/2*r+1 denote the low/high nibble of packed
    // i8 register r.
    LinearLayout valElementLayout =
        triton::gpu::toLinearLayout(op.getVal().getType());
    if (fromFp4) {
      StringAttr packDim = outDims[packAxis];
      valElementLayout =
          LinearLayout::identity1D(2, kElement, packDim) * valElementLayout;
    }
    valElementLayout = valElementLayout.removeZeroBasesAlongDim(kElement);
    LinearLayout scaleLayout =
        triton::gpu::toLinearLayout(op.getScale().getType())
            .removeZeroBasesAlongDim(kElement);
    LinearLayout outElementLayout =
        triton::gpu::toLinearLayout(op.getType())
            .removeZeroBasesAlongDim(kElement);
    if (valElementLayout != outElementLayout)
      return rewriter.notifyMatchFailure(
          op, "logical val layout is inconsistent with output layout");

    if (fromFp4 && kFactor % 2 != 0)
      return rewriter.notifyMatchFailure(op, "fp4 k_scale must be even");
    int valElementCount = valElementLayout.getInDimSize(kElement);
    int scaleElementCount = scaleLayout.getInDimSize(kElement);
    if (valElementCount !=
        static_cast<int>(valValues.size()) * (fromFp4 ? 2 : 1))
      return rewriter.notifyMatchFailure(
          op, "logical val element count is inconsistent with val");
    if (scaleElementCount != static_cast<int>(scaleValues.size()))
      return rewriter.notifyMatchFailure(
          op, "scale layout element count is inconsistent with scale");

    SmallVector<StringAttr> nonAxisDims;
    for (StringAttr dim : outDims)
      if (dim != axisDim)
        nonAxisDims.push_back(dim);
    auto allInputDims = llvm::to_vector(valElementLayout.getInDimNames());
    LinearLayout valNonAxisLayout =
        valElementLayout.sublayout(allInputDims, nonAxisDims)
            .removeZeroBasesAlongDim(kElement);
    LinearLayout valAxisLayout =
        valElementLayout.sublayout(allInputDims, {axisDim})
            .removeZeroBasesAlongDim(kElement);

    // Select the requested fp type combination once, outside the per-pk8
    // emitter.
    Pk8Emitter pk8Emitter = selectPk8Emitter(fromFp4, isE4M3, elementType);
    auto emitPk8 = [&](Value src, Value scaleI32, int32_t opSel) -> Value {
      return pk8Emitter(rewriter, loc, resultType, src, scaleI32, opSel);
    };

    auto withAxisCoord = [&](const TensorCoords &nonAxisCoords,
                             int32_t axisCoord) {
      TensorCoords tensorCoords;
      tensorCoords.reserve(outDims.size());
      int nonAxisCoordIdx = 0;
      for (StringAttr dim : outDims) {
        if (dim == axisDim) {
          tensorCoords.push_back({dim, axisCoord});
        } else {
          assert(nonAxisCoordIdx < static_cast<int>(nonAxisCoords.size()) &&
                 nonAxisCoords[nonAxisCoordIdx].first == dim);
          tensorCoords.push_back(nonAxisCoords[nonAxisCoordIdx++]);
        }
      }
      return tensorCoords;
    };

    struct AxisElementGroup {
      int32_t scaleCoord;
      SmallVector<int32_t> axisCoords;
    };

    // Split the logical element layout into its non-axis and axis projections.
    // Element bases that cover the low scale-block bits may appear anywhere
    // in the element domain, so group projected elements by their actual
    // scale coordinate and order each group by its low axis coordinate.
    int nonAxisElementCount = valNonAxisLayout.getInDimSize(kElement);
    int axisElementCount = valAxisLayout.getInDimSize(kElement);
    int elementsPerInstr = std::min(kElementsPerPk8, kFactor);
    if (axisElementCount % kFactor != 0)
      return rewriter.notifyMatchFailure(
          op, "axis element count is not divisible by kFactor");
    assert(nonAxisElementCount * axisElementCount == valElementCount);

    SmallVector<AxisElementGroup> axisElementGroups;
    for (int axisElementIdx = 0; axisElementIdx < axisElementCount;
         ++axisElementIdx) {
      TensorCoords axisCoords =
          getElementCoords(valAxisLayout, axisElementIdx, ctx);
      assert(axisCoords.size() == 1 && axisCoords[0].first == axisDim);
      int32_t axisCoord = axisCoords[0].second;
      int32_t scaleCoord = axisCoord / kFactor;
      int32_t elementOffset = axisCoord % kFactor;

      auto groupIt = llvm::find_if(
          axisElementGroups, [&](const AxisElementGroup &group) {
            return group.scaleCoord == scaleCoord;
          });
      if (groupIt == axisElementGroups.end()) {
        axisElementGroups.push_back(
            {scaleCoord, SmallVector<int32_t>(kFactor, -1)});
        groupIt = std::prev(axisElementGroups.end());
      }
      if (groupIt->axisCoords[elementOffset] != -1)
        return rewriter.notifyMatchFailure(
            op, "element mapping does not uniquely cover the low axis bits");
      groupIt->axisCoords[elementOffset] = axisCoord;
    }
    if (static_cast<int>(axisElementGroups.size()) * kFactor !=
            axisElementCount ||
        llvm::any_of(axisElementGroups, [](const AxisElementGroup &group) {
          return llvm::is_contained(group.axisCoords, -1);
        }))
      return rewriter.notifyMatchFailure(
          op, "element mapping does not cover all low scale-block axis bits");

    // Pair every non-axis element with each locally-held axis scale block;
    // each pair emits ceil(kFactor / kElementsPerPk8) pk8 calls.
    // Build wider vectors from equal-width inputs using a balanced shuffle
    // tree, preserving the packed-register pairs for LLVM's shuffle combiner.
    auto concatVectors = [&](SmallVector<Value> vectors) -> Value {
      assert(!vectors.empty());
      int paddedCount = 1;
      while (paddedCount < static_cast<int>(vectors.size()))
        paddedCount *= 2;
      while (static_cast<int>(vectors.size()) < paddedCount)
        vectors.push_back(b.undef(vectors.front().getType()));

      while (vectors.size() > 1) {
        SmallVector<Value> next;
        next.reserve(vectors.size() / 2);
        for (int i = 0; i < static_cast<int>(vectors.size()); i += 2) {
          int vectorWidth =
              cast<VectorType>(vectors[i].getType()).getNumElements();
          SmallVector<int32_t> mask;
          mask.reserve(2 * vectorWidth);
          for (int elementIdx = 0; elementIdx < 2 * vectorWidth; ++elementIdx)
            mask.push_back(elementIdx);
          next.push_back(LLVM::ShuffleVectorOp::create(
              rewriter, loc, vectors[i], vectors[i + 1], mask));
        }
        vectors = std::move(next);
      }
      return vectors.front();
    };

    SmallVector<Value> resultElements(valElementCount);
    for (int nonAxisElementIdx = 0;
         nonAxisElementIdx < nonAxisElementCount; ++nonAxisElementIdx) {
      TensorCoords nonAxisCoords =
          getElementCoords(valNonAxisLayout, nonAxisElementIdx, ctx);
      for (const AxisElementGroup &axisElementGroup : axisElementGroups) {
        int32_t opSel =
            opSels[axisElementGroup.scaleCoord % opSels.size()];
        TensorCoords scaleCoords =
            withAxisCoord(nonAxisCoords, axisElementGroup.scaleCoord);
        std::optional<int> scaleElementIdx =
            getLocalElement(scaleLayout, scaleCoords, ctx);
        if (!scaleElementIdx)
          return rewriter.notifyMatchFailure(
              op, "scale coordinate does not map to a local element");
        if (*scaleElementIdx < 0 || *scaleElementIdx >= scaleElementCount)
          return rewriter.notifyMatchFailure(
              op, "scale coordinate does not map to a local element");
        Value scaleI32 = scaleToI32(scaleValues[*scaleElementIdx]);

        for (int elementOffset = 0; elementOffset < kFactor;
             elementOffset += kElementsPerPk8) {
          SmallVector<Value> srcElements;
          // For fp4, bitcast each unique packed i8 register to its two i4
          // elements once, then record the packed register and nibble selected
          // by every logical source element.
          SmallVector<Value> fp4PackedRegs;
          SmallVector<Value> fp4ElementsByPackedReg;
          SmallVector<int32_t> fp4ShuffleMask;
          SmallVector<int> resultElementIndices;
          srcElements.reserve(elementsPerInstr);
          fp4PackedRegs.reserve(elementsPerInstr);
          fp4ElementsByPackedReg.reserve(elementsPerInstr);
          fp4ShuffleMask.reserve(kElementsPerPk8);
          resultElementIndices.reserve(elementsPerInstr);

          for (int instrElementIdx = 0; instrElementIdx < elementsPerInstr;
               ++instrElementIdx) {
            int32_t axisCoord =
                axisElementGroup.axisCoords[elementOffset + instrElementIdx];
            TensorCoords valCoords =
                withAxisCoord(nonAxisCoords, axisCoord);
            std::optional<int> elementIdx =
                getLocalElement(valElementLayout, valCoords, ctx);
            if (!elementIdx || *elementIdx < 0 ||
                *elementIdx >= valElementCount)
              return rewriter.notifyMatchFailure(
                  op, "val coordinate does not map to a local element");

            if (fromFp4) {
              int packedReg = *elementIdx >> 1;
              if (packedReg < 0 ||
                  packedReg >= static_cast<int>(valValues.size()))
                return rewriter.notifyMatchFailure(
                    op, "packed val register is out of range");
              Value packedRegValue = valValues[packedReg];
              auto packedRegIt =
                  llvm::find(fp4PackedRegs, packedRegValue);
              int packedRegIdx =
                  std::distance(fp4PackedRegs.begin(), packedRegIt);
              if (packedRegIt == fp4PackedRegs.end()) {
                fp4PackedRegs.push_back(packedRegValue);
                fp4ElementsByPackedReg.push_back(
                    b.bitcast(packedRegValue, vec_ty(int_ty(4), 2)));
              }
              fp4ShuffleMask.push_back(2 * packedRegIdx + (*elementIdx & 1));
            } else {
              srcElements.push_back(valValues[*elementIdx]);
            }
            resultElementIndices.push_back(*elementIdx);
          }

          Value src;
          if (fromFp4) {
            Value packedRegElements = concatVectors(fp4ElementsByPackedReg);
            int32_t undefElementIdx = static_cast<int32_t>(
                2 * fp4ElementsByPackedReg.size());
            fp4ShuffleMask.resize(kElementsPerPk8, undefElementIdx);
            Value fp4Elements = LLVM::ShuffleVectorOp::create(
                rewriter, loc, packedRegElements,
                b.undef(packedRegElements.getType()), fp4ShuffleMask);
            src = b.bitcast(fp4Elements, i32_ty);
          } else {
            Value srcVec = b.undef(vec_ty(i8_ty, kElementsPerPk8));
            for (auto [i, srcElement] : llvm::enumerate(srcElements))
              srcVec = b.insert_element(srcVec, srcElement, b.i32_val(i));
            src = b.bitcast(srcVec, vec_ty(i32_ty, 2));
          }
          Value convertedElements = emitPk8(src, scaleI32, opSel);
          for (auto [instrElementIdx, resultElementIdx] :
               llvm::enumerate(resultElementIndices)) {
            if (resultElements[resultElementIdx])
              return rewriter.notifyMatchFailure(
                  op, "output element was produced more than once");
            resultElements[resultElementIdx] =
                b.extract_element(convertedElements,
                                  b.i32_val(instrElementIdx));
          }
        }
      }
    }

    if (llvm::any_of(resultElements, [](Value value) { return !value; }))
      return rewriter.notifyMatchFailure(op,
                                         "not all output values were produced");

    Value result = packUniqueTensorElements(
        loc, getTypeConverter(), resultElements, rewriter, op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }

  const AMD::TargetInfo &targetInfo;
};

} // anonymous namespace

void mlir::triton::AMD::populateCvtScalePkOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const AMD::TargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<CvtScalePkOpPattern>(typeConverter, targetInfo, benefit);
}
