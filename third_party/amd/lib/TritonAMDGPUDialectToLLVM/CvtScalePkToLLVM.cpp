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

// Find the register and lane that own `tensorCoords`, with warp/block held at
// zero. Keeping both coordinates is important for packed fp4: equal register
// indices in different lanes are different packed sources and must not be
// coalesced.
static std::optional<std::pair<int32_t, int32_t>>
getRegisterAndLane(const LinearLayout &layout,
                   const TensorCoords &tensorCoords, MLIRContext *ctx) {
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  int32_t reg = 0;
  int32_t lane = 0;
  for (auto [inDim, value] : layout.pseudoinvert().apply(tensorCoords)) {
    if (inDim == kRegister)
      reg = value;
    else if (inDim == kLane)
      lane = value;
    else if (value != 0)
      return std::nullopt;
  }
  return std::pair{reg, lane};
}

// Find the local register whose value at `lane` owns `tensorCoords`, with the
// remaining hierarchy coordinates held at zero.  cvt.scale.pk8 can source its
// scale from either lane x or lane x^16, so scale lookup cannot always use the
// current lane.
static std::optional<int>
getLocalElementAtLane(const LinearLayout &layout,
                      const TensorCoords &tensorCoords, int32_t lane,
                      MLIRContext *ctx) {
  StringAttr kElement = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  int elementCount = layout.getInDimSize(kElement);
  for (int elementIdx = 0; elementIdx < elementCount; ++elementIdx) {
    SmallVector<std::pair<StringAttr, int32_t>> inputs;
    for (StringAttr inDim : layout.getInDimNames()) {
      int32_t value = 0;
      if (inDim == kElement)
        value = elementIdx;
      else if (inDim == kLane)
        value = lane;
      inputs.push_back({inDim, value});
    }
    if (layout.apply(inputs) == tensorCoords)
      return elementIdx;
  }
  return std::nullopt;
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
    bool fromFp4 = isa<IntegerType>(valElementType);
    int fp4ValuesPerStorage =
        fromFp4 ? valElementType.getIntOrFloatBitWidth() / 4 : 1;
    bool isE4M3 = isa<Float8E4M3FNType>(valElementType);

    auto valValues =
        unpackUniqueTensorElements(loc, adaptor.getVal(), rewriter);
    auto scaleValues =
        unpackUniqueTensorElements(loc, adaptor.getScale(), rewriter);

    int32_t axis = op.getAxis();
    int32_t packAxis = op.getPackAxis().value_or(axis);

    SmallVector<int64_t> valShape(op.getVal().getType().getShape());
    if (fromFp4)
      valShape[packAxis] *= fp4ValuesPerStorage;
    int64_t valK = valShape[axis];
    int64_t scaleK = op.getScale().getType().getShape()[axis];
    if (valK % scaleK != 0)
      return rewriter.notifyMatchFailure(op, "invalid scale/output axis sizes");
    int scaleFactor = static_cast<int>(valK / scaleK);
    int kWidth = op.getKWidth().value_or(
        triton::gpu::getContigPerThread(op.getType())[axis]);
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
      return scale;
    };

    auto *ctx = rewriter.getContext();
    StringAttr kElement = StringAttr::get(ctx, "register");
    auto outDims = standardOutDimNames(ctx, op.getType().getRank());
    StringAttr axisDim = outDims[axis];

    // Work in the logical element layout. For fp4, insert the minor element
    // bits along pack_axis so each i8/i16/i32 storage element expands to
    // 2/4/8 consecutive logical nibbles.
    LinearLayout valStorageLayout =
        triton::gpu::toLinearLayout(op.getVal().getType())
            .removeZeroBasesAlongDim(kElement);
    LinearLayout valElementLayout = valStorageLayout;
    if (fromFp4) {
      StringAttr packDim = outDims[packAxis];
      valElementLayout = LinearLayout::identity1D(
                             fp4ValuesPerStorage, kElement, packDim) *
                         valElementLayout;
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

    if (fromFp4 && scaleFactor % 2 != 0)
      return rewriter.notifyMatchFailure(op, "fp4 scale_factor must be even");
    int valElementCount = valElementLayout.getInDimSize(kElement);
    int scaleElementCount = scaleLayout.getInDimSize(kElement);
    if (valElementCount !=
        static_cast<int>(valValues.size()) * fp4ValuesPerStorage)
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

    struct AxisSelectionGroup {
      int32_t selectorCoord;
      SmallVector<int32_t> axisCoords;
    };

    // Split the logical element layout into its non-axis and axis projections.
    // Element bases that cover the low k_width bits may appear anywhere in the
    // element domain, so group projected elements by their actual selection
    // coordinate and order each group by its low axis coordinate.
    int nonAxisElementCount = valNonAxisLayout.getInDimSize(kElement);
    int axisElementCount = valAxisLayout.getInDimSize(kElement);
    if (axisElementCount % kWidth != 0)
      return rewriter.notifyMatchFailure(
          op, "axis element count is not divisible by k_width");
    assert(nonAxisElementCount * axisElementCount == valElementCount);

    SmallVector<AxisSelectionGroup> axisSelectionGroups;
    for (int axisElementIdx = 0; axisElementIdx < axisElementCount;
         ++axisElementIdx) {
      TensorCoords axisCoords =
          getElementCoords(valAxisLayout, axisElementIdx, ctx);
      assert(axisCoords.size() == 1 && axisCoords[0].first == axisDim);
      int32_t axisCoord = axisCoords[0].second;
      int32_t selectorCoord = axisCoord / kWidth;
      int32_t elementOffset = axisCoord % kWidth;

      auto groupIt = llvm::find_if(
          axisSelectionGroups, [&](const AxisSelectionGroup &group) {
            return group.selectorCoord == selectorCoord;
          });
      if (groupIt == axisSelectionGroups.end()) {
        axisSelectionGroups.push_back(
            {selectorCoord, SmallVector<int32_t>(kWidth, -1)});
        groupIt = std::prev(axisSelectionGroups.end());
      }
      if (groupIt->axisCoords[elementOffset] != -1)
        return rewriter.notifyMatchFailure(
            op, "element mapping does not uniquely cover the low axis bits");
      groupIt->axisCoords[elementOffset] = axisCoord;
    }
    if (static_cast<int>(axisSelectionGroups.size()) * kWidth !=
            axisElementCount ||
        llvm::any_of(axisSelectionGroups,
                     [](const AxisSelectionGroup &group) {
          return llvm::is_contained(group.axisCoords, -1);
        }))
      return rewriter.notifyMatchFailure(
          op, "element mapping does not cover every k_width group");

    // Pair every non-axis element with each locally-held k_width group.  Split
    // again at scale_factor boundaries because scale selection and scale reuse
    // are independent.
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
      for (const AxisSelectionGroup &selectionGroup : axisSelectionGroups) {
        int32_t opSel = opSels[selectionGroup.selectorCoord % opSels.size()];
        int32_t sourceLane = (opSel & 1) ? 16 : 0;

        struct AxisScaleGroup {
          int32_t scaleCoord;
          SmallVector<int32_t> axisCoords;
        };
        SmallVector<AxisScaleGroup> scaleGroups;
        for (int32_t axisCoord : selectionGroup.axisCoords) {
          int32_t scaleCoord = axisCoord / scaleFactor;
          auto scaleGroupIt = llvm::find_if(
              scaleGroups, [&](const AxisScaleGroup &group) {
                return group.scaleCoord == scaleCoord;
              });
          if (scaleGroupIt == scaleGroups.end()) {
            scaleGroups.push_back({scaleCoord, {}});
            scaleGroupIt = std::prev(scaleGroups.end());
          }
          scaleGroupIt->axisCoords.push_back(axisCoord);
        }

        for (const AxisScaleGroup &scaleGroup : scaleGroups) {
          TensorCoords scaleCoords =
              withAxisCoord(nonAxisCoords, scaleGroup.scaleCoord);
          std::optional<int> scaleElementIdx = getLocalElementAtLane(
              scaleLayout, scaleCoords, sourceLane, ctx);
          if (!scaleElementIdx || *scaleElementIdx < 0 ||
              *scaleElementIdx >= scaleElementCount)
            return rewriter.notifyMatchFailure(
                op, "required scale is not owned by the selected lane or its "
                    "half-warp peer");
          Value scaleI32 = scaleToI32(scaleValues[*scaleElementIdx]);

          for (int elementOffset = 0;
               elementOffset < static_cast<int>(scaleGroup.axisCoords.size());
               elementOffset += kElementsPerPk8) {
            int elementsPerInstr = std::min(
                kElementsPerPk8,
                static_cast<int>(scaleGroup.axisCoords.size()) -
                    elementOffset);
            SmallVector<Value> srcElements;
            // For fp4, retain each logical value's packed (register, lane,
            // nibble) provenance. Bitcast each unique i8/i16/i32 source once
            // to <2/4/8xi4>; do not scalarize a contiguous packed source before
            // the final pk8 shuffle.
            SmallVector<std::pair<int32_t, int32_t>> fp4PackedSources;
            SmallVector<Value> fp4ElementsByPackedSource;
            SmallVector<int32_t> fp4ShuffleMask;
            SmallVector<int> resultElementIndices;
            srcElements.reserve(elementsPerInstr);
            fp4PackedSources.reserve(elementsPerInstr);
            fp4ElementsByPackedSource.reserve(elementsPerInstr);
            fp4ShuffleMask.reserve(kElementsPerPk8);
            resultElementIndices.reserve(elementsPerInstr);

            for (int instrElementIdx = 0; instrElementIdx < elementsPerInstr;
                 ++instrElementIdx) {
              int32_t axisCoord =
                  scaleGroup.axisCoords[elementOffset + instrElementIdx];
              TensorCoords valCoords =
                  withAxisCoord(nonAxisCoords, axisCoord);
              std::optional<int> elementIdx =
                  getLocalElement(valElementLayout, valCoords, ctx);
              if (!elementIdx || *elementIdx < 0 ||
                  *elementIdx >= valElementCount)
                return rewriter.notifyMatchFailure(
                    op, "val coordinate does not map to a local element");

              if (fromFp4) {
                TensorCoords packedCoords = valCoords;
                assert(packedCoords[packAxis].first == outDims[packAxis]);
                int32_t packedNibble =
                    packedCoords[packAxis].second % fp4ValuesPerStorage;
                packedCoords[packAxis].second /= fp4ValuesPerStorage;
                std::optional<std::pair<int32_t, int32_t>> packedSource =
                    getRegisterAndLane(valStorageLayout, packedCoords, ctx);
                if (!packedSource || packedSource->first < 0 ||
                    packedSource->first >= static_cast<int>(valValues.size()))
                  return rewriter.notifyMatchFailure(
                      op, "packed val coordinate does not map to a register "
                          "and lane");
                auto packedSourceIt =
                    llvm::find(fp4PackedSources, *packedSource);
                int packedSourceIdx =
                    std::distance(fp4PackedSources.begin(), packedSourceIt);
                if (packedSourceIt == fp4PackedSources.end()) {
                  fp4PackedSources.push_back(*packedSource);
                  fp4ElementsByPackedSource.push_back(b.bitcast(
                      valValues[packedSource->first],
                      vec_ty(int_ty(4), fp4ValuesPerStorage)));
                }
                fp4ShuffleMask.push_back(
                    fp4ValuesPerStorage * packedSourceIdx + packedNibble);
              } else {
                srcElements.push_back(valValues[*elementIdx]);
              }
              resultElementIndices.push_back(*elementIdx);
            }

            Value src;
            if (fromFp4) {
              Value packedRegElements =
                  concatVectors(fp4ElementsByPackedSource);
              int32_t undefElementIdx = static_cast<int32_t>(
                  fp4ValuesPerStorage * fp4ElementsByPackedSource.size());
              fp4ShuffleMask.resize(kElementsPerPk8, undefElementIdx);
              int packedWidth =
                  cast<VectorType>(packedRegElements.getType()).getNumElements();
              bool isIdentity = packedWidth == kElementsPerPk8;
              for (int i = 0; isIdentity && i < kElementsPerPk8; ++i)
                isIdentity = fp4ShuffleMask[i] == i;
              Value fp4Elements = packedRegElements;
              if (!isIdentity)
                fp4Elements = LLVM::ShuffleVectorOp::create(
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
