#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::triton;

namespace {

// Emit a single `cvt.scale.pk8` ROCDL op producing 8 packed results.
template <typename ConvertOp>
static Value createPk8(RewriterBase &rewriter, Location loc, Type resType,
                       Value src, Value scale, int32_t scaleSel) {
  return ConvertOp::create(rewriter, loc, resType, src, scale, scaleSel)
      .getRes();
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

    Type elemType = op.getType().getElementType();
    Type valElemType = op.getVal().getType().getElementType();
    bool fromFp4 = valElemType.isInteger(8);
    bool isE4M3 = isa<Float8E4M3FNType>(valElemType);

    auto valVals = unpackUniqueTensorElements(loc, adaptor.getVal(), rewriter);
    auto scaleVals =
        unpackUniqueTensorElements(loc, adaptor.getScale(), rewriter);

    int32_t axis = op.getAxis();
    // The lowering assumes the fp4 x2 expansion runs along `axis`; the verifier
    // enforces this, but guard defensively.
    if (auto packAttr = op.getPackAxisAttr(); packAttr && packAttr.getInt() != axis)
      return rewriter.notifyMatchFailure(op, "pack_axis != axis is not supported");

    int64_t outK = op.getType().getShape()[axis];
    int64_t scaleK = op.getScale().getType().getShape()[axis];
    if (scaleK <= 0 || outK % scaleK != 0)
      return rewriter.notifyMatchFailure(op, "invalid scale/output axis sizes");
    int kScale = static_cast<int>(outK / scaleK);
    int32_t scaleSel = op.getScaleSel();
    bool scale16 = op.getScale().getType().getElementTypeBitWidth() == 16;

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type resType = vec_ty(elemType, 8);

    // One pk8 always converts 8 values. `valRegsPerPk8` source registers feed
    // it: 4 i8 (fp4, 2 values each) or 8 fp8. `valRegsPerScale` source
    // registers make up one scale block (k_scale output values).
    int valRegsPerPk8 = fromFp4 ? 4 : 8;
    int valRegsPerScale = fromFp4 ? (kScale / 2) : kScale;
    if (fromFp4 && kScale % 2 != 0)
      return rewriter.notifyMatchFailure(op, "fp4 k_scale must be even");
    if (valRegsPerScale <= 0 || valVals.size() % valRegsPerScale != 0)
      return rewriter.notifyMatchFailure(
          op, "val register count is inconsistent with k_scale");
    int numScale = valVals.size() / valRegsPerScale;
    if (static_cast<size_t>(numScale) > scaleVals.size())
      return rewriter.notifyMatchFailure(
          op, "scale register count is inconsistent with val");

    // Emit the pk8 intrinsic for the requested fp type combination.
    auto emitPk8 = [&](Value src, Value scaleI32) -> Value {
      if (fromFp4)
        return elemType.isF16()  ? createPk8<ROCDL::CvtPkScalePk8F16Fp4Op>(
                                       rewriter, loc, resType, src, scaleI32, scaleSel)
               : elemType.isBF16() ? createPk8<ROCDL::CvtPkScalePk8Bf16Fp4Op>(
                                        rewriter, loc, resType, src, scaleI32, scaleSel)
                                   : createPk8<ROCDL::CvtPkScalePk8F32Fp4Op>(
                                        rewriter, loc, resType, src, scaleI32, scaleSel);
      if (isE4M3)
        return elemType.isF16()  ? createPk8<ROCDL::CvtPkScalePk8F16Fp8Op>(
                                       rewriter, loc, resType, src, scaleI32, scaleSel)
               : elemType.isBF16() ? createPk8<ROCDL::CvtPkScalePk8Bf16Fp8Op>(
                                        rewriter, loc, resType, src, scaleI32, scaleSel)
                                   : createPk8<ROCDL::CvtPkScalePk8F32Fp8Op>(
                                        rewriter, loc, resType, src, scaleI32, scaleSel);
      return elemType.isF16()  ? createPk8<ROCDL::CvtPkScalePk8F16Bf8Op>(
                                     rewriter, loc, resType, src, scaleI32, scaleSel)
             : elemType.isBF16() ? createPk8<ROCDL::CvtPkScalePk8Bf16Bf8Op>(
                                      rewriter, loc, resType, src, scaleI32, scaleSel)
                                 : createPk8<ROCDL::CvtPkScalePk8F32Bf8Op>(
                                      rewriter, loc, resType, src, scaleI32, scaleSel);
    };

    SmallVector<Value> results;
    results.reserve(numScale * kScale);

    for (int s = 0; s < numScale; ++s) {
      // The compact scale register (i32 = 4 E8M0 bytes; i16 = 2 bytes) is used
      // directly as the Vscale operand; scaleSel/opsel routes the byte per
      // lane-half in hardware.
      Value scaleI32 =
          scale16 ? b.zext(i32_ty, scaleVals[s]) : scaleVals[s];
      int base = s * valRegsPerScale;
      // Split the scale block into pk8 chunks. When k_scale < pk8 (or the last
      // chunk is short), the pk8 input is padded with undef and the extra
      // outputs are dropped.
      for (int off = 0; off < valRegsPerScale; off += valRegsPerPk8) {
        int realRegs = std::min(valRegsPerPk8, valRegsPerScale - off);
        Value packedVec = b.undef(vec_ty(i8_ty, valRegsPerPk8));
        for (int j = 0; j < realRegs; ++j)
          packedVec =
              b.insert_element(packedVec, valVals[base + off + j], b.i32_val(j));
        Value src = fromFp4 ? b.bitcast(packedVec, i32_ty)
                            : b.bitcast(packedVec, vec_ty(i32_ty, 2));
        Value res = emitPk8(src, scaleI32);
        // Keep only the outputs backed by real inputs (fp4: 2 per i8).
        int realOuts = fromFp4 ? realRegs * 2 : realRegs;
        for (int ii = 0; ii < realOuts; ++ii)
          results.push_back(b.extract_element(res, b.i32_val(ii)));
      }
    }

    Value result = packUniqueTensorElements(loc, getTypeConverter(), results,
                                            rewriter, op.getType());
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
