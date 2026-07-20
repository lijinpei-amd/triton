/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_AMDGPU_ATTRDEFS
#define TRITON_AMDGPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "TritonAMDGPUDialect.td"

class TritonAMDGPU_Attr<string name, list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<TritonAMDGPU_Dialect, name, traits, baseCppClass> {
}

def AMDGPU_CvtScalePkLaneAttr : I32EnumAttr<
    "CvtScalePkLane", "source half-warp containing the packed scale",
    [
      I32EnumAttrCase<"H0", 0, "h0">,
      I32EnumAttrCase<"H1", 1, "h1">
    ]> {
  let cppNamespace = "::mlir::triton::amdgpu";
}

def AMDGPU_CvtScalePkBytesAttr : I32EnumAttr<
    "CvtScalePkBytes", "Vscale byte routing for the two destination half-warps",
    [
      I32EnumAttrCase<"B0", 0, "b0">,
      I32EnumAttrCase<"B1", 1, "b1">,
      I32EnumAttrCase<"B2", 2, "b2">,
      I32EnumAttrCase<"B3", 3, "b3">,
      I32EnumAttrCase<"B0B1", 4, "b0b1">,
      I32EnumAttrCase<"B2B3", 5, "b2b3">,
      I32EnumAttrCase<"B0B2", 6, "b0b2">,
      I32EnumAttrCase<"B1B3", 7, "b1b3">
    ]> {
  let cppNamespace = "::mlir::triton::amdgpu";
}

def AMDGPU_CvtScalePkScaleSelAttr
    : TritonAMDGPU_Attr<"CvtScalePkScaleSel"> {
  let mnemonic = "cvt_scale_pk_scale_sel";
  let parameters = (ins
    EnumParameter<AMDGPU_CvtScalePkLaneAttr>:$lane,
    EnumParameter<AMDGPU_CvtScalePkBytesAttr>:$bytes
  );
  let assemblyFormat = "`<` $lane `,` $bytes `>`";
}

def AMDGPU_CvtScalePkScaleSelArrayAttr
    : TypedArrayAttrBase<AMDGPU_CvtScalePkScaleSelAttr,
                         "array of cvt_scale_pk scale selections">;

def SetFP8Clamping : TritonAMDGPU_Attr<"SetFP8Clamping"> {
  let mnemonic = "amdgcn.set.fp8.clamping";
}

#endif
