// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250" --canonicalize --cse | FileCheck %s

// -----

// FP8 -> BF16 with scale_factor = 256. Each lane owns eight consecutive
// values, while one scale is shared by all 32 lanes. k_width defaults to 8.
#val = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp8_cross_lane_scale(%val: tensor<1x256xf8E4M3FN, #val>, %scale: tensor<1x1xi32, #scale>, %out: tensor<1x256x!tt.ptr<bf16>, #val>) {
    // CHECK-LABEL: @cvt_scale_pk_fp8_cross_lane_scale
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32)>
    // CHECK: rocdl.cvt.scale.pk8.bf16.fp8 {{.*}}, %[[SCALE]][0] : vector<8xbf16>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0>] {axis = 1 : i32} : tensor<1x256xf8E4M3FN, #val>, tensor<1x1xi32, #scale> -> tensor<1x256xbf16, #val>
    tt.store %out, %r : tensor<1x256x!tt.ptr<bf16>, #val>
    tt.return
  }
}

// -----

// scale_factor and k_width are independent. One i32 scale is shared across
// all 1024 values, while each lane advances scale_sel every eight values.
#val = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp8_explicit_k_width(%val: tensor<1x1024xf8E4M3FN, #val>, %scale: tensor<1x1xi32, #scale>, %out: tensor<1x1024x!tt.ptr<f32>, #val>) {
    // CHECK-LABEL: @cvt_scale_pk_fp8_explicit_k_width
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32)>
    // CHECK: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[SCALE]][0] : vector<8xf32>
    // CHECK: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[SCALE]][3] : vector<8xf32>
    // CHECK: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[SCALE]][0] : vector<8xf32>
    // CHECK: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[SCALE]][3] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0>, #amdg.cvt_scale_pk_scale_sel<h1, b2>] {axis = 1 : i32, k_width = 8 : i32} : tensor<1x1024xf8E4M3FN, #val>, tensor<1x1xi32, #scale> -> tensor<1x1024xf32, #val>
    tt.store %out, %r : tensor<1x1024x!tt.ptr<f32>, #val>
    tt.return
  }
}

// -----

// A k_width smaller than pk8 is padded independently for each selection.
#val = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp8_short_k_width(%val: tensor<1x256xf8E4M3FN, #val>, %scale: tensor<1x1xi32, #scale>, %out: tensor<1x256x!tt.ptr<f32>, #val>) {
    // CHECK-LABEL: @cvt_scale_pk_fp8_short_k_width
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32)>
    // CHECK: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[SCALE]][0] : vector<8xf32>
    // CHECK: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[SCALE]][2] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0>, #amdg.cvt_scale_pk_scale_sel<h0, b2>] {axis = 1 : i32, k_width = 4 : i32} : tensor<1x256xf8E4M3FN, #val>, tensor<1x1xi32, #scale> -> tensor<1x256xf32, #val>
    tt.store %out, %r : tensor<1x256x!tt.ptr<f32>, #val>
    tt.return
  }
}

// -----

// BF8 with a 16-bit scale is zero-extended before conversion.
#val = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_bf8_i16_scale(%val: tensor<1x256xf8E5M2, #val>, %scale: tensor<1x1xi16, #scale>, %out: tensor<1x256x!tt.ptr<f32>, #val>) {
    // CHECK-LABEL: @cvt_scale_pk_bf8_i16_scale
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i16)>
    // CHECK: %[[SCALE_I32:.*]] = llvm.zext %[[SCALE]] : i16 to i32
    // CHECK: rocdl.cvt.scale.pk8.f32.bf8 {{.*}}, %[[SCALE_I32]][0] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0>] {axis = 1 : i32} : tensor<1x256xf8E5M2, #val>, tensor<1x1xi16, #scale> -> tensor<1x256xf32, #val>
    tt.store %out, %r : tensor<1x256x!tt.ptr<f32>, #val>
    tt.return
  }
}

// -----

// FP8 accepts an i8 scale and zero-extends it to the Vscale operand.
#val = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp8_i8_scale(%val: tensor<1x256xf8E4M3FN, #val>, %scale: tensor<1x1xi8, #scale>, %out: tensor<1x256x!tt.ptr<f32>, #val>) {
    // CHECK-LABEL: @cvt_scale_pk_fp8_i8_scale
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i8)>
    // CHECK: %[[SCALE_I32:.*]] = llvm.zext %[[SCALE]] : i8 to i32
    // CHECK: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[SCALE_I32]][0] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0>] {axis = 1 : i32} : tensor<1x256xf8E4M3FN, #val>, tensor<1x1xi8, #scale> -> tensor<1x256xf32, #val>
    tt.store %out, %r : tensor<1x256x!tt.ptr<f32>, #val>
    tt.return
  }
}

// -----

// Packed FP4 requires at least i16 because the two destination half-warps
// cannot select the same E8M0 byte.
#val = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#out = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp4_i16_scale(%val: tensor<1x128xi8, #val>, %scale: tensor<1x1xi16, #scale>, %out: tensor<1x256x!tt.ptr<f16>, #out>) {
    // CHECK-LABEL: @cvt_scale_pk_fp4_i16_scale
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i16)>
    // CHECK: %[[SCALE_I32:.*]] = llvm.zext %[[SCALE]] : i16 to i32
    // CHECK: rocdl.cvt.scale.pk8.f16.fp4 {{.*}}, %[[SCALE_I32]][0] : vector<8xf16>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0b1>] {axis = 1 : i32} : tensor<1x128xi8, #val>, tensor<1x1xi16, #scale> -> tensor<1x256xf16, #out>
    tt.store %out, %r : tensor<1x256x!tt.ptr<f16>, #out>
    tt.return
  }
}

// -----

// An i32 carries all eight contiguous fp4 inputs for one pk8 instruction.
// Preserve that word as a single packed source through lowering.
#val = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#out = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp4_i32_packed(%val: tensor<1x32xi32, #val>, %scale: tensor<1x1xi16, #scale>, %out: tensor<1x256x!tt.ptr<f16>, #out>) {
    // CHECK-LABEL: @cvt_scale_pk_fp4_i32_packed
    // CHECK: %[[VAL:.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(i32)>
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i16)>
    // CHECK: %[[SCALE_I32:.*]] = llvm.zext %[[SCALE]] : i16 to i32
    // CHECK: rocdl.cvt.scale.pk8.f16.fp4 %[[VAL]], %[[SCALE_I32]][0] : vector<8xf16>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0b1>] {axis = 1 : i32} : tensor<1x32xi32, #val>, tensor<1x1xi16, #scale> -> tensor<1x256xf16, #out>
    tt.store %out, %r : tensor<1x256x!tt.ptr<f16>, #out>
    tt.return
  }
}

// -----

// Each i16 contributes one contiguous <4xi4> cast; two such sources are
// concatenated for one pk8 instruction without extracting individual nibbles.
#val = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#out = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp4_i16_packed(%val: tensor<1x64xi16, #val>, %scale: tensor<1x1xi16, #scale>, %out: tensor<1x256x!tt.ptr<f16>, #out>) {
    // CHECK-LABEL: @cvt_scale_pk_fp4_i16_packed
    // CHECK: llvm.bitcast {{.*}} : i16 to vector<4xi4>
    // CHECK: llvm.bitcast {{.*}} : i16 to vector<4xi4>
    // CHECK: rocdl.cvt.scale.pk8.f16.fp4
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0b1>] {axis = 1 : i32} : tensor<1x64xi16, #val>, tensor<1x1xi16, #scale> -> tensor<1x256xf16, #out>
    tt.store %out, %r : tensor<1x256x!tt.ptr<f16>, #out>
    tt.return
  }
}

// -----

// FP4 packing and scaling can use different axes. k_width is measured after
// pack expansion, along the scaled axis.
#val = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#out = #ttg.linear<{register = [[1, 0], [0, 1], [0, 2], [2, 0], [4, 0]], lane = [[0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp4_cross_axis(%val: tensor<2x128xi8, #val>, %scale: tensor<4x1xi32, #scale>, %out: tensor<4x128x!tt.ptr<f32>, #out>) {
    // CHECK-LABEL: @cvt_scale_pk_fp4_cross_axis
    // CHECK: llvm.bitcast {{.*}} : i8 to vector<2xi4>
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.f32.fp4 {{.*}}[4] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0b2>] {axis = 1 : i32, k_width = 4 : i32, pack_axis = 0 : i32} : tensor<2x128xi8, #val>, tensor<4x1xi32, #scale> -> tensor<4x128xf32, #out>
    tt.store %out, %r : tensor<4x128x!tt.ptr<f32>, #out>
    tt.return
  }
}
