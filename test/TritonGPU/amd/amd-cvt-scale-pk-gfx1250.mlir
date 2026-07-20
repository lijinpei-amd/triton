// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250" --canonicalize --cse | FileCheck %s

// -----

// fp8 -> bf16, k_scale = 32, scale selection (h0, b0). Each thread holds one
// 32-wide row sharing a single i32 scale value, so 4 pk8 conversions all
// read the same scale with opsel 0.
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp8(%val: tensor<32x32xf8E4M3FN, #blocked>, %scale: tensor<32x1xi32, #blocked1>, %out: tensor<32x32x!tt.ptr<bf16>, #blocked>) {
    // CHECK-LABEL: @cvt_scale_pk_fp8
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32)>
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.bf16.fp8 {{.*}}, %[[SCALE]][0] : vector<8xbf16>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0>] {axis = 1 : i32} : tensor<32x32xf8E4M3FN, #blocked>, tensor<32x1xi32, #blocked1> -> tensor<32x32xbf16, #blocked>
    tt.store %out, %r : tensor<32x32x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// k_scale controls per-lane scale reuse independently of the half-wave routing
// selected by OPSEL. Here k_scale = 32 and b0b1 lowers to OPSEL 8.
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp8_independent_k_scale_and_scale_sel(%val: tensor<32x32xf8E4M3FN, #blocked>, %scale: tensor<32x1xi32, #blocked1>, %out: tensor<32x32x!tt.ptr<bf16>, #blocked>) {
    // CHECK-LABEL: @cvt_scale_pk_fp8_independent_k_scale_and_scale_sel
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32)>
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.bf16.fp8 {{.*}}, %[[SCALE]][8] : vector<8xbf16>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0b1>] {axis = 1 : i32} : tensor<32x32xf8E4M3FN, #blocked>, tensor<32x1xi32, #blocked1> -> tensor<32x32xbf16, #blocked>
    tt.store %out, %r : tensor<32x32x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// Four 32-element scale groups use two selections in round-robin order. Every
// pk8 in one 32-element group must use the same selection.
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp8_scale_sel_round_robin(%val: tensor<32x128xf8E4M3FN, #blocked>, %scale: tensor<32x4xi32, #blocked1>, %out: tensor<32x128x!tt.ptr<f32>, #blocked>) {
    // CHECK-LABEL: @cvt_scale_pk_fp8_scale_sel_round_robin
    // CHECK: %[[S0:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: %[[S1:.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: %[[S2:.*]] = llvm.extractvalue %arg1[2] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: %[[S3:.*]] = llvm.extractvalue %arg1[3] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[S0]][0] : vector<8xf32>
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[S1]][2] : vector<8xf32>
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[S2]][0] : vector<8xf32>
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[S3]][2] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0>, #amdg.cvt_scale_pk_scale_sel<h0, b2>] {axis = 1 : i32} : tensor<32x128xf8E4M3FN, #blocked>, tensor<32x4xi32, #blocked1> -> tensor<32x128xf32, #blocked>
    tt.store %out, %r : tensor<32x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// bf8 -> f32 with a 16-bit scale. The lowering zero-extends the packed scale
// before selecting the bf8 variant of the pk8 instruction.
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_bf8_i16_scale(%val: tensor<32x32xf8E5M2, #blocked>, %scale: tensor<32x1xi16, #blocked1>, %out: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    // CHECK-LABEL: @cvt_scale_pk_bf8_i16_scale
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i16)>
    // CHECK: %[[SCALE_I32:.*]] = llvm.zext %[[SCALE]] : i16 to i32
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.f32.bf8 {{.*}}, %[[SCALE_I32]][0] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0>] {axis = 1 : i32} : tensor<32x32xf8E5M2, #blocked>, tensor<32x1xi16, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.store %out, %r : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// An 8-bit scale is zero-extended to the i32 Vscale operand.
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp8_i8_scale(%val: tensor<32x32xf8E4M3FN, #blocked>, %scale: tensor<32x1xi8, #blocked1>, %out: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    // CHECK-LABEL: @cvt_scale_pk_fp8_i8_scale
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i8)>
    // CHECK: %[[SCALE_I32:.*]] = llvm.zext %[[SCALE]] : i8 to i32
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[SCALE_I32]][0] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0>] {axis = 1 : i32} : tensor<32x32xf8E4M3FN, #blocked>, tensor<32x1xi8, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.store %out, %r : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A 64-bit scale is truncated to its low 32-bit Vscale payload.
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp8_i64_scale(%val: tensor<32x32xf8E4M3FN, #blocked>, %scale: tensor<32x1xi64, #blocked1>, %out: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    // CHECK-LABEL: @cvt_scale_pk_fp8_i64_scale
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i64)>
    // CHECK: %[[SCALE_I32:.*]] = llvm.trunc %[[SCALE]] : i64 to i32
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.f32.fp8 {{.*}}, %[[SCALE_I32]][6] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b3>] {axis = 1 : i32} : tensor<32x32xf8E4M3FN, #blocked>, tensor<32x1xi64, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.store %out, %r : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// fp4 packed along dim0 and scaled along dim1. The element layout keeps
// low/high nibbles adjacent in element order, while the scale-friendly element
// order groups each 16-wide dim1 block. The lowering maps between those orders,
// repacks eight selected nibbles into four temporary bytes, and therefore emits
// two pk8 conversions for each 16-element scale block.
#val = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#scale = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#out = #ttg.blocked<{sizePerThread = [2, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp4_cross_axis(%val: tensor<32x32xi8, #val>, %scale: tensor<64x2xi32, #scale>, %out: tensor<64x32x!tt.ptr<f32>, #out>) {
    // CHECK-LABEL: @cvt_scale_pk_fp4_cross_axis
    // CHECK: %[[S0:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: %[[S1:.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: %[[S2:.*]] = llvm.extractvalue %arg1[2] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: %[[S3:.*]] = llvm.extractvalue %arg1[3] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: llvm.bitcast {{.*}} : i8 to vector<2xi4>
    // CHECK: llvm.shufflevector {{.*}} : vector<16xi4>
    // CHECK: llvm.bitcast {{.*}} : vector<8xi4> to i32
    // CHECK-COUNT-2: rocdl.cvt.scale.pk8.f32.fp4 {{.*}}, %[[S0]][4] : vector<8xf32>
    // CHECK-COUNT-2: rocdl.cvt.scale.pk8.f32.fp4 {{.*}}, %[[S2]][4] : vector<8xf32>
    // CHECK-COUNT-2: rocdl.cvt.scale.pk8.f32.fp4 {{.*}}, %[[S1]][4] : vector<8xf32>
    // CHECK-COUNT-2: rocdl.cvt.scale.pk8.f32.fp4 {{.*}}, %[[S3]][4] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0b2>] {axis = 1 : i32, pack_axis = 0 : i32} : tensor<32x32xi8, #val>, tensor<64x2xi32, #scale> -> tensor<64x32xf32, #out>
    tt.store %out, %r : tensor<64x32x!tt.ptr<f32>, #out>
    tt.return
  }
}

// -----

// The four low bits of each 16-element scale block are element-owned but do
// not appear first in element-basis order: the axis-16 basis precedes them.
// The lowering must group by the mapped axis coordinates rather than assuming
// that consecutive projected element indices form a scale block.
#packed = #ttg.linear<{register = [[0, 16], [0, 1], [0, 2], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
#scale = #ttg.linear<{register = [[1, 0], [0, 1]], lane = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [], block = []}>
#out = #ttg.linear<{register = [[1, 0], [0, 16], [0, 1], [0, 2], [0, 4], [0, 8]], lane = [[2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp4_permuted_axis_bases(%val: tensor<32x32xi8, #packed>, %scale: tensor<64x2xi32, #scale>, %out: tensor<64x32x!tt.ptr<f32>, #out>) {
    // CHECK-LABEL: @cvt_scale_pk_fp4_permuted_axis_bases
    // CHECK: %[[P0:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: %[[P1:.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: %[[P2:.*]] = llvm.extractvalue %arg1[2] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK: %[[P3:.*]] = llvm.extractvalue %arg1[3] : !llvm.struct<(i32, i32, i32, i32)>
    // CHECK-COUNT-2: rocdl.cvt.scale.pk8.f32.fp4 {{.*}}, %[[P0]][4] : vector<8xf32>
    // CHECK-COUNT-2: rocdl.cvt.scale.pk8.f32.fp4 {{.*}}, %[[P2]][4] : vector<8xf32>
    // CHECK-COUNT-2: rocdl.cvt.scale.pk8.f32.fp4 {{.*}}, %[[P1]][4] : vector<8xf32>
    // CHECK-COUNT-2: rocdl.cvt.scale.pk8.f32.fp4 {{.*}}, %[[P3]][4] : vector<8xf32>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b0b2>] {axis = 1 : i32, pack_axis = 0 : i32} : tensor<32x32xi8, #packed>, tensor<64x2xi32, #scale> -> tensor<64x32xf32, #out>
    tt.store %out, %r : tensor<64x32x!tt.ptr<f32>, #out>
    tt.return
  }
}

// -----

// fp4 -> f16, k_scale = 32, scale selection (h0, b2b3). Each thread holds a
// 64-wide row (2 scales), so the first 4 pk8 groups use scale[0] and the next
// 4 use scale[1], all with opsel 2.
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp4(%val: tensor<32x32xi8, #blocked>, %scale: tensor<32x2xi32, #blocked1>, %out: tensor<32x64x!tt.ptr<f16>, #blocked2>) {
    // CHECK-LABEL: @cvt_scale_pk_fp4
    // CHECK: %[[S0:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32, i32)>
    // CHECK: %[[S1:.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(i32, i32)>
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.f16.fp4 {{.*}}, %[[S0]][2] : vector<8xf16>
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.f16.fp4 {{.*}}, %[[S1]][2] : vector<8xf16>
    %r = amdg.cvt_scale_pk %val scale %scale scale_sel = [#amdg.cvt_scale_pk_scale_sel<h0, b2b3>] {axis = 1 : i32} : tensor<32x32xi8, #blocked>, tensor<32x2xi32, #blocked1> -> tensor<32x64xf16, #blocked2>
    tt.store %out, %r : tensor<32x64x!tt.ptr<f16>, #blocked2>
    tt.return
  }
}
