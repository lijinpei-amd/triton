// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1250" --canonicalize --cse | FileCheck %s

// -----

// fp8 -> bf16, k_scale = 32 (block32), scale_sel = 0. Each thread holds one
// 32-wide row sharing a single i32 scale register, so 4 pk8 conversions all
// read the same scale with opsel 0.
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cvt_scale_pk_fp8(%val: tensor<32x32xf8E4M3FN, #blocked>, %scale: tensor<32x1xi32, #blocked1>, %out: tensor<32x32x!tt.ptr<bf16>, #blocked>) {
    // CHECK-LABEL: @cvt_scale_pk_fp8
    // CHECK: %[[SCALE:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i32)>
    // CHECK-COUNT-4: rocdl.cvt.scale.pk8.bf16.fp8 {{.*}}, %[[SCALE]][0] : vector<8xbf16>
    %r = amdg.cvt_scale_pk %val scale %scale {axis = 1 : i32, scale_sel = 0 : i32} : tensor<32x32xf8E4M3FN, #blocked>, tensor<32x1xi32, #blocked1> -> tensor<32x32xbf16, #blocked>
    tt.store %out, %r : tensor<32x32x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// fp4 -> f16, k_scale = 32 (block32), scale_sel = 2. Each thread holds a
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
    %r = amdg.cvt_scale_pk %val scale %scale {axis = 1 : i32, scale_sel = 2 : i32} : tensor<32x32xi8, #blocked>, tensor<32x2xi32, #blocked1> -> tensor<32x64xf16, #blocked2>
    tt.store %out, %r : tensor<32x64x!tt.ptr<f16>, #blocked2>
    tt.return
  }
}
