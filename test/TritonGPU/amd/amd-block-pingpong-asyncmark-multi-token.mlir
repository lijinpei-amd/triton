// RUN: triton-opt %s --tritonamdgpu-block-pingpong=num-stages=3 | FileCheck %s
// RUN: triton-opt %s --tritonamdgpu-block-pingpong=num-stages=3 --tritonamdgpu-update-async-wait-count=arch-generation-name=gfx950 | FileCheck %s

// BlockPingpong's transformTwoClusterWithLocalLoadAndAll combines per-operand
// ttg.async_waits into a single multi-token wait. The Pipeline pass's
// updateWaits already set each input wait's `num` to the in-flight commit-
// group count it can tolerate; the merged wait must preserve the *minimum*
// of those counts so the pipeline isn't accidentally serialized.
//
// On asyncmark targets (CDNA3/CDNA4) this `num` lowers straight to
// rocdl.wait.asyncmark(N), and UpdateAsyncWaitCount is a no-op since PR #9883
// - so whatever num BlockPingpong writes is what reaches the hardware. The
// second RUN line confirms UpdateAsyncWaitCount leaves the wait untouched.

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_ns3_gemm_pingpong_multi_token
  // The two per-operand waits in the input carry num=2 and num=1; pingpong
  // fuses them into a single multi-token wait that preserves min(2, 1) = 1.
  // CHECK: scf.for
  // CHECK: ttg.async_wait %{{[^,]+}}, %{{[^,]+}} {num = 1 : i32}
  tt.func public @async_ns3_gemm_pingpong_multi_token(
      %arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32},
      %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32},
      %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32},
      %arg9: i32,
      %arg10: tensor<256x32x!tt.ptr<bf16>, #blocked>, %arg11: tensor<32x256x!tt.ptr<bf16>, #blocked1>,
      %arg12: !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>,
      %arg13: !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>,
      %arg14: !ttg.async.token,
      %arg15: !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>,
      %arg16: !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>,
      %arg17: !ttg.async.token, %arg18: !ttg.async.token, %arg19: !ttg.async.token,
      %arg20: tensor<256x32xi32, #blocked>, %arg21: tensor<32x256xi32, #blocked1>,
      %arg22: !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable>,
      %arg23: !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable>,
      %arg24: tensor<256x256x!tt.ptr<bf16>, #mma>,
      %arg25: tensor<256x256xi1, #mma>) {
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:12 = scf.for %arg26 = %c0_i32 to %arg9 step %c1_i32 iter_args(%arg27 = %cst, %arg28 = %arg10, %arg29 = %arg11, %arg30 = %c1_i32, %arg31 = %arg12, %arg32 = %arg13, %arg33 = %arg14, %arg34 = %arg15, %arg35 = %arg16, %arg36 = %arg17, %arg37 = %arg18, %arg38 = %arg19) -> (tensor<256x256xf32, #mma>, tensor<256x32x!tt.ptr<bf16>, #blocked>, tensor<32x256x!tt.ptr<bf16>, #blocked1>, i32, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %4 = tt.addptr %arg28, %arg20 : tensor<256x32x!tt.ptr<bf16>, #blocked>, tensor<256x32xi32, #blocked>
      %5 = tt.addptr %arg29, %arg21 : tensor<32x256x!tt.ptr<bf16>, #blocked1>, tensor<32x256xi32, #blocked1>
      %6 = arith.addi %arg30, %c1_i32 : i32
      %7 = arith.cmpi slt, %6, %c3_i32 : i32
      %8 = arith.select %7, %6, %c0_i32 : i32
      %9 = ttg.memdesc_index %arg22[%8] : !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>
      %10 = ttg.async_copy_global_to_local %4, %9 : tensor<256x32x!tt.ptr<bf16>, #blocked> -> <256x32xbf16, #shared, #smem, mutable>
      %11 = ttg.async_commit_group tokens %10
      %12 = ttg.local_load %arg31 token %arg33 : !ttg.memdesc<256x32xbf16, #shared, #smem, mutable> -> tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %13 = ttg.memdesc_index %arg23[%8] : !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>
      %14 = ttg.async_copy_global_to_local %5, %13 : tensor<32x256x!tt.ptr<bf16>, #blocked1> -> <32x256xbf16, #shared1, #smem, mutable>
      %15 = ttg.async_commit_group tokens %14
      %16 = ttg.local_load %arg34 token %arg36 : !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %17 = tt.dot %12, %16, %arg27 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x256xf32, #mma>
      %18 = ttg.async_wait %arg37 {num = 2 : i32}
      %19 = ttg.async_wait %arg38 {num = 1 : i32}
      scf.yield %17, %4, %5, %8, %arg32, %9, %18, %arg35, %13, %19, %11, %15 : tensor<256x256xf32, #mma>, tensor<256x32x!tt.ptr<bf16>, #blocked>, tensor<32x256x!tt.ptr<bf16>, #blocked1>, i32, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    %1 = ttg.async_wait %0#10 {num = 0 : i32}
    %2 = ttg.async_wait %0#11 {num = 0 : i32}
    ttg.local_dealloc %arg22 : !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable>
    ttg.local_dealloc %arg23 : !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable>
    %3 = arith.truncf %0#0 : tensor<256x256xf32, #mma> to tensor<256x256xbf16, #mma>
    tt.store %arg24, %3, %arg25 : tensor<256x256x!tt.ptr<bf16>, #mma>
    tt.return
  }
}
