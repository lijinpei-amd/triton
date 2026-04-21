// RUN: triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=arch-generation-name=gfx950 | FileCheck %s

// The asyncmark wait-count derivation lives in
// third_party/amd/lib/TritonAMDGPUTransforms/Pipeline.cpp (call to
// mlir::triton::updateWaits) and BlockPingpong.cpp (min of input nums on
// fusion). UpdateAsyncWaitCount stays a no-op for CDNA3/CDNA4 since PR #9883,
// so this lit test - which only invokes --tritonamdgpu-update-async-wait-count
// - simply pins that no-op behavior. Each token-bearing case below seeds the
// wait with `num = 7`, a value that derivation could never produce on the
// given def chain (the real counts are 0, 1, or 2). The CHECK lines verifying
// `num = 7` survives prove the pass left it alone. The actual pipeline-time
// analysis is exercised by amd-pipeline-asyncmark-wait-num.mlir.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: single_token_two_crossed
  // Wait on %1 with %3 and %5 between - derivation would yield num=2; sentinel
  // num=7 must survive.
  // CHECK: ttg.async_wait %{{[^,]+}} {num = 7 : i32}
  tt.func public @single_token_two_crossed(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_wait %1 {num = 7 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: single_token_none_crossed
  // Wait on %5 (most-recent commit), no commits between - derivation would
  // yield num=0; sentinel num=7 must survive.
  // CHECK: ttg.async_wait %{{[^,]+}} {num = 7 : i32}
  tt.func public @single_token_none_crossed(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_wait %5 {num = 7 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: multi_token_min
  // Wait on %1, %3 - per-token counts are 2 and 1, so derivation would yield
  // min=1; sentinel num=7 must survive.
  // CHECK: ttg.async_wait %{{[^,]+}}, %{{[^,]+}} {num = 7 : i32}
  tt.func public @multi_token_min(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_wait %1, %3 {num = 7 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tokenless_wait_preserved
  // Tokenless wait carries a producer-authored num that derivation cannot
  // recover from a def chain - num=3 stays put.
  // CHECK: ttg.async_wait {num = 3 : i32}
  tt.func public @tokenless_wait_preserved(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    ttg.async_wait {num = 3 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tokenless_wait_with_surrounding_commits
  // Even with commit groups straddling the tokenless wait, no walk happens
  // (no operand) - sentinel num=5 must survive.
  // CHECK: ttg.async_wait {num = 5 : i32}
  tt.func public @tokenless_wait_with_surrounding_commits(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    ttg.async_wait {num = 5 : i32}
    %4 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: loop_carried_token
  // Token threaded through scf.for iter arg: each path crosses 0 commits, so
  // derivation would yield num=0; sentinel num=7 must survive.
  // CHECK: scf.for
  // CHECK: ttg.async_wait %{{[^,]+}} {num = 7 : i32}
  tt.func public @loop_carried_token(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %lb: i32, %ub: i32, %step: i32) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = scf.for %iv = %lb to %ub step %step iter_args(%tok = %1) -> (!ttg.async.token) : i32 {
      %w = ttg.async_wait %tok {num = 7 : i32}
      %c = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %g = ttg.async_commit_group tokens %c
      scf.yield %g : !ttg.async.token
    }
    tt.return
  }
}

