// RUN: triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=arch-generation-name=gfx950 | FileCheck %s

// On asyncmark targets (CDNA3/CDNA4) UpdateAsyncWaitCount delegates to
// mlir::triton::updateWaits, which keeps ttg.async_wait in place and
// rewrites its `num` to a commit-group count. The lowering then emits
// wait.asyncmark(num) directly.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: single_token_two_crossed
  // Wait on %1: 2 commit groups (%3, %5) lie between %1's producer and the wait.
  // CHECK: ttg.async_wait %{{[^,]+}} {num = 2 : i32}
  tt.func public @single_token_two_crossed(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_wait %1 {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: single_token_none_crossed
  // Wait on %5 (the most-recent commit): no other commit groups between.
  // CHECK: ttg.async_wait %{{[^,]+}} {num = 0 : i32}
  tt.func public @single_token_none_crossed(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_wait %5 {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: multi_token_min
  // Wait on (%1, %3): %1 crosses 2 commits (%3, %5); %3 crosses 1 (%5). min = 1.
  // CHECK: ttg.async_wait %{{[^,]+}}, %{{[^,]+}} {num = 1 : i32}
  tt.func public @multi_token_min(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    %4 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %5 = ttg.async_commit_group tokens %4
    %6 = ttg.async_wait %1, %3 {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tokenless_wait_preserved
  // Tokenless wait carries a producer-authored num that updateWaits can't
  // derive from a def chain - leave it alone.
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
  // CHECK-LABEL: loop_carried_token
  // Token threaded through scf.for iter arg: walk both the init-arg path
  // (from the prologue commit) and the previous-iteration path (from the
  // commit yielded at end of body). Each path crosses 0 commits between
  // the producer and the in-body wait, so num=0.
  // CHECK: scf.for
  // CHECK: ttg.async_wait %{{[^,]+}} {num = 0 : i32}
  tt.func public @loop_carried_token(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %lb: i32, %ub: i32, %step: i32) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = scf.for %iv = %lb to %ub step %step iter_args(%tok = %1) -> (!ttg.async.token) : i32 {
      %w = ttg.async_wait %tok {num = 0 : i32}
      %c = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %g = ttg.async_commit_group tokens %c
      scf.yield %g : !ttg.async.token
    }
    tt.return
  }
}

// -----

// Conservatism characterization tests. minNumInterleavedCommitOps walks
// siblings via getNextNode() and intentionally skips into block ops' children
// (WGMMAPipeline.cpp:67-68). The AMD-side helper deduceMinCountBetweeOps
// (Utility.cpp:21-52) descends into scf.if (taking min across then/else) and
// scf.for (multiplying by static trip count). On these shapes updateWaits
// underestimates the in-flight count - safe (over-conservative wait) but
// suboptimal. The CHECKs below pin the current conservative output; tighten
// them if minNumInterleavedCommitOps is improved.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: conservative_skip_scf_if
  // Sibling commit %3 contributes 1; the scf.if branches each carry a commit
  // but the body is skipped. AMD's helper would add min(1,1)=1 for the if,
  // yielding num=2.
  // CHECK: ttg.async_wait %{{[^,]+}} {num = 1 : i32}
  tt.func public @conservative_skip_scf_if(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %cond: i1) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    scf.if %cond {
      %a = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %g = ttg.async_commit_group tokens %a
    } else {
      %a = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %g = ttg.async_commit_group tokens %a
    }
    %w = ttg.async_wait %1 {num = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: conservative_skip_nested_wait
  // The wait sits inside scf.if, but the token is defined in the parent region.
  // minNumInterleavedCommitOps's region-peeling loop walks `cursor` up to the
  // scf.if without counting the two commits (%b, %d) that precede the wait
  // inside the if-body. AMD's deduceMinCountOnDefChain calls
  // deduceMinCountBetweeOps(&block-front, consumer, ...) before peeling, so it
  // would count those siblings and yield num=2.
  // CHECK: ttg.async_wait %{{[^,]+}} {num = 0 : i32}
  tt.func public @conservative_skip_nested_wait(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %cond: i1) {
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    scf.if %cond {
      %a = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %b = ttg.async_commit_group tokens %a
      %c = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %d = ttg.async_commit_group tokens %c
      %w = ttg.async_wait %1 {num = 0 : i32}
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: conservative_skip_scf_for
  // Sibling commit %3 contributes 1; the scf.for body has 1 commit per iter
  // and a static trip count of 4, but the body is skipped. AMD's helper would
  // add 4*1=4 for the for, yielding num=5.
  // CHECK: ttg.async_wait %{{[^,]+}} {num = 1 : i32}
  tt.func public @conservative_skip_scf_for(%arg0: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg1: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %0 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2
    scf.for %i = %c0 to %c4 step %c1 : i32 {
      %a = ttg.async_copy_global_to_local %arg1, %arg0 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      %g = ttg.async_commit_group tokens %a
    }
    %w = ttg.async_wait %1 {num = 0 : i32}
    tt.return
  }
}
