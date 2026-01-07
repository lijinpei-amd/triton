// RUN: triton-opt --pass-pipeline 'any(tritonamdgpu-schedule-loops{num_stages=4},tritonamdgpu-pipeline{use_async_copy=true use_pingpong=true})' %s | FileCheck %s
// CHECK-not: ttg.async_copy_global_to_local
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [2, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [8, 1], instrShape = [32, 32, 2], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @vec_copy(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %6 = tt.broadcast %1 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %7 = tt.broadcast %4 : tensor<32x1xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<32x128xi32, #blocked>
    %9 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %10 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %11 = tt.expand_dims %9 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x32xi32, #blocked2>
    %12 = tt.expand_dims %10 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %15 = tt.broadcast %11 : tensor<1x32xi32, #blocked2> -> tensor<128x32xi32, #blocked2>
    %16 = tt.broadcast %14 : tensor<128x1xi32, #blocked2> -> tensor<128x32xi32, #blocked2>
    %17 = arith.addi %15, %16 : tensor<128x32xi32, #blocked2>
    %18 = tt.broadcast %12 : tensor<1x32xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %19 = tt.broadcast %5 : tensor<32x1xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %20 = arith.addi %18, %19 : tensor<32x32xi32, #blocked1>
    scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32  : i32 {
      %21 = arith.muli %arg3, %c4096_i32 : i32
      %22 = tt.addptr %arg0, %21 : !tt.ptr<f32>, i32
      %23 = tt.splat %22 : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>, #blocked>
      %24 = tt.addptr %23, %8 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %25 = tt.load %24 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %26 = tt.splat %22 : !tt.ptr<f32> -> tensor<128x32x!tt.ptr<f32>, #blocked2>
      %27 = tt.addptr %26, %17 : tensor<128x32x!tt.ptr<f32>, #blocked2>, tensor<128x32xi32, #blocked2>
      %28 = tt.load %27 : tensor<128x32x!tt.ptr<f32>, #blocked2>
      %29 = ttg.convert_layout %25 : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %30 = ttg.convert_layout %28 : tensor<128x32xf32, #blocked2> -> tensor<128x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %31 = tt.dot %29, %30, %cst : tensor<32x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %32 = arith.muli %arg3, %c1024_i32 : i32
      %33 = tt.addptr %arg1, %32 : !tt.ptr<f32>, i32
      %34 = tt.splat %33 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
      %35 = tt.addptr %34, %20 : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
      %36 = ttg.convert_layout %35 : tensor<32x32x!tt.ptr<f32>, #blocked1> -> tensor<32x32x!tt.ptr<f32>, #mma>
      tt.store %36, %31 : tensor<32x32x!tt.ptr<f32>, #mma>
    } {tt.num_stages = 4 : i32}
    tt.return
  }
}
