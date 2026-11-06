// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="gfx-arch=gfx950" --canonicalize --cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#unpacked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
// Four fp32 per thread and 32 values per scale: a thread's registers span two
// scale blocks, so only the CDNA4 pair rule accepts it.
#in4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#out2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#scale32 = #ttg.linear<{register = [[0, 8]], lane = [[0, 0], [0, 0], [0, 0], [0, 1], [0, 2], [0, 4]], warp = [], block = []}>
// Register basis 1 is zero: a thread holds 16 registers but 8 distinct values,
// and register bases 2 and 3 step into other scale blocks.
#bcast = #ttg.linear<{register = [[0, 1], [0, 0], [0, 128], [0, 256]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], warp = [], block = []}>
#bcast_packed = #ttg.linear<{register = [[0, 0], [0, 64], [0, 128]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [], block = []}>
#bcast_scale = #ttg.linear<{register = [[0, 4], [0, 8]], lane = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 2]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @scaled_downcast_fp4(
      %input: tensor<1x512xbf16, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x256x!tt.ptr<i8>, #blocked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp4
    // CHECK: %[[CVT0:.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 {{.*}} -> {{.*}}[0] : i32
    // CHECK: %[[CVT1:.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 {{.*}} -> %[[CVT0]][1] : i32
    // CHECK: %[[CVT2:.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 {{.*}} -> %[[CVT1]][2] : i32
    // CHECK: %[[CVT3:.+]] = rocdl.cvt.scalef32.pk.fp4.bf16 {{.*}} -> %[[CVT2]][3] : i32
    // CHECK: llvm.lshr %[[CVT3]]
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x512xbf16, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x256xi8, #blocked>
    tt.store %output, %result : tensor<1x256x!tt.ptr<i8>, #blocked>
    tt.return
  }

  tt.func public @scaled_downcast_fp16(
      %input: tensor<1x512xf16, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x256x!tt.ptr<i8>, #blocked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp16
    // CHECK: rocdl.cvt.scalef32.pk.fp4.f16
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x512xf16, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x256xi8, #blocked>
    tt.store %output, %result : tensor<1x256x!tt.ptr<i8>, #blocked>
    tt.return
  }

  tt.func public @scaled_downcast_fp32(
      %input: tensor<1x512xf32, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x256x!tt.ptr<i8>, #blocked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp32
    // CHECK: rocdl.cvt.scalef32.pk.fp4.f32
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x512xf32, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x256xi8, #blocked>
    tt.store %output, %result : tensor<1x256x!tt.ptr<i8>, #blocked>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_e4m3_bf16(
      %input: tensor<1x512xbf16, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x512x!tt.ptr<f8E4M3FN>, #unpacked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_e4m3_bf16
    // CHECK: %[[CVT0:.+]] = rocdl.cvt.scalef32.pk.fp8.bf16 {{.*}} -> {{.*}}[false] : vector<2xi16>
    // CHECK: %[[CVT1:.+]] = rocdl.cvt.scalef32.pk.fp8.bf16 {{.*}} -> %[[CVT0]][true] : vector<2xi16>
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x512xbf16, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x512xf8E4M3FN, #unpacked>
    tt.store %output, %result : tensor<1x512x!tt.ptr<f8E4M3FN>, #unpacked>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_e5m2_f16(
      %input: tensor<1x512xf16, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x512x!tt.ptr<f8E5M2>, #unpacked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_e5m2_f16
    // CHECK: rocdl.cvt.scalef32.pk.bf8.f16
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x512xf16, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x512xf8E5M2, #unpacked>
    tt.store %output, %result : tensor<1x512x!tt.ptr<f8E5M2>, #unpacked>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_e4m3_f32(
      %input: tensor<1x512xf32, #unpacked>,
      %scale: tensor<1x64xi8, #scale>,
      %output: tensor<1x512x!tt.ptr<f8E4M3FN>, #unpacked>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_e4m3_f32
    // CHECK: rocdl.cvt.scalef32.pk.fp8.f32
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x512xf32, #unpacked>, tensor<1x64xi8, #scale> -> tensor<1x512xf8E4M3FN, #unpacked>
    tt.store %output, %result : tensor<1x512x!tt.ptr<f8E4M3FN>, #unpacked>
    tt.return
  }

  tt.func public @scaled_downcast_fp4_pair_groups(
      %input: tensor<1x512xf32, #in4>,
      %scale: tensor<1x16xi8, #scale32>,
      %output: tensor<1x256x!tt.ptr<i8>, #out2>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp4_pair_groups
    // CHECK-COUNT-4: rocdl.cvt.scalef32.pk.fp4.f32
    // CHECK-NOT: rocdl.cvt.scalef32.pk.fp4.f32
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x512xf32, #in4>, tensor<1x16xi8, #scale32> -> tensor<1x256xi8, #out2>
    tt.store %output, %result : tensor<1x256x!tt.ptr<i8>, #out2>
    tt.return
  }

  tt.func public @scaled_downcast_fp8_pair_groups(
      %input: tensor<1x512xf32, #in4>,
      %scale: tensor<1x16xi8, #scale32>,
      %output: tensor<1x512x!tt.ptr<f8E4M3FN>, #in4>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_pair_groups
    // CHECK-COUNT-4: rocdl.cvt.scalef32.pk.fp8.f32
    // CHECK-NOT: rocdl.cvt.scalef32.pk.fp8.f32
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x512xf32, #in4>, tensor<1x16xi8, #scale32> -> tensor<1x512xf8E4M3FN, #in4>
    tt.store %output, %result : tensor<1x512x!tt.ptr<f8E4M3FN>, #in4>
    tt.return
  }

  // Each of the 4 distinct packed bytes lies in a different scale block, so
  // each conversion takes a different scale register.
  tt.func public @scaled_downcast_fp4_broadcast_register(
      %input: tensor<1x512xf32, #bcast>,
      %scale: tensor<1x16xi8, #bcast_scale>,
      %output: tensor<1x256x!tt.ptr<i8>, #bcast_packed>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp4_broadcast_register
    // CHECK-DAG: %[[S0:.*]] = llvm.extractvalue %arg1[0]
    // CHECK-DAG: %[[S1:.*]] = llvm.extractvalue %arg1[1]
    // CHECK-DAG: %[[S2:.*]] = llvm.extractvalue %arg1[2]
    // CHECK-DAG: %[[S3:.*]] = llvm.extractvalue %arg1[3]
    // CHECK: llvm.zext %[[S0]] : i8 to i32
    // CHECK: %[[F0:.*]] = llvm.bitcast %{{.*}} : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.fp4.f32 %{{.*}}, %{{.*}}, %[[F0]] -> %{{.*}}[0]
    // CHECK: llvm.zext %[[S1]] : i8 to i32
    // CHECK: %[[F1:.*]] = llvm.bitcast %{{.*}} : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.fp4.f32 %{{.*}}, %{{.*}}, %[[F1]] -> %{{.*}}[1]
    // CHECK: llvm.zext %[[S2]] : i8 to i32
    // CHECK: %[[F2:.*]] = llvm.bitcast %{{.*}} : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.fp4.f32 %{{.*}}, %{{.*}}, %[[F2]] -> %{{.*}}[2]
    // CHECK: llvm.zext %[[S3]] : i8 to i32
    // CHECK: %[[F3:.*]] = llvm.bitcast %{{.*}} : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.fp4.f32 %{{.*}}, %{{.*}}, %[[F3]] -> %{{.*}}[3]
    %result = amdg.scaled_downcast_fp4 %input scale %scale {axis = 1 : i32} : tensor<1x512xf32, #bcast>, tensor<1x16xi8, #bcast_scale> -> tensor<1x256xi8, #bcast_packed>
    tt.store %output, %result : tensor<1x256x!tt.ptr<i8>, #bcast_packed>
    tt.return
  }

  // Each of the 4 distinct value pairs lies in a different scale block.
  tt.func public @scaled_downcast_fp8_broadcast_register(
      %input: tensor<1x512xf32, #bcast>,
      %scale: tensor<1x16xi8, #bcast_scale>,
      %output: tensor<1x512x!tt.ptr<f8E4M3FN>, #bcast>) {
    // CHECK-LABEL: llvm.func @scaled_downcast_fp8_broadcast_register
    // CHECK-DAG: %[[S0:.*]] = llvm.extractvalue %arg1[0]
    // CHECK-DAG: %[[S1:.*]] = llvm.extractvalue %arg1[1]
    // CHECK-DAG: %[[S2:.*]] = llvm.extractvalue %arg1[2]
    // CHECK-DAG: %[[S3:.*]] = llvm.extractvalue %arg1[3]
    // CHECK: llvm.zext %[[S0]] : i8 to i32
    // CHECK: %[[F0:.*]] = llvm.bitcast %{{.*}} : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.fp8.f32 %{{.*}}, %{{.*}}, %[[F0]] -> %{{.*}}[false]
    // CHECK: llvm.zext %[[S1]] : i8 to i32
    // CHECK: %[[F1:.*]] = llvm.bitcast %{{.*}} : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.fp8.f32 %{{.*}}, %{{.*}}, %[[F1]] -> %{{.*}}[true]
    // CHECK: llvm.zext %[[S2]] : i8 to i32
    // CHECK: %[[F2:.*]] = llvm.bitcast %{{.*}} : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.fp8.f32 %{{.*}}, %{{.*}}, %[[F2]] -> %{{.*}}[false]
    // CHECK: llvm.zext %[[S3]] : i8 to i32
    // CHECK: %[[F3:.*]] = llvm.bitcast %{{.*}} : i32 to f32
    // CHECK: rocdl.cvt.scalef32.pk.fp8.f32 %{{.*}}, %{{.*}}, %[[F3]] -> %{{.*}}[true]
    %result = amdg.scaled_downcast_fp8 %input scale %scale {axis = 1 : i32} : tensor<1x512xf32, #bcast>, tensor<1x16xi8, #bcast_scale> -> tensor<1x512xf8E4M3FN, #bcast>
    tt.store %output, %result : tensor<1x512x!tt.ptr<f8E4M3FN>, #bcast>
    tt.return
  }
}
