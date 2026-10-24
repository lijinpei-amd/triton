#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "LDSEncoding.h"
#include "StreamOps.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "amd/lib/TritonAMDGPUTransforms/Pipeliner/PipelineUtility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/CGAEncodingAttr.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "llvm/Support/Debug.h"
#include <numeric>

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-pipeline-lower-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using mlir::triton::amdgpu::TargetFeatures;

//===----------------------------------------------------------------------===//
// This file will conditionally allocate lds memory, create local/async load
// operations, and create schedule for these operations. After lowerLoops,
// schedule will be passed to expandLoops and eventually to PipelineExpander.
//
// LDS encodings are chosen in LDSEncoding.h. The ops that replace each load
// are built in StreamOps.h. This file decides which ops to build and where
// they go in the schedule.
//===----------------------------------------------------------------------===//

using mlir::triton::AMD::AttrBypassLDS;

namespace mlir {
namespace {

void scheduleLocalLoad(ttg::LocalLoadOp localLoadOp,
                       tt::CoarseSchedule &schedule, int stage,
                       const tt::CoarseSchedule::Cluster &cluster) {
  schedule.insert(localLoadOp, stage, cluster);
  // If its only user is a ConvertLayout, we place it into the same stage so
  // it can be folded by a later pass
  if (localLoadOp->hasOneUse()) {
    auto cvt = *localLoadOp->getUsers().begin();
    if (isa<ttg::ConvertLayoutOp>(cvt)) {
      schedule.insert(cvt, stage, cluster);
    }
  }
}

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers.
LoadToStreamOpMap
createStreamOps(const LoadToInfoMap &loadToInfo, scf::ForOp &forOp,
                const int &numBuffers, bool useAsyncCopy,
                tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  IRRewriter builder(forOp);
  Location loc = forOp.getLoc();
  Value minusOne = arith::ConstantIntOp::create(builder, loc, -1, 32);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  Value extractIdx = minusOne;
  Value numBuffersVal =
      arith::ConstantIntOp::create(builder, loc, numBuffers, 32);

  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependency.
  forOp = addIterArgsToLoop(builder, forOp, {extractIdx});

  // Create one counter for the extract indices to avoid creating long
  // live range.
  extractIdx = forOp.getBody()->getArgument(newOperandIndex);

  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  extractIdx = arith::AddIOp::create(builder, loc, extractIdx, one);
  Value cndExt = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                       extractIdx, numBuffersVal);
  extractIdx = arith::SelectOp::create(builder, loc, cndExt, extractIdx, zero);

  // Patch the yield with the updated counter.
  appendToForOpYield(forOp, {extractIdx});

  LoadToStreamOpMap loadToStreamOp;
  for (auto &[op, info] : loadToInfo) {
    if (!info.sharedEncoding)
      continue;

    auto loadOp = dyn_cast<tt::LoadOp>(op);
    auto descLoadOp = dyn_cast<tt::DescriptorLoadOp>(op);
    auto gatherOp = dyn_cast<tt::DescriptorGatherOp>(op);
    if (!loadOp && !descLoadOp && !gatherOp)
      continue;

    // Create an allocation that can hold distance number of loadOp shapes.
    auto ty = cast<RankedTensorType>(op->getResultTypes()[0]);
    Value alloc = triton::createAlloc(forOp, ty, op->getLoc(),
                                      info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    triton::AMD::TargetInfo targetInfo(
        getAMDArch(op->getParentOfType<ModuleOp>()));

    // Replace the old load with multi-buffered loads
    if (descLoadOp) {
      loadToStreamOp[op] = createTDMAsyncCopy(descLoadOp, alloc, extractIdx);
      continue;
    }
    if (gatherOp) {
      loadToStreamOp[op] = createTDMAsyncGather(gatherOp, alloc, extractIdx);
      continue;
    }

    if (useAsyncCopy &&
        canBeConvertedToAsyncLoad(numBuffers, loadOp, info.sharedEncoding,
                                  axisInfoAnalysis, targetInfo)) {
      unsigned vec = axisInfoAnalysis.getContiguity(loadOp.getPtr());
      if (auto mask = loadOp.getMask())
        vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));
      loadToStreamOp[loadOp] = createAsyncCopy(loadOp, alloc, extractIdx, vec);
      continue;
    }

    loadToStreamOp[loadOp] = createStreamCopy(loadOp, alloc, extractIdx);
  }

  return loadToStreamOp;
}

} // namespace

namespace SingleDotSchedule {
namespace {
using namespace mlir::SingleDotSchedule;
using ClusterMap = DenseMap<tt::CoarseSchedule::ClusterHash, int>;

ClusterMap createClusterMap(tt::CoarseSchedule &schedule) {
  DenseMap<tt::CoarseSchedule::ClusterHash, int> clusterMap;
  for (auto &[op, stageAndCluster] : schedule.opToStageAndCluster) {
    auto [stage, cluster] = stageAndCluster;
    tt::CoarseSchedule::ClusterHash clusterHash =
        tt::CoarseSchedule::hashCluster(cluster);
    clusterMap[clusterHash] = *cluster;
  }

  return clusterMap;
}

// Remap global and compute clusters to the right place
void remapClusters(tt::CoarseSchedule &schedule, ClusterMap clusterMap,
                   Clusters &clusters) {
  for (auto &[op, stageAndCluster] : schedule.opToStageAndCluster) {
    tt::CoarseSchedule::ClusterHash clusterHash =
        tt::CoarseSchedule::hashCluster(stageAndCluster.second);
    int oldClusterId = clusterMap[clusterHash];
    if (oldClusterId == 0) {
      stageAndCluster.second = clusters[SCHED_GLOBAL_LOAD];
    } else {
      assert(oldClusterId == 1);
      stageAndCluster.second = clusters[SCHED_COMPUTE];
    }
  }
}

// Init Schedule Config based on settings and loop characteristics.
// Create clusters in order of ops in loop. This can interleave ops
// from different stages in the same cluster to achieve better backend
// scheduling.
//   WARNING: Changing the order of schedule.clusters.newAtBack() calls
//            can cause invalid schedules to be produced.
LogicalResult initSchedule(int maxDist, Stages &stages, int numStages,
                           int &numBuffers, bool useAsyncCopy, bool hasTDMLoad,
                           bool hasScaledDot, bool waitAtTail,
                           Clusters &clusters, tt::CoarseSchedule &schedule) {
  LDBG("Init SingleDotSchedule");
  int lastStage = numStages - 1;
  stages[SCHED_GLOBAL_LOAD] = 0;
  stages[SCHED_LOCAL_STORE] = 0;
  stages[SCHED_LOCAL_LOAD] = lastStage;
  stages[SCHED_COMPUTE] = lastStage;

  // LDS prefetch of dot operands: schedule ttg.local_load one stage before
  // the tt.dot that consumes it, so the LDS read of the next K-tile overlaps
  // the current tile's matrix op. for num_stages<3 there is no room to separate
  // them.
  // We take a conservative route and only enable LDS prefetch when the dot is
  // not scaled until more testing is done.
  // waitAtTail (pingpong) requires very specific scheduling, which is
  // incompatible with the prefetch cluster ordering below, so disable prefetch
  // in that case.
  bool ldsPrefetch =
      hasTDMLoad && !hasScaledDot && numStages >= 3 && !waitAtTail;
  if (ldsPrefetch) {
    stages[SCHED_LOCAL_LOAD] = lastStage - 1;
  }

  stages[SCHED_ASYNC_WAIT] = stages[SCHED_LOCAL_LOAD];

  bool pairedGlobalLoadLocalStore = stages[SCHED_LOCAL_STORE] == 0;
  stages[SCHED_LOCAL_STORE] += maxDist;
  if (waitAtTail) {
    stages[SCHED_ASYNC_WAIT] = std::max(0, stages[SCHED_LOCAL_LOAD] - 1);
  }

  LDBG(
      "Stage schedule:" << "  GLOBAL_LOAD stage = " << stages[SCHED_GLOBAL_LOAD]
                        << ", LOCAL_STORE stage = " << stages[SCHED_LOCAL_STORE]
                        << ", LOCAL_LOAD stage = " << stages[SCHED_LOCAL_LOAD]
                        << ", COMPUTE stage = " << stages[SCHED_COMPUTE]
                        << ", ASYNC_WAIT stage = " << stages[SCHED_ASYNC_WAIT]
                        << "; total = " << numStages);

  if (stages[SCHED_LOCAL_STORE] >= numStages ||
      stages[SCHED_LOCAL_STORE] > stages[SCHED_LOCAL_LOAD]) {
    LDBG("Invalid stage schedule");
    return failure();
  }

  // Calculate the number of buffers needed for each load.
  // TODO: Use the precise number of buffers needed by the particular load.
  numBuffers =
      std::max(1, stages[SCHED_LOCAL_LOAD] - stages[SCHED_LOCAL_STORE]);
  // If we use AsyncCopy or TDM loads, we need one more buffer since we are not
  // using a register buffer
  if (useAsyncCopy || hasTDMLoad) {
    numBuffers += 1;
  }

  LDBG("deduced max shared memory buffer number = " << numBuffers);

  // We place async wait as the first cluster because we want to have it being
  // the first in the main loop after pipelining.
  // In case we use async_copy with pingpong, we need to place async_wait at
  // the end of the previous iteration, so it can guarantee the correct
  // dependency when warp0 and warp1 are pipelined.
  int asyncWaitCluster = waitAtTail ? 4 : 0;
  // If tt.load and ttg.local_store are in the same stage
  //   spread them apart to allow overlap with compute
  // else
  //   Initiate ttg.local_store before tt.load
  int globalLoadCluster = 1;
  int localStoreCluster = 3;
  if (!pairedGlobalLoadLocalStore) {
    globalLoadCluster = 3;
    localStoreCluster = 2;
  }

  // If ttg.local_load and ttg.local_store are in the same stage
  //   spread them apart to allow overlap with compute
  // else if they share the buffer
  //   ttg.local_load must come first
  // else
  //   schedule ttg.local_load in the middle
  int localLoadCluster = globalLoadCluster;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_LOCAL_STORE]) {
    localLoadCluster = std::max(3, localStoreCluster + 1);
  } else if (numBuffers == 1 && localLoadCluster >= localStoreCluster) {
    // For 1 buffer, ttg.local_load must occur before ttg.local_store
    localLoadCluster = localStoreCluster - 1;
  }

  // Schedule compute with ttg.local_load if paired
  // otherwise, schedule in the middle
  int computeCluster = 2;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_COMPUTE]) {
    computeCluster = localLoadCluster;
  }

  // ldsPrefetch ordering: lead the loop body with the tt.dot (it consumes
  // operands pre-loaded into registers the previous iteration, so it has no
  // intra-iteration dependency), then the async_wait, the next-tile TDM copy,
  // and finally the ttg.local_load that feeds the next iteration.
  if (ldsPrefetch) {
    computeCluster = 0;
    asyncWaitCluster = 1;
    globalLoadCluster = 2;
    localLoadCluster = 3;
    localStoreCluster = 4;
  }

  // TDM copy reordering to improve latency hiding.
  bool tdmReorder = hasTDMLoad && hasScaledDot && !waitAtTail;
  if (tdmReorder) {
    globalLoadCluster = 0; // all TDM copies first
    asyncWaitCluster = 1;  // single wait after the copies
    localLoadCluster = 2;  // local_loads after the wait
    computeCluster = 3;    // dot_scaled last
    localStoreCluster = 4;
  }

  // Create a hash map to associate cluster hash in old schedule with its
  // clusterID
  ClusterMap clusterMap = createClusterMap(schedule);

  // Make assignments
  Clusters clusterVec;
  schedule.clusters.clear();
  std::ranges::generate(clusterVec,
                        [&] { return schedule.clusters.newAtBack(); });

  clusters[SCHED_GLOBAL_LOAD] = clusterVec[globalLoadCluster];
  clusters[SCHED_LOCAL_STORE] = clusterVec[localStoreCluster];
  clusters[SCHED_LOCAL_LOAD] = clusterVec[localLoadCluster];
  clusters[SCHED_COMPUTE] = clusterVec[computeCluster];
  clusters[SCHED_ASYNC_WAIT] = clusterVec[asyncWaitCluster];

  remapClusters(schedule, clusterMap, clusters);

  LDBG("Cluster schedule:" << "  GLOBAL_LOAD cluster = " << globalLoadCluster
                           << ", LOCAL_STORE cluster = " << localStoreCluster
                           << ", LOCAL_LOAD cluster = " << localLoadCluster
                           << ", COMPUTE cluster = " << computeCluster
                           << ", ASYNC_WAIT cluster = " << asyncWaitCluster
                           << "; total = " << SCHED_SIZE);

  return success();
}

void scheduleTDMOps(Operation *tdmOp, Operation *waitOp,
                    ttg::LocalLoadOp maybeLocalLoadOp, Operation *origLoadOp,
                    tt::CoarseSchedule &schedule, const Stages &stages,
                    const Clusters &clusters) {
  auto [loadStage, loadCluster] = schedule[origLoadOp];
  schedule.insert(tdmOp, loadStage, loadCluster);
  // If the LocalLoads are scheduled to a later stage than AsyncCopy we need to
  // place the AsyncCopy prefetches after the AsyncWaits which create a barrier
  // to ensure all warps are finished reading the shared buffer we will write
  // into. This is done by scheduling AsyncWait as the first cluster.
  // If AsyncCopy and LocalLoads are in the same stage we do not assign a
  // schedule so they are placed before the LocalLoads
  if (loadStage != stages[SCHED_LOCAL_LOAD])
    schedule.insert(waitOp, stages[SCHED_ASYNC_WAIT],
                    clusters[SCHED_ASYNC_WAIT]);

  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE]) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void scheduleAsyncCopy(const AsyncCopyChainOps &asyncOps, tt::LoadOp loadOp,
                       tt::CoarseSchedule &schedule, const Stages &stages,
                       const Clusters &clusters) {
  auto [copyOp, commitOp, waitOp, maybeLocalLoadOp] = asyncOps;
  auto [loadStage, loadCluster] = schedule[loadOp];
  schedule.insert(copyOp, loadStage, loadCluster);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, loadStage, loadCluster);
  // If the LocalLoads are scheduled to a later stage than AsyncCopy we need to
  // place the AsyncCopy prefetches after the AsyncWaits which create a barrier
  // to ensure all warps are finished reading the shared buffer we will write
  // into. This is done by scheduling AsyncWait as the first cluster.
  // If AsyncCopy and LocalLoads are in the same stage we do not assign a
  // schedule so they are placed before the LocalLoads
  if (loadStage != stages[SCHED_LOCAL_LOAD])
    schedule.insert(waitOp, stages[SCHED_ASYNC_WAIT],
                    clusters[SCHED_ASYNC_WAIT]);

  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE]) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void scheduleStreamCopy(const StreamCopyChainOps &streamOps,
                        tt::LoadOp oldLoadOp, tt::CoarseSchedule &schedule,
                        const Stages &stages, const Clusters &clusters) {
  auto [newLoadOp, subviewOp, localStoreOp, maybeLocalLoadOp] = streamOps;
  auto [loadStage, loadCluster] = schedule[oldLoadOp];

  schedule.insert(newLoadOp, loadStage, loadCluster);
  schedule.insert(subviewOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  schedule.insert(localStoreOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE]) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void scheduleStreamOps(const LoadToStreamOpMap &loadToStreamOp,
                       tt::CoarseSchedule &schedule, const Stages &stages,
                       const Clusters &clusters) {
  for (auto [l, streamOps] : loadToStreamOp) {
    if (auto asyncOps = std::get_if<TDMChainOps>(&streamOps)) {
      auto [tdmOp, waitOp, localLoad] = *asyncOps;
      scheduleTDMOps(tdmOp, waitOp, localLoad, l, schedule, stages, clusters);
    } else if (auto asyncOps = std::get_if<AsyncCopyChainOps>(&streamOps)) {
      auto loadOp = cast<tt::LoadOp>(l);
      scheduleAsyncCopy(*asyncOps, loadOp, schedule, stages, clusters);
    } else if (auto sOps = std::get_if<StreamCopyChainOps>(&streamOps)) {
      auto loadOp = cast<tt::LoadOp>(l);
      scheduleStreamCopy(*sOps, loadOp, schedule, stages, clusters);
    }
  }
}

// VMEM loads (register loads and async copies to LDS) all retire through the
// in-order vmcnt counter, so a wait for one load also waits for every load
// issued before it. Issue them in the order their data is first needed: an
// async copy is needed at its async_wait, a register load at its first user.
// A register load carried across two or more iterations is rotated through a
// register copy at the next iteration's start, which needs the data too.
using ConsumptionKey = std::tuple<int, int, int, int>;
using Position = std::pair<int, int>; // (cluster index, op order)

// Mode 1 keys a load by (iteration distance, consumer position). Mode 2 first
// buckets consumers by where they sit relative to the dots: before the first
// dot, among the dots, or after the last dot, and treats "after the last dot of
// iteration i" and "before the first dot of iteration i+1" as the same point;
// within a bucket outside the dots, async copies go before register loads.
void orderLoadsByConsumption(scf::ForOp forOp,
                             const LoadToStreamOpMap &loadToStreamOps,
                             tt::CoarseSchedule &schedule,
                             const Clusters &clusters, int mode) {
  Block *body = forOp.getBody();
  DenseMap<Operation *, int> opOrder;
  for (Operation &op : body->without_terminator())
    opOrder[&op] = opOrder.size();
  const int lastStage = schedule.getNumStages() - 1;

  // Stage and position of an op. An op that is not scheduled yet is expected
  // to follow its earliest scheduled (transitive) user, else to end up in the
  // compute cluster of the last stage.
  std::function<std::tuple<int, int, int>(Operation *, int)> place =
      [&](Operation *op, int depth) -> std::tuple<int, int, int> {
    if (schedule.contains(op)) {
      auto [stage, cluster] = schedule[op];
      return {stage, *cluster, opOrder.lookup(op)};
    }
    std::optional<std::tuple<int, int, int>> best;
    if (depth < 8) {
      for (Operation *user : op->getUsers()) {
        Operation *u = body->findAncestorOpInBlock(*user);
        if (!u || u == op || isa<scf::YieldOp>(u))
          continue;
        auto p = place(u, depth + 1);
        if (!best || p < *best)
          best = p;
      }
    }
    if (best)
      return {std::get<0>(*best), std::get<1>(*best), opOrder.lookup(op)};
    return {lastStage, *clusters[SCHED_COMPUTE], opOrder.lookup(op)};
  };

  // Give register loads that are not scheduled yet (consumed in the same
  // iteration) the stage and cluster of their earliest scheduled user, which
  // is where scheduleDependencies would put them, so they can be ordered.
  SmallVector<Operation *> preScheduled;
  for (Operation &op : body->without_terminator()) {
    if (!isa<tt::LoadOp, tt::amdgpu::BufferLoadOp>(op) || op.use_empty() ||
        schedule.contains(&op) || loadToStreamOps.contains(&op))
      continue;
    std::optional<std::tuple<int, int, Operation *>> best;
    SmallVector<std::pair<Operation *, int>> work{{&op, 0}};
    DenseSet<Operation *> seen;
    while (!work.empty()) {
      auto [cur, depth] = work.pop_back_val();
      for (Operation *user : cur->getUsers()) {
        Operation *u = body->findAncestorOpInBlock(*user);
        if (!u || isa<scf::YieldOp>(u) || !seen.insert(u).second)
          continue;
        if (schedule.contains(u)) {
          auto [stage, cluster] = schedule[u];
          std::tuple<int, int, Operation *> cand{stage, *cluster, u};
          if (!best || cand < *best)
            best = cand;
        } else if (depth < 8) {
          work.push_back({u, depth + 1});
        }
      }
    }
    if (best) {
      Operation *u = std::get<2>(*best);
      schedule.insert(&op, schedule[u].first, schedule[u].second);
    } else {
      schedule.insert(&op, lastStage, clusters[SCHED_COMPUTE]);
    }
    preScheduled.push_back(&op);
  }
  auto undoPreSchedule = [&] {
    for (Operation *op : preScheduled)
      schedule.erase(op);
  };

  std::optional<Position> firstDot, lastDot;
  for (Operation &op : body->without_terminator()) {
    if (!isa<tt::DotOp, tt::DotScaledOp>(op))
      continue;
    auto [stage, cluster, idx] = place(&op, 0);
    Position p{cluster, idx};
    if (!firstDot || p < *firstDot)
      firstDot = p;
    if (!lastDot || *lastDot < p)
      lastDot = p;
  }
  auto keyOf = [&](int dist, Position consumer, bool isAsync) -> ConsumptionKey {
    if (mode != 2 || !firstDot)
      return {dist, consumer.first, consumer.second, 0};
    int kind = isAsync ? 0 : 1;
    if (consumer < *firstDot)
      return {2 * dist - 1, 0, 0, kind};
    if (*lastDot < consumer)
      return {2 * dist + 1, 0, 0, kind};
    return {2 * dist, consumer.first, consumer.second, kind};
  };
  // A register load carried over two or more iterations is consumed by a
  // rotation copy at the start of the next iteration.
  const ConsumptionKey rotationKey =
      mode == 2 && firstDot ? ConsumptionKey{1, 0, 0, 1}
                            : ConsumptionKey{1, -1, 0, 0};

  struct Load {
    Operation *op;
    SmallVector<Operation *> group; // ops that move together, op first
    int stage;
    Position pos;
    ConsumptionKey key;
  };
  SmallVector<Load> loads;
  DenseSet<Operation *> replaced;
  // The replaced load stays in the body (dead) with the same operands as its
  // replacement; it must move along, or the expander may find it scheduled
  // before the pointer update it depends on and refuse to pipeline the loop.
  DenseMap<Operation *, Operation *> replacedBy;
  for (auto &[orig, streamOps] : loadToStreamOps) {
    if (auto sOps = std::get_if<StreamCopyChainOps>(&streamOps))
      replacedBy[sOps->loadOp] = orig;
  }
  auto withReplaced = [&](SmallVector<Operation *> group, Operation *orig) {
    if (orig && orig->getBlock() == body && schedule.contains(orig))
      group.push_back(orig);
    return group;
  };
  for (auto &[orig, streamOps] : loadToStreamOps) {
    replaced.insert(orig);
    auto asyncOps = std::get_if<AsyncCopyChainOps>(&streamOps);
    if (!asyncOps || !schedule.contains(asyncOps->waitOp))
      continue;
    Operation *copy = asyncOps->copyOp;
    auto [copyStage, copyCluster, copyIdx] = place(copy, 0);
    auto [waitStage, waitCluster, waitIdx] = place(asyncOps->waitOp, 0);
    loads.push_back({copy,
                     withReplaced({copy, asyncOps->commitOp}, orig),
                     copyStage,
                     {copyCluster, copyIdx},
                     keyOf(waitStage - copyStage, {waitCluster, waitIdx},
                           /*isAsync=*/true)});
  }
  for (Operation &op : body->without_terminator()) {
    if (!isa<tt::LoadOp, tt::amdgpu::BufferLoadOp>(op) || op.use_empty() ||
        replaced.contains(&op))
      continue;
    auto [loadStage, loadCluster, loadIdx] = place(&op, 0);
    std::optional<ConsumptionKey> key;
    for (Operation *user : op.getUsers()) {
      Operation *u = body->findAncestorOpInBlock(*user);
      ConsumptionKey k = rotationKey;
      if (u && !isa<scf::YieldOp>(u)) {
        auto [userStage, userCluster, userIdx] = place(u, 0);
        int dist = userStage - loadStage;
        if (dist < 2)
          k = keyOf(dist, {userCluster, userIdx}, /*isAsync=*/false);
      }
      if (!key || k < *key)
        key = k;
    }
    loads.push_back({&op, withReplaced({&op}, replacedBy.lookup(&op)),
                     loadStage, {loadCluster, loadIdx}, *key});
  }
  if (loads.size() < 2) {
    undoPreSchedule();
    return;
  }

  // Current order of the loads in the loop body, and the wanted order. Equal
  // keys keep their current relative order.
  llvm::stable_sort(loads, [](const Load &a, const Load &b) {
    return a.pos < b.pos;
  });
  SmallVector<int> wanted(loads.size());
  std::iota(wanted.begin(), wanted.end(), 0);
  llvm::stable_sort(wanted, [&](int a, int b) {
    return loads[a].key < loads[b].key;
  });
  if (llvm::equal(wanted, llvm::seq<int>(0, loads.size()))) {
    undoPreSchedule();
    return;
  }

  // Keep the longest run of loads that is already in the wanted order in
  // place, so copies stay interleaved with their local_loads, and move only
  // the others, each right before the next kept load it must precede.
  SmallVector<int> rank(loads.size());
  for (auto [r, i] : llvm::enumerate(wanted))
    rank[i] = r;
  SmallVector<int> tails, prev(loads.size(), -1), tailIdx;
  for (int i = 0, e = loads.size(); i < e; ++i) {
    auto it = llvm::lower_bound(tails, rank[i]);
    int len = it - tails.begin();
    if (it == tails.end()) {
      tails.push_back(rank[i]);
      tailIdx.push_back(i);
    } else {
      *it = rank[i];
      tailIdx[len] = i;
    }
    prev[i] = len ? tailIdx[len - 1] : -1;
  }
  DenseSet<int> kept;
  for (int i = tailIdx.back(); i >= 0; i = prev[i])
    kept.insert(i);

  auto posOf = [&](Operation *op) -> Position {
    auto [stage, cluster, idx] = place(op, 0);
    return {cluster, idx};
  };
  // Puts the group of load `l` in a new cluster right before `anchor`, if
  // every operand defined in the same stage is still available there and no
  // same-stage user would end up before it.
  auto canPlaceAt = [&](Load &l, Position at) {
    for (Operation *op : l.group) {
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def || def->getBlock() != body || !schedule.contains(def) ||
            llvm::is_contained(l.group, def))
          continue;
        if (std::get<0>(place(def, 0)) == l.stage && !(posOf(def) < at))
          return false;
      }
      for (Operation *user : op->getUsers()) {
        Operation *u = body->findAncestorOpInBlock(*user);
        if (!u || llvm::is_contained(l.group, u) || isa<scf::YieldOp>(u))
          continue;
        if (std::get<0>(place(u, 0)) == l.stage && posOf(u) < at)
          return false;
      }
    }
    return true;
  };
  auto moveBefore = [&](Load &l, Operation *anchor) {
    if (!schedule.contains(anchor) || !canPlaceAt(l, posOf(anchor)))
      return false;
    auto cluster = schedule.splitClusterBefore(anchor, forOp);
    auto target = schedule.clusters.newBefore(cluster);
    for (Operation *op : l.group)
      schedule.insert(op, l.stage, target);
    return true;
  };

  // The first scheduled op after every op of `group` in the final order.
  auto nextAfter = [&](ArrayRef<Operation *> group) -> Operation * {
    Position at = posOf(group.front());
    for (Operation *op : group)
      at = std::max(at, posOf(op));
    Operation *next = nullptr;
    std::optional<Position> nextPos;
    for (Operation &o : body->without_terminator()) {
      if (llvm::is_contained(group, &o) || !schedule.contains(&o))
        continue;
      Position p = posOf(&o);
      if (at < p && (!nextPos || p < *nextPos)) {
        nextPos = p;
        next = &o;
      }
    }
    return next;
  };

  Load *lastPlaced = nullptr;
  for (auto [r, i] : llvm::enumerate(wanted)) {
    if (kept.contains(i)) {
      lastPlaced = &loads[i];
      continue;
    }
    // Move right before the next kept load in the wanted order, or, if there
    // is none, right after the load placed before it.
    Operation *anchor = nullptr;
    for (int rr = r + 1, e = wanted.size(); rr < e && !anchor; ++rr)
      if (kept.contains(wanted[rr]))
        anchor = loads[wanted[rr]].op;
    if (!anchor && lastPlaced) {
      anchor = nextAfter(lastPlaced->group);
      if (anchor && llvm::is_contained(loads[i].group, anchor)) {
        SmallVector<Operation *> both(lastPlaced->group);
        both.append(loads[i].group.begin(), loads[i].group.end());
        anchor = nextAfter(both);
      }
    }
    if (anchor && moveBefore(loads[i], anchor)) {
      lastPlaced = &loads[i];
      continue;
    }
    if (!anchor && lastPlaced && schedule.contains(lastPlaced->op) &&
        canPlaceAt(loads[i], {static_cast<int>(schedule.clusters.size()), 0})) {
      auto after = std::next(schedule[lastPlaced->group.back()].second);
      auto target = after == schedule.clusters.end()
                        ? schedule.clusters.newAtBack()
                        : schedule.clusters.newBefore(after);
      for (Operation *op : loads[i].group)
        schedule.insert(op, loads[i].stage, target);
      lastPlaced = &loads[i];
      continue;
    }
    LDBG("Keep load in place: " << *loads[i].op);
  }
}

// Puts an llvm.amdgcn.sched.barrier in front of every VMEM load op of the loop
// (register loads and async copies to LDS), in the same stage and cluster, so
// LLVM keeps the loads in this order while scheduling everything else freely.
// One barrier per op keeps the op's own memory instructions mergeable.
void pinLoadOrder(scf::ForOp forOp, const LoadToStreamOpMap &loadToStreamOps,
                  tt::CoarseSchedule &schedule) {
  DenseSet<Operation *> replaced;
  for (auto &[orig, _] : loadToStreamOps)
    replaced.insert(orig);
  SmallVector<Operation *> loads;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::AsyncCopyGlobalToLocalOp>(op) ||
        (isa<tt::LoadOp, tt::amdgpu::BufferLoadOp>(op) && !op.use_empty() &&
         !replaced.contains(&op)))
      loads.push_back(&op);
  }
  using Mask = ROCDL::SchedGroupMask;
  const Mask crossable = Mask::non_mem_non_sideeffect | Mask::valu |
                         Mask::salu | Mask::mfma_wmma | Mask::vmem_write |
                         Mask::ds_read | Mask::ds_write | Mask::transcendental;
  for (Operation *load : loads) {
    if (!schedule.contains(load))
      continue;
    auto [stage, cluster] = schedule[load];
    OpBuilder builder(load);
    auto barrier =
        ROCDL::SchedBarrier::create(builder, load->getLoc(), crossable);
    schedule.insert(barrier, stage, cluster);
  }
}

void updateSchedule(scf::ForOp &forOp, const LoadToInfoMap &loadToInfo,
                    tt::CoarseSchedule &schedule,
                    triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    bool useAsyncCopy, bool hasTDMLoad, bool waitAtTail) {
  LDBG("SingleDotSchedule::updateSchedule");
  Stages stages;
  Clusters clusters;

  int maxDist = 0;
  for (auto &[l, info] : loadToInfo) {
    maxDist = std::max(maxDist, info.distToUse);
  }

  bool hasScaledDot = false;
  forOp.walk([&](tt::DotScaledOp) { hasScaledDot = true; });

  int numBuffers = 1;
  if (failed(initSchedule(maxDist, stages, schedule.getNumStages(), numBuffers,
                          useAsyncCopy, hasTDMLoad, hasScaledDot, waitAtTail,
                          clusters, schedule)))
    return;

  // Convert the loads into shared memory allocations and loads from them.
  auto loadToStreamOps = createStreamOps(loadToInfo, forOp, numBuffers,
                                         useAsyncCopy, axisInfoAnalysis);

  scheduleStreamOps(loadToStreamOps, schedule, stages, clusters);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Coarse schedule stream ops:");

  // With control flow in the loop body, LLVM schedules each block separately
  // and register pressure across them is not ours to trade, so leave such
  // loops alone.
  bool straightLineBody =
      llvm::none_of(forOp.getBody()->without_terminator(),
                    [](Operation &op) { return op.getNumRegions() > 0; });
  std::string orderMode =
      triton::tools::getStrEnv("TRITON_HIP_LOAD_CONSUMPTION_ORDER");
  if (!orderMode.empty() && orderMode != "0" && !hasTDMLoad &&
      straightLineBody) {
    orderLoadsByConsumption(forOp, loadToStreamOps, schedule, clusters,
                            orderMode == "2" ? 2 : 1);
    dumpScheduleDebug(schedule, DEBUG_TYPE,
                      "Coarse schedule with loads in consumption order:");
  }

  for (auto [l, _] : loadToInfo) {
    if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(l)) {
      schedule.erase(l);
      l->erase();
    }
  }

  scheduleDependencies(forOp, schedule);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Coarse schedule with dependencies:");
  ttg::scheduleDistanceOneDependencies(forOp, schedule);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Coarse schedule with dist 1:");
  tt::CoarseSchedule::Cluster computeCluster = clusters[SCHED_COMPUTE];
  ttg::scheduleRemainingToLastStage(forOp, schedule, computeCluster);
  if (triton::tools::getBoolEnv("TRITON_HIP_PIN_VMEM_LOAD_ORDER") &&
      straightLineBody)
    pinLoadOrder(forOp, loadToStreamOps, schedule);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Final coarse schedule:");
}
} // namespace
} // namespace SingleDotSchedule

namespace ChainedDotSchedule {
namespace {
using namespace mlir::ChainedDotSchedule;

void scheduleAsyncCopy(const AsyncCopyChainOps &asyncOps, tt::LoadOp loadOp,
                       tt::CoarseSchedule &schedule,
                       const ChainedDotClusters &clusters) {
  auto [loadStage, loadCluster] = schedule[loadOp];
  auto [copyOp, commitOp, waitOp, maybeLocalLoadOp] = asyncOps;

  schedule.insert(copyOp, loadStage, loadCluster);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, loadStage, loadCluster);

  if (loadStage == STAGE_GLOBAL_LOAD_1) {
    schedule.insert(waitOp, STAGE_LOCAL_LOAD_1, clusters[CLUSTER_ASYNC_WAIT_1]);
    if (maybeLocalLoadOp)
      scheduleLocalLoad(maybeLocalLoadOp, schedule, STAGE_LOCAL_LOAD_1,
                        clusters[CLUSTER_LOCAL_LOAD_1]);
  } else {
    schedule.insert(waitOp, STAGE_LOCAL_LOAD_2, clusters[CLUSTER_ASYNC_WAIT_2]);
    if (maybeLocalLoadOp)
      scheduleLocalLoad(maybeLocalLoadOp, schedule, STAGE_LOCAL_LOAD_2,
                        clusters[CLUSTER_LOCAL_LOAD_2]);
  }
}

void scheduleStreamCopy(const StreamCopyChainOps &streamOps, tt::LoadOp loadOp,
                        tt::CoarseSchedule &schedule,
                        const ChainedDotClusters &clusters) {
  auto [loadStage, loadCluster] = schedule[loadOp];
  auto [copyOp, subviewOp, localStoreOp, maybeLocalLoadOp] = streamOps;
  schedule.insert(copyOp, loadStage, loadCluster);

  if (loadStage == STAGE_GLOBAL_LOAD_1) {
    schedule.insert(subviewOp, STAGE_LOCAL_WRITE_1,
                    clusters[CLUSTER_LOCAL_WRITE_1]);
    schedule.insert(localStoreOp, STAGE_LOCAL_WRITE_1,
                    clusters[CLUSTER_LOCAL_WRITE_1]);

    if (maybeLocalLoadOp)
      schedule.insert(maybeLocalLoadOp, STAGE_LOCAL_LOAD_1,
                      clusters[CLUSTER_LOCAL_LOAD_1]);
  } else {
    schedule.insert(subviewOp, STAGE_LOCAL_WRITE_2,
                    clusters[CLUSTER_LOCAL_WRITE_2]);
    schedule.insert(localStoreOp, STAGE_LOCAL_WRITE_2,
                    clusters[CLUSTER_LOCAL_WRITE_2]);
    if (maybeLocalLoadOp)
      schedule.insert(maybeLocalLoadOp, STAGE_LOCAL_LOAD_2,
                      clusters[CLUSTER_LOCAL_LOAD_2]);
  }

  if (maybeLocalLoadOp) {
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(
            *maybeLocalLoadOp->getUsers().begin())) {
      auto [localLoadStage, localLoadCluster] = schedule[maybeLocalLoadOp];
      schedule.insert(cvt, localLoadStage, localLoadCluster);
    }
  }
}

void scheduleStreamOps(const LoadToStreamOpMap &loadToStreamOp,
                       tt::CoarseSchedule &schedule,
                       const ChainedDotClusters &clusters) {
  for (auto [l, streamOps] : loadToStreamOp) {
    auto loadOp = dyn_cast<tt::LoadOp>(l);
    if (!loadOp)
      continue;

    if (auto asyncOps = std::get_if<AsyncCopyChainOps>(&streamOps)) {
      scheduleAsyncCopy(*asyncOps, loadOp, schedule, clusters);
    } else if (auto sOps = std::get_if<StreamCopyChainOps>(&streamOps)) {
      scheduleStreamCopy(*sOps, loadOp, schedule, clusters);
    }
  }
}

void updateSchedule(scf::ForOp &forOp, const LoadToInfoMap &loadToInfo,
                    tt::CoarseSchedule &schedule,
                    triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    bool useAsyncCopy) {
  LDBG("ChainedDotSchedule::updateSchedule");
  ChainedDotClusters clusters;
  int cnt = clusters.size() - schedule.clusters.size();
  for (int i = 0; i < cnt; i++) {
    schedule.clusters.newAtBack();
  }
  auto it = schedule.clusters.begin();
  for (int i = 0; i < clusters.size(); i++, it++) {
    clusters[i] = it;
  }

  // Convert the loads into shared memory allocations and loads from them.
  // TODO support different numBuffers
  int numBuffers = useAsyncCopy ? 2 : 1;
  auto loadToStreamOps = createStreamOps(loadToInfo, forOp, numBuffers,
                                         useAsyncCopy, axisInfoAnalysis);
  scheduleStreamOps(loadToStreamOps, schedule, clusters);

  // Now that the original load is replaced by a sequence of new Ops. It become
  // dead, and hence should be removed from the schedule and can be erased.
  for (auto [l, _] : loadToStreamOps) {
    schedule.erase(l);
    l->erase();
  }

  scheduleDependencies(forOp, schedule);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Coarse schedule with dependencies:");

  triton::gpu::scheduleDistanceOneDependencies(forOp, schedule);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Coarse schedule with dist 1:");

  tt::CoarseSchedule::Cluster lastCluster = clusters.back();
  triton::gpu::scheduleRemainingToLastStage(forOp, schedule, lastCluster);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Final coarse schedule:");
}
} // namespace
} // namespace ChainedDotSchedule

static void lowerLoop(scf::ForOp forOp,
                      triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                      bool useAsyncCopy, bool usePingpong) {
  tt::CoarseSchedule schedule;
  if (failed(schedule.deSerialize(forOp, /*normalizeClusterId=*/false))) {
    return;
  }

  dumpScheduleDebug(schedule, DEBUG_TYPE, "[lowerLoops]deserialized schedule:");

  int numStages = schedule.getNumStages();

  // i.e., we can still disable `waitAtTail` by explicitly disabling
  // pingpong, which is the only use case of this scheduling variant.
  bool waitAtTail = usePingpong && (numStages == 3) && useAsyncCopy;

  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel =
      getIndirectLevel(axisInfoAnalysis, forOp, numStages);

  TargetFeatures targetFeatures(getAMDArch(forOp->getParentOfType<ModuleOp>()));
  triton::AMD::TargetInfo targetInfo(
      getAMDArch(forOp->getParentOfType<ModuleOp>()));

  bool hasTDMLoad = false;
  LoadToInfoMap loadToInfo;
  for (const auto &[load, info] : loadOpToIndLevel) {
    auto [distance, use] = info;
    if (load->hasAttrOfType<BoolAttr>(AttrBypassLDS)) {
      load->removeAttr(AttrBypassLDS);
      loadToInfo[load] = {nullptr, distance, use};
    } else {
      if (auto descLoadLike =
              dyn_cast<tt::DescriptorLoadLikeOpInterface>(load)) {
        hasTDMLoad = true;
        auto resultType = cast<RankedTensorType>(load->getResult(0).getType());
        auto paddedEncoding =
            getEncodingFromDescriptor(load, resultType, descLoadLike.getDesc());
        LoadInfo ldInfo;
        ldInfo.sharedEncoding = paddedEncoding;
        ldInfo.distToUse = distance;
        ldInfo.use = use;
        loadToInfo[load] = ldInfo;
      } else {
        LDBG("Deduce shared encoding for: " << *load);
        auto sharedEncoding = getSharedEncIfAllUsersAreDotEnc(
                                  load, axisInfoAnalysis, targetFeatures,
                                  targetInfo, useAsyncCopy)
                                  .value_or(nullptr);
        loadToInfo[load] = {sharedEncoding, distance, use};
        LDBG("Populate loadInfo with shared encoding: " << sharedEncoding);
      }
    }
  }

  if (succeeded(mlir::ChainedDotSchedule::checkPreconditions(forOp, numStages,
                                                             loadToInfo))) {
    ChainedDotSchedule::updateSchedule(forOp, loadToInfo, schedule,
                                       axisInfoAnalysis, useAsyncCopy);
  } else {
    SingleDotSchedule::updateSchedule(forOp, loadToInfo, schedule,
                                      axisInfoAnalysis, useAsyncCopy,
                                      hasTDMLoad, waitAtTail);
  }

  dumpScheduleDebug(schedule, DEBUG_TYPE, "[lowerLoops]updated schedule:");

  schedule.serialize(forOp);
}

void lowerLoops(ModuleOp moduleOp, bool useAsyncCopy, bool usePingpong) {
  triton::AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    lowerLoop(forOp, axisInfoAnalysis, useAsyncCopy, usePingpong);
  }
}

} // namespace mlir
