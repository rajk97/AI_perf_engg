Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters
- This chapter is intra-kernel optimization and next is inter-kernel optimization

- Intra-kernel pipelining = overlap memory and compute WITHIN a single kernel (not across kernels).
- Double buffering: all warps do both load+compute, ping-pong between two shared memory buffers — simpler, works with any warp count.
- Warp specialization: assign warps to distinct roles (loader, compute, storer) — forms a pipeline within one thread block, needs multiple warps.
- Double buffering = uniform work per warp. Warp specialization = dedicated roles per warp.
- Both use CUDA Pipeline API (<cuda/pipeline>) for async coordination without __syncthreads().
- Warp specialization suits persistent kernels with clear stage boundaries; double buffering suits loop-based tiling.

Block level (double buffering):
  ┌─────────────────────────────────┐
  │  Thread Block (all warps)       │
  │                                 │
  │  ALL warps: producer_acquire()  │  ← whole block acts as producer
  │  ALL warps: memcpy_async()      │  ← each thread copies its portion
  │  ALL warps: producer_commit()   │
  │                                 │
  │  ALL warps: consumer_wait()     │  ← whole block acts as consumer
  │  ALL warps: computeTile()       │  ← each thread computes its portion
  │  ALL warps: consumer_release()  │
  └─────────────────────────────────┘
  
  The block as a whole switches between producer and consumer phases.
  Individual warps don't have separate roles.

Warp level (warp specialization):
  ┌─────────────────────────────────┐
  │  Thread Block                   │
  │                                 │
  │  Warp 0: if (warpId == 0)      │  ← dedicated producer warp
  │          load tiles forever     │
  │                                 │
  │  Warp 1: if (warpId == 1)      │  ← dedicated consumer/compute warp
  │          compute tiles forever  │
  │                                 │
  │  Warp 2: if (warpId == 2)      │  ← dedicated storer warp
  │          store results forever  │
  └─────────────────────────────────┘
  
  Each warp has a PERMANENT role for the kernel's lifetime.
  Pipeline stages run IN PARALLEL across warps.

                      Double Buffering              Warp Specialization
Pipeline unit:      Block                         Individual warps
Role assignment:    All warps do everything        Each warp has one role
Overlap source:     Async DMA ‖ compute            Warp 0 loads ‖ Warp 1 computes
Sync mechanism:     Pipeline barriers (block-wide) Pipeline barriers (warp-to-warp)
Code structure:     One code path, all warps       if(warpId==0) load; else compute;

- Nsight Compute’s asynchronous copy metrics
- Kernel execution time decreases and SM Active % increases

Warp Specialization and the Producer-Consumer Model: 

- Warp specialization = assign permanent roles to warps within a thread block: loader, compute, storer.
- All warps share the SAME shared memory on the SAME SM — communication via shared memory buffers + pipeline barriers, not data copying.
- Loader warp: TMA/cp.async moves data HBM → shared memory (no registers involved, async).
- Compute warp: reads tiles from shared memory → processes in private registers (Tensor Core MMA) → writes results back to shared memory.
- Storer warp: reads results from shared memory → writes to HBM.
- After pipeline ramp-up, all three run SIMULTANEOUSLY on different buffers — loader fills buf[N+1], compute processes buf[N], storer writes buf[N-1].
- Advantage over double buffering: different warp schedulers issue load/compute/store instructions in the SAME cycle — true multi-issue parallelism.
- SM's 4 warp schedulers enable this — each scheduler can issue one instruction per cycle from a different warp.

HBM                    Shared Memory              Registers
 │                     (on SM, shared)             (per thread, private)
 │                          │                           │
 │   LOADER WARP            │                           │
 │   ───────────            │                           │
 ├──TMA/cp.async──────────→ │ buf[0]                    │
 │   (async, no regs)       │                           │
 │                          │ ──barrier signal──→       │
 │                          │                           │
 │   COMPUTE WARP           │                           │
 │   ────────────           │                           │
 │                          │ buf[0] ──read──→  regs (accumulators)
 │                          │                   │ Tensor Core MMA
 │                          │                   │ result in regs
 │                          │ ←──write result── regs
 │                          │ result_buf                 │
 │                          │ ──barrier signal──→       │
 │                          │                           │
 │   STORER WARP            │                           │
 │   ───────────            │                           │
 │ ←────store──────────────│ result_buf                 │
 │                          │                           │
 │                          │                           │
 ▼                          ▼                           ▼

 All three happen SIMULTANEOUSLY on different buffers:
 ┌──────────────────────────────────────────────────────┐
 │ Loader:  HBM →→ buf[1]                              │
 │ Compute:        buf[0] →→ regs →→ result_buf[0]     │  SAME CYCLE
 │ Storer:                           result_buf[-1] →→ HBM │
 └──────────────────────────────────────────────────────┘

- ALL warps on an SM share the same shared memory — regardless of which scheduler manages them.
- An SM can have up to 64 warps (Blackwell) resident simultaneously.
- 4 schedulers round-robin across these warps, issuing 1 instruction per scheduler per cycle.
- Shared memory is the communication channel between warps on the same SM.

Normal (no specialization) — every warp does load→compute→store sequentially:

Warp 0: LOAD LOAD LOAD LOAD | COMPUTE COMPUTE COMPUTE | STORE STORE STORE
Warp 1: LOAD LOAD LOAD LOAD | COMPUTE COMPUTE COMPUTE | STORE STORE STORE
Warp 2: LOAD LOAD LOAD LOAD | COMPUTE COMPUTE COMPUTE | STORE STORE STORE

Cycle:  ──────────────────────────────────────────────────────→

Scheduler 0 picks Warp 0: issues LOAD
Scheduler 1 picks Warp 1: issues LOAD
Scheduler 2 picks Warp 2: issues LOAD
                           ↑
                    ALL hitting memory subsystem
                    Compute units IDLE
                    
Then later:
Scheduler 0 picks Warp 0: issues COMPUTE
Scheduler 1 picks Warp 1: issues COMPUTE
Scheduler 2 picks Warp 2: issues COMPUTE
                           ↑
                    ALL hitting compute units
                    Memory subsystem IDLE

Warp specialized — each warp has a DIFFERENT instruction stream:

Warp 0 (loader):  LOAD LOAD LOAD LOAD LOAD LOAD LOAD LOAD LOAD ...
Warp 1 (compute): COMPUTE COMPUTE COMPUTE COMPUTE COMPUTE COMPUTE ...
Warp 2 (storer):  STORE STORE STORE STORE STORE STORE STORE STORE ...

Cycle:  ──────────────────────────────────────────────────────→

EVERY cycle:
Scheduler 0 picks Warp 0: issues LOAD      → memory subsystem busy
Scheduler 1 picks Warp 1: issues COMPUTE   → compute units busy
Scheduler 2 picks Warp 2: issues STORE     → memory subsystem busy
                           ↑
                    BOTH memory AND compute busy SIMULTANEOUSLY
                    No phase where one is idle!

Normal:          Memory busy    Compute busy    Memory busy
                 ████████████   ████████████    ████████████
                 compute idle   memory idle     compute idle
                 ────────────   ────────────    ────────────

Specialized:     Memory busy    Memory busy     Memory busy
                 ████████████   ████████████    ████████████
                 Compute busy   Compute busy    Compute busy
                 ████████████   ████████████    ████████████
                 ↑
                 BOTH pipelines saturated ALL the time

- Warp specialization guarantees memory+compute overlap BY DESIGN — not accidental drift like normal execution.
- Profiling proof: before specialization, L2 bandwidth and Tensor Core util were out-of-phase. After: in-phase → higher throughput.
- CUTLASS exposes three-role ping-pong architecture: dedicated loader warps + two consumer warp sets that alternate compute and store roles.
- Epilogue = post-MMA cleanup (accumulate, scale, write back, advance pointers, signal done). Prologue = pre-fill pipeline with TMA loads before compute starts.
- FlashAttention v3 uses warp specialization to overlap GEMM + softmax + TMA data movement → near-peak FLOPS for attention.
- torch.compile and Triton can generate warp-specialized kernels — but selectively, based on heuristics, not for every op.
- Use warp specialization when compute alone can't hide memory latency. For small/extremely memory-bound kernels, simpler double buffering is good enough.          

Small kernels -- no warp specialization why? 
1. Warp specialization has overhead that small kernels can't amortize
2. Small kernels don't have enough work to keep specialized warps busy

- CUDA Pipeline API enables warp specialization with fine-grained sync — only the warp that NEEDS data waits, others keep running.
- producer_acquire/commit + consumer_wait/release = per-stage handoff, NOT block-wide barrier.
- __syncthreads() stalls ALL warps (even unrelated ones). Pipeline API stalls only the consumer waiting for its specific producer.
- Pipeline object is block-scoped and tracks stage ordering internally — no manual barrier management.
- Three-role pattern: loader → pipe → compute → pipe → storer, each stage connected by producer/consumer handoffs.

__syncthreads():  block-level barrier — ALL warps in the block must arrive before ANY proceed.
Pipeline API:     stage-level barrier — only warps involved in THAT stage wait, others keep running.

- Naive tiling → double buffering = 2× speedup (overlap load+compute). Double buffering → warp specialization = ~10% more (finer sync).
- Warp specialization: 96% SM utilization vs 92% (double buffering) vs 68% (naive) — gains come from eliminating block-wide stalls.
- Warp specialization scales to 64 warps/SM (Blackwell limit). Double buffering saturates at ~6 warps/SM. Naive at 2-3 warps/SM.
- Compute warp acts as BOTH consumer (of loader's data) AND producer (of results for storer) — calls consumer_wait then producer_commit in sequence.
- Double buffering = best for simple tiled GEMMs. Warp specialization = best for irregular/deep pipelines (e.g., fused attention kernels like FlashAttention v3).
- Diminishing returns: the big win is naive→double buffering (2×). Warp specialization adds ~10% on top — worth it only for long-running/persistent kernels.

Double buffering — block-wide stall:
  
  ALL warps: [async load tile 1] → [consumer_wait — ALL STALL] → [compute tile 0] → [consumer_wait — ALL STALL]
                                          ↑
                                    if load takes 400 cycles,
                                    ALL warps idle for 400 cycles
                                    compute units idle, store units idle

Warp specialization — only compute stalls:

  Loader:  [load tile 1][load tile 2][load tile 3]...  ← keeps memory pipe busy
  Compute: [wait 400cy][compute 0][compute 1]...       ← stalls, but only this warp
  Storer:  [store prev][store prev]...                 ← keeps store pipe busy
                ↑
          loader is FILLING the next buffer during compute's stall
          → by the time compute finishes tile 0, tile 1 is ALREADY in shared mem
          → compute never stalls again after ramp-up
The real advantage isn't just "fewer warps stall" — it's that the loader stays ahead

DOUBLE BUFFERING (load takes longer than compute):
══════════════════════════════════════════════════

  All warps do BOTH load + compute (same job, block-wide sync)

  Time:    0    100   200   300   400   500   600   700   800   900
           |     |     |     |     |     |     |     |     |     |
  DMA:     [====LOAD tile 0====]  [====LOAD tile 1====]  [====LOAD tile 2====]
  Warps:                          [COMP 0]                [COMP 1]
                                  ↑       ↑               ↑       ↑
                            data ready   done,       data ready   done,
                                         but tile 1               but tile 2
                                         not ready yet!           not ready yet!
  
  Warps:   IDLE IDLE IDLE IDLE    BUSY    IDLE IDLE IDLE   BUSY    IDLE IDLE IDLE
           ←── waiting for ──→            ←── waiting ──→          ←── waiting ──→
               tile 0 load                    tile 1 load              tile 2 load

  Problem: compute finishes fast, but ALL warps sit idle waiting for next tile.
  Why: consumer_wait() is block-wide → every warp must wait, even if unrelated.
  Hardware impact: compute units IDLE during wait, memory pipe IDLE during compute.
  Result: memory and compute are OUT OF PHASE — never busy at the same time.


WARP SPECIALIZATION (same workload):
════════════════════════════════════

  Each warp has ONE dedicated job, per-stage sync (only relevant warp waits)

  Time:    0    100   200   300   400   500   600   700   800   900
           |     |     |     |     |     |     |     |     |     |
  Loader:  [====LOAD tile 0====][====LOAD tile 1====][====LOAD tile 2====]
                                 ↑ starts immediately, no waiting!
  
  Compute:  waiting...          [COMP 0]  waiting   [COMP 1]  waiting  [COMP 2]
                                 ↑ only this warp pauses
  
  Storer:   waiting...                    [STORE 0]           [STORE 1]
                                           ↑ runs while compute waits

  Key: loader NEVER stops → always one tile ahead
       while compute waits, loader + storer keep memory pipe busy
       only 1 warp idle at a time, not ALL warps
  Why: pipeline API syncs per-stage → only consumer of THAT stage waits.
  Hardware impact: memory pipe + compute units busy SIMULTANEOUSLY (in-phase).
  Result: ~96% SM utilization vs ~92% for double buffering.


WHEN TO USE WHICH:
══════════════════

  compute time ≥ load time:  double buffering ≈ warp specialization (both overlap well)
  load time > compute time:  warp specialization wins (loader stays ahead, no block stall)
  small kernel / few tiles:  double buffering (warp specialization overhead not worth it)
  persistent / deep pipeline: warp specialization (overhead amortized over many iterations)

PyTorch, CUDA Pipeline API, and Warp Specialization
- torch.compile generates Triton kernels with pipeline barriers + warp specialization automatically — you don't write CUDA Pipeline API code yourself.
- scaled_dot_product_attention fuses 3 kernels (QK matmul → softmax → PV matmul) into one warp-specialized kernel — eliminates global memory round trips between stages.
- Under the hood: loader warps fetch tiles, compute warps process, storer warps write back — same pattern as handwritten CUDA pipeline.
- Even FlashAttention-3 reaches only ~75% peak FP16 FLOPS with warp specialization — 100% utilization isn't needed to be well-optimized.
- Warp specialization scales nearly linearly across warps until SMs or memory bandwidth saturate.

2/16/26:
- Persistent kernel = launch ONE long-running kernel instead of 1000 small ones → eliminates launch overhead (20ms saved in example).
- Threads loop, grabbing tasks via atomicAdd on a global counter — no exit/relaunch between tasks.
- 1000 tiny kernels: ~35% GPU utilization (SMs idle between launches). Persistent kernel: ~100% SMs active.
- Queue = pre-filled task array in device memory + atomic counter. CPU fills before launch, GPU threads consume.
- For continuous work: CPU writes tasks + signals via flag, GPU polls — needs system-scope memory fences on unified memory.
- Persistent kernels pair well with TMA: some warps prefetch next task's data while others compute current task.
- Low per-SM occupancy (12% in example) is fine — the point is ALL SMs stay active, not max warps per SM.

- Persistent kernels shine for many small/uneven tasks — dynamic load balancing via atomic counter, faster threads grab more work, no SM goes idle early.
- Common workloads: graph traversals, batched transforms, per-token LLM inference — anything with variable task sizes.
- Downsides: atomic contention on shared counter, harder to debug (one bad thread hangs entire kernel), can monopolize GPU blocking other work.
- 2-3× throughput vs naive separate kernels by eliminating launch overhead and keeping all SMs busy.
- PyTorch doesn't auto-fuse into persistent megakernels yet — requires custom CUDA or specialized compilers.
- Megakernel = persistent kernel that fuses multiple pipeline stages into one kernel, keeping data on-chip across stages.

Persistent Kernels and Warp Specialization:
- Persistent kernels + warp specialization = natural pairing: long-running loop amortizes warp role setup, deeper pipeline stays full across many tasks.
- "Short kernels don't benefit" = few loop iterations can't amortize pipeline ramp-up/drain overhead — use simple double buffering instead.
- Dynamic load balancing: each block grabs next task via atomicAdd — faster blocks process more tasks, no SM sits idle.
- Limitation: persistent kernel must find enough free SMs to launch — if other kernels occupy SMs, resources may not be available.
- Thread block clusters (next topic) help by grouping blocks across nearby SMs for persistent kernels.

Cooperative Groups:
- Cooperative Groups = sync threads at any granularity: tile (sub-warp), block, cluster (multi-block), or entire grid.
- grid.sync() = global barrier across ALL blocks — impossible with normal __syncthreads().
- Enables multi-phase algorithms in a single persistent kernel — no exit/relaunch between phases.
- Must launch with cudaLaunchCooperativeKernel() — CUDA guarantees all blocks fit on GPU simultaneously (otherwise launch fails).
- Size grid using cudaOccupancyMaxActiveBlocksPerMultiprocessor to avoid launch failure.

So "nonexistent blocks" = blocks that haven't launched yet because they're queued, waiting for SMs to free up. Cooperative launch prevents this by refusing to launch if all blocks can't fit -- basically, deadlock -- grid.sync() for all blocks but if all blocks aren't launched, there's deadlock right -- this is avoided. 

- cooperative launch grid size = SMs × max_blocks_per_SM (e.g. 132 × 4 = 528)
- exceed that limit → launch fails (not a silent bug, it just returns an error)
- use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to calculate safe grid size
- always check `cudaLaunchCooperativeKernel` return status

- pre-Blackwell cross-block coordination = global memory atomics/flags
  - works but forces intermediate data through HBM → extra traffic
- Blackwell alternative = thread block clusters + DSMEM
  - share data in on-chip SRAM, sync without global memory round trips

- use cooperative kernels when:
  - you need a true all-block barrier in a single launch
  - you want intermediate results to stay in fast memory (regs/shmem)

- downsides:
  - grid size capped by GPU concurrent block capacity
  - may force fewer, larger blocks
  - if ANY thread skips `grid.sync()` (e.g. divergent if) → deadlock
  - `grid.sync()` is heavyweight (coordinates across all SMs)
    - use sparingly, do significant work between calls

- for limited cross-block coordination:
  - global atomics / per-block flags = simpler, safer than full grid barrier
  - but less efficient than thread block clusters + DSMEM

- Persistent kernel + cooperative groups = best of both worlds
  - Persistent loop stays on GPU (no relaunch overhead)
  - `grid.sync()` provides global barrier between stages (no host sync needed)

- Example: 2 stages × 1000 iterations
  - Naive: 2000 kernel launches (launch overhead each time)
  - Cooperative persistent: 1 launch, `grid.sync()` between stages inside loop

- Pattern:
  - Stage 1: All blocks process dataA (grid-stride loop)
  - `grid.sync()` ← Every block finishes Stage 1 before any starts Stage 2
  - Stage 2: All blocks use finished dataA to update dataB
  - `grid.sync()` ← Barrier before next iteration

- Host side:
  - Check `prop.cooperativeLaunch` support
  - `cudaOccupancyMaxActiveBlocksPerMultiprocessor` → Get max blocks/SM
  - Clamp grid size to `maxBlocksPerSM × SM_count`
  - Launch with `cudaLaunchCooperativeKernel`

- Data locality across `grid.sync()`:
  - Per-thread data stays in registers
  - Per-block data stays in shared memory
  - Cross-block exchange → Global memory (or clusters + DSMEM on Blackwell)

- Ideal for: Multistep LLM inference (e.g. reduction/norm → per-element transform)
  - Eliminates all inter-kernel round trips
  - Maximizes on-chip data reuse

- Best practice: 1 thread block per SM for persistent kernels (if resources allow)
  - Each SM has a resident block that loops over a work queue
  - Use `grid.sync()` to coordinate between phases

- Decision guide:
  - Multiple phases needing global sync? → Cooperative kernel
  - Many small/irregular tasks with launch overhead? → Persistent kernel
  - Can reserve entire grid exclusively? → Cooperative + persistent (best perf)
  - Need to share SMs with other work? → Thread block clusters instead

- Key caveat: Cooperative kernels reserve ALL SMs in the grid
  - No other kernels can run concurrently on those SMs
  - If you need co-scheduling (e.g. async prefetch, lower-priority inference)
    → Partition GPU using thread block clusters

Thread Block Clusters and Distributed Shared Memory (DSMEM):
- Cooperative groups = software abstraction for syncing any grouping of threads (warps, tiles, blocks, grids)
- Thread block clusters (CTA clusters) = hardware-level hierarchy
  - GPU co-schedules a cluster's blocks on the same GPC (group of nearby SMs)
  - Only reserves a subset of SMs → remaining SMs free for other kernels
  - Unlike cooperative kernels which monopolize all SMs

- GPC = collection of nearby SMs: GPU Processing Cluster
  - GPU schedules cluster blocks onto a GPC like it schedules threads onto an SM
  - Blackwell B200: 2 GPC partitions (one per die), linked by NV-HBI
    - Coherent L2 across dies → appears as single logical GPC to software

- DSMEM (distributed shared memory):
  - On-chip SRAM shared across all blocks in a cluster
  - Low-latency cross-block communication without going through HBM
  - Native hardware support

- `cluster.sync()` = barrier for only the blocks in that cluster
  - Lighter than `grid.sync()` (which blocks every block on every SM)
  - Lets you synchronize a subset of blocks without stalling the whole GPU

- Thread block clusters subdivide the grid into smaller cooperative groups
  - Each cluster can share data via DSMEM and sync via `cluster.sync()`
  - Blocks outside the cluster are unaffected

- Without clusters: cross-block communication required global memory + `grid.sync()`
  - Both are slow → bottleneck for fine-grained cross-block cooperation
- With clusters: blocks in a cluster share data via DSMEM (on-chip) + sync via `cluster.sync()`
  - Hardware-supported barriers (PTX instructions + CUDA intrinsics) → much faster than `grid.sync()`

- Cluster launch control = hardware mechanism that:
  - Schedules persistent thread block clusters
  - Maintains balanced load even when some SMs are occupied by other work
  - Provides the foundation for efficient warp specialization

- Key shift: shared memory was per-block only → DSMEM makes it per-cluster
