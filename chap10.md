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

