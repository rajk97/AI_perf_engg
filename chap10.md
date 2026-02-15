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