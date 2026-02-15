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