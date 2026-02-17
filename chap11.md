Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocations

- CUDA streams provide the foundation for this inter-kernel concurrency. 
- A CUDA stream is a sequence of operations that execute in order on the GPU.
- Streams are fundamentally a CPU-side scheduling mechanism for GPU work
- Streams = parallel work queues feeding the GPU
- CPU submits work to multiple queues simultaneously
- GPU pulls from all queues and runs tasks whenever hardware is available

Using Streams to Overlap Compute with Data Transfers
- Pinned memory required for truly async transfers (`cudaMallocHost` or `pin_memory=True`)
  - Pageable memory → hidden staging copy → blocks CPU → kills stream overlap
  - Pinned memory → DMA copy engine reads directly → CPU returns immediately

- Use `cudaMallocAsync` / `cudaFreeAsync` instead of `cudaMalloc` / `cudaFree`
  - `cudaMalloc` is blocking — stalls ALL streams on the GPU until allocation completes
  - `cudaMallocAsync` is per-stream — only affects that stream, others keep running

- cudaMalloc = device-wide sync barrier; cudaMallocAsync = stream-local, zero cross-stream stall --> CPU calls cudaMalloc which basically stalls the entire GPU until the allocation completes. 

Stream-Ordered Memory Allocator: 

- Streams are fundamentally a CPU-side scheduling mechanism for GPU work
- Caching allocator = fast until the pool dries up, then everyone stops; stream-ordered = never stops anyone

=== PyTorch Default Caching Allocator ===

  - PyTorch grabs a BIG chunk of GPU memory upfront via cudaMalloc
  - Then manages it internally like a memory pool
  - tensor alloc = grab from pool (fast, no CUDA call)
  - tensor free = return to pool (fast, no CUDA call)
  - Problem: when pool runs out → calls cudaMalloc → device-wide stall
  - Also: fragmentation over time → pool has free space but not contiguous → OOM

  Pool:
  [████ used ████][  free  ][██ used ██][free][██ used ██]
                                        ↑ 
                              Need 100MB but only 50MB contiguous
                              → OOM even though total free > 100MB


=== Stream-Ordered Allocator (cudaMallocAsync backend) ===

  - CUDA driver manages memory, not PyTorch
  - Every alloc/free is just another task in the stream's queue
  - Driver knows which stream freed what → can reuse immediately for same stream
  - No fragmentation problem — driver defragments/reuses intelligently
  - Never causes cross-stream stalls

  Stream 1: [alloc A] → [kernel] → [free A]
  Stream 2: [alloc B] → [kernel] → [free B]
            ↑ driver knows A is free after stream 1 finishes
              → can reuse A's memory for stream 2's next alloc


=== ACTUAL DIFFERENCES ===

                        Caching Allocator         Stream-Ordered Allocator
                        ─────────────────         ────────────────────────
  Who manages memory    PyTorch (user-space)      CUDA driver (OS-level)
  Pool exhaustion       cudaMalloc → stall ALL    Never stalls other streams
  Fragmentation         Can fragment → OOM        Driver handles reuse better
  Cross-stream reuse    Needs manual sync         Driver tracks stream deps
  Enable in PyTorch     Default                   PYTORCH_ALLOC_CONF=backend:cudaMallocAsync
  Overhead per alloc    Near-zero (pool hit)      Near-zero (stream-enqueued)
  Worst case            cudaMalloc stall          No worst case stall

  The driver maintains a pool of GPU memory per device:

  cudaMallocAsync → grabs from pool (fast)
  cudaFreeAsync   → returns to pool (fast, doesn't go back to OS)

  But the pool can grow large if you keep allocating and freeing:

  Pool: [████████████████████████]  ← 8GB held by pool
  Actual use: [██████]               ← only 2GB needed right now
               6GB sitting unused in pool, unavailable to other processes


=== RELEASE THRESHOLD ===

  Controls: when should the pool RETURN unused memory to the OS?

  cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, threshold)

  Low threshold (e.g. 0 bytes):
  - Pool returns memory to OS aggressively
  - Other processes can use the freed GPU memory
  - But next batch → pool needs to re-request from OS → slow OS call
  - Good for: multi-process / multi-model sharing the GPU

  High threshold (e.g. UINT64_MAX):
  - Pool hoards memory, never returns to OS
  - Next batch → memory already in pool → instant reuse, no OS call
  - But other processes can't use that GPU memory
  - Good for: single dedicated workload that needs consistent performance

  Timeline:

  Low threshold:
  Batch 0: [alloc from OS] [use] [free → return to OS]
  Batch 1: [alloc from OS] [use] [free → return to OS]  ← OS call every time ❌
           ↑ slow                ↑ slow

  High threshold:
  Batch 0: [alloc from OS] [use] [free → stays in pool]
  Batch 1: [reuse from pool] [use] [free → stays in pool]  ← instant ✅
           ↑ fast, no OS call

- Copy engine = dedicated DMA hardware on the GPU
  - ONLY job: move data between CPU RAM ↔ GPU HBM over PCIe/NVLink
  - Cannot compute — purely a data mover
  - Runs independently of SMs → that's why copy + compute can overlap

- Most modern GPUs have 2 copy engines:
  - Engine 1: Host → Device (upload)
  - Engine 2: Device → Host (download)
  - Both can run simultaneously + SMs compute = 3-way overlap

- asyncEngineCount tells you how many your GPU has:
  - 1 = upload and download take turns (but still overlap with compute)
  - 2 = upload + download + compute all at the same time

- Check: cudaGetDeviceProperties → prop.asyncEngineCount

- GB200 (Blackwell): 3 copy engines
  - Engine 1: Host → Device (H2D)
  - Engine 2: Device → Host (D2H)
  - Engine 3: Peer-to-peer (Device → Device, e.g. GPU-to-GPU over NVLink)

- So GB200 can do 4-way overlap:
  - H2D copy + D2H copy + GPU↔GPU copy + SM compute — all simultaneously


  CPU RAM (host)                          GPU HBM (device)
  ┌──────────────┐                        ┌──────────────┐
  │              │     PCIe / NVLink       │              │
  │  src buffer  │ ══════════════════════► │  dst buffer  │
  │  (pinned)    │    Copy Engine          │              │
  │              │    does this transfer   │              │
  └──────────────┘                        └──────────────┘

=== BANDWIDTH ===

  The copy engine is limited by the BUS, not itself:
  - PCIe Gen5 x16: ~64 GB/s
  - NVLink (GB200): ~900 GB/s per direction (GPU↔GPU)
  - Copy engine can saturate these buses
  - It's the pipe that's the bottleneck, not the engine

- Blackwell B200/GB200: 160 SMs (not 128)
  - 2 dies × 80 SMs per die = 160 SMs total

- 128 concurrent kernels (grids) is a SEPARATE hardware limit
  - This is a scheduler/firmware limit, not related to SM count
  - It's been 128 since Volta/Turing era
  - Applies to ALL GPUs regardless of SM count

- So on GB200:
  - 160 SMs for compute
  - 128 max concurrent grids (kernel launches)
  - These are independent numbers

- On H100:
  - 132 SMs for compute
  - 128 max concurrent grids
  - Still independent

=== YES, 1 SM CAN RUN BLOCKS FROM MULTIPLE KERNELS ===

  SM has resources: 64 max warps, 256KB registers, 228KB shared memory

  If kernel A's block uses: 8 warps + 32KB shmem + 64KB regs
  And kernel B's block uses: 8 warps + 32KB shmem + 64KB regs

  SM can fit both:
  ┌────────────────────────────────────┐
  │ SM 0                              │
  │                                   │
  │ [Block from Kernel A]  8 warps    │
  │ [Block from Kernel B]  8 warps    │
  │                                   │
  │ Total: 16/64 warps, 64KB/228KB    │
  │        shmem, 128KB/256KB regs    │
  │        Room for more! ✅          │
  └────────────────────────────────────┘

  The warp scheduler doesn't care WHICH kernel a warp belongs to.
  It has 4 schedulers picking from ALL resident warps every cycle:

  Cycle N:   scheduler 0 picks warp from kernel A
             scheduler 1 picks warp from kernel B
             scheduler 2 picks warp from kernel A
             scheduler 3 picks warp from kernel B
  → Interleaved execution, same SM, different kernels

  But if kernel A's block uses: 48 warps + 200KB shmem
  → No room for kernel B on this SM ❌ (resources exhausted)


- cudaMallocAsync supports variable-length buffers within a stream
  - Alloc token caches / activations → immediately launch dependent kernel → same stream, no sync
  - Useful for LLM inference where buffer sizes vary per request (variable sequence lengths)

- Tuning rule: chunk size should balance copy time ≈ compute time for max overlap
  - Too small → launch overhead dominates, engines underutilized
  - Too large → one kernel hogs all SMs, no room for inter-kernel concurrency

- 128 concurrent grids limit per device (scheduler hardware limit, not SM count)
  - Exceeding → kernels queue until a slot frees up

- Multiple kernels CAN share an SM if combined resources fit (regs + shmem + warps)

For LLMs: 
- Scratch buffer = scrap paper for intermediate math — different problem sizes need different sheets, async alloc hands them out without stopping the class --> variable length is important here coz. the alternative is max. size which wastes memory too. 
- Variable-length sequences → scratch buffer sizes fluctuate per batch
  - Batch N (512 tokens) needs 16MB scratch, batch N+1 (1024 tokens) needs 32MB
  - Can't just reuse previous allocation → need stream-ordered alloc for different sizes

- Autoregressive decoding: KV cache grows with each generated token
  - Need to reallocate/extend scratch buffer as sequence gets longer
  - cudaMallocAsync lets you grow the buffer in the same stream as the attention kernel
  - Other streams (data loading, result copying) keep running in parallel