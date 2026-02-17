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

2/17/26:

- Layerwise pipelining: split LLM across streams (e.g. stream 0 = layers 0-5, stream 1 = layers 6-11)
  - Stream 0 finishes batch N → allocs scratch for batch N+1 via cudaMallocAsync
  - Stream 1 keeps computing layers 6-11 on batch N's results — no stall
  - cudaMalloc would pause BOTH streams for every alloc → pipeline broken

- "Scratch buffer" = umbrella term for ALL temporary buffers in LLM:
  - Attention intermediates (Q×K^T scores)
  - LayerNorm temporaries
  - Softmax temporaries
  - KV cache
  - Intermediate activations between layers

Legacy Default Stream:
- Stream 0 (legacy default) = global serializer
  - Everything in stream 0 runs one-after-another (no overlap within)
  - AND it blocks ALL other streams:
    - Op goes into stream 0 → waits for ALL other streams to finish first
    - Op goes into any other stream → waits for stream 0 to finish first
  - Effectively a device-wide barrier every time you touch stream 0

- This kills ALL concurrency:
  - No kernel-kernel overlap
  - No kernel-copy overlap
  - Copy engines sit idle while kernels run and vice versa

- Rule: never use stream 0 unless you explicitly need global serialization (rare)
  - Always create explicit streams with cudaStreamCreateWithFlags

Modern Per-Thread Default Stream:
- Per-thread default stream (PTDS) = each CPU thread gets its OWN independent "stream 0"
  - Thread A's default stream doesn't block thread B's default stream
  - Full concurrency between threads without explicitly creating streams

- Enable via:
  - Compile: `nvcc --default-stream per-thread`
  - Env var: `CUDA_API_PER_THREAD_DEFAULT_STREAM=1` (before CUDA headers)

- Legacy stream 0: 1 global stream shared by all threads → global barrier
- PTDS: N threads → N independent default streams → no cross-thread barriers

- Caveat: if you mix PTDS with legacy stream 0 in the same process
  → PTDS streams still sync with the legacy default stream (don't mix them)

Default Versus Explicit (Nondefault) Streams:
- `cudaStreamNonBlocking` flag = stream will NOT sync with legacy stream 0
  - Without this flag → even explicit streams can get blocked by stream 0
  - Always use: `cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)`

- PyTorch uses non-default streams internally:
  - cuDNN, cuBLAS calls → own streams (avoid blocking stream 0)
  - NCCL distributed comms → separate HIGH-PRIORITY streams
    - Overlaps gradient communication with compute

- Best practice combo:
  - Enable PTDS (per-thread default streams)
  - AND create explicit streams with `cudaStreamNonBlocking`
  - Both together = zero accidental synchronization  

Best Practices: 
=== Stream Types Cheat Sheet ===
  Legacy default (stream 0): Global barrier — blocks and is blocked by everything
  Per-thread default (PTDS): Private per CPU thread — serializes own work, doesn't block others
  Explicit (non-default):    Fully independent — only syncs when YOU add dependencies

=== Best Practices ===
- Never put perf-critical work on stream 0 (even 1 stray op stalls everything)
- Some older CUDA APIs use stream 0 implicitly → migrate to newer APIs or always pass explicit stream
- Enable PTDS for multi-threaded CPU apps
- Create explicit streams with `cudaStreamNonBlocking` for all kernels/copies/allocs
- Use `cudaStreamWaitEvent()` for fine-grained dependencies between streams
  - NOT `cudaStreamSynchronize()` which stalls the whole stream
  - Only use `cudaStreamSynchronize()` at well-defined global points (e.g. end of epoch)

  Lightest:  cudaStreamWaitEvent()       → GPU-side, stream-to-stream, CPU free: 1 stream blocked/waits until another one is done
  Medium:    cudaStreamSynchronize()     → CPU blocks on 1 stream, GPU others run -- no host side ops allowed until everything on that specific stream's queue is done. 
  Heaviest:  cudaDeviceSynchronize()     → CPU blocks until ALL GPU work done  

=== Events Summary ===
- `cudaEventRecord(event, stream0)` = drop a marker at a point in stream 0
- `cudaStreamWaitEvent(stream1, event)` = stream 1 pauses until that marker is reached
- CPU never blocks, all other streams keep running
- Only the waiting stream pauses, and only until the specific event fires

=== Host Callback ===
- `cudaLaunchHostFunc(stream, callbackFn, userData)` = run a CPU function when stream reaches this point
- Enqueued INTO the stream like any other op — runs in order

- Caveat: NEVER call CUDA APIs inside the callback → deadlock
  (runtime thread is waiting for callback to finish, callback is waiting for runtime)

- Host memory pool = pre-allocated pinned CPU memory, managed like a buffer ring
  - Avoids repeated cudaMallocHost/cudaFreeHost OS calls (~ms each)
  - Pool alloc/free = pointer math (~μs)

- Must recycle buffers back to pool after GPU finishes with them
  - Pool has finite slots — if you don't return, pool runs out → stall
  - Use `cudaLaunchHostFunc` callback to recycle automatically when GPU is done
  - No CPU polling, no device sync needed

- With recycling: 3-4 pool slots can handle hundreds of batches
- Without recycling: need one slot per batch → memory explodes  

=== CUDA Events for Cross-Stream Sync ===
- Event = lightweight marker recorded at a specific point in a stream's queue
- Other streams call `cudaStreamWaitEvent()` to wait for that marker
- No CPU blocking, no device-wide stall — only the waiting stream pauses

- Chain events for complex dependency graphs:
  Stream A: [produce data] → record eventA
  Stream B: wait(eventA) → [process data] → record eventB
  Stream C: [independent work, runs freely]
  Stream D: wait(eventB) → [consume results]

- Real-world use in LLM training:
  - Compute stream records event when gradients are ready
  - NCCL communication stream waits on that event → starts all-reduce
  - Remaining compute continues in parallel with communication

- Multi-GPU pipeline parallelism:
  - GPU 0 records event when output tensor is ready
  - GPU 1 waits on that event before consuming it
  - No idle time on either GPU for independent work

- Events also useful for profiling (record timestamps in GPU timeline)  

Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel):

=== TWO-LEVEL PIPELINE: THE BIG PICTURE ===

  Level 1 (Inter-kernel): CUDA streams overlap DIFFERENT BATCHES
  Level 2 (Intra-kernel): Warp specialization overlaps DIFFERENT TILES within one batch

  Zoom levels:
  ┌─────────────────────────────────────────────────────────────┐
  │ LEVEL 1: Streams (batch level)                              │
  │                                                             │
  │ Stream 0: [upload B0][══ kernel B0 ══][download B0]         │
  │ Stream 1:      [upload B1][══ kernel B1 ══][download B1]    │
  │ Stream 0:                [upload B2][══ kernel B2 ══]...    │
  │                                                             │
  │ Copy engines + SMs + copy engines all busy simultaneously   │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                        zoom into one kernel
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ LEVEL 2: Warp specialization (tile level)                   │
  │                                                             │
  │ Inside "kernel B0":                                         │
  │                                                             │
  │ Warp 0 (loader):  [load T0][load T1][load T2][load T3]     │
  │ Warp 1 (compute):      [compute T0][compute T1][compute T2]│
  │ Warp 2 (storer):            [store T0][store T1][store T2]  │
  │                                                             │
  │ 3 warps overlap on 3 different tiles at once                │
  └─────────────────────────────────────────────────────────────┘


=== WHAT EACH LEVEL OVERLAPS ===

  Level 1 (streams):
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │Copy Eng  │  │   SMs    │  │Copy Eng  │
  │  H→D     │  │ compute  │  │  D→H     │
  │          │  │          │  │          │
  │ batch    │  │ batch    │  │ batch    │
  │ N+1      │  │ N        │  │ N-1      │
  └──────────┘  └──────────┘  └──────────┘
  3 different hardware units, 3 different batches, all at once

  Level 2 (warp specialization):
  ┌──────────────────────────────────┐
  │ Inside ONE kernel on ONE SM:     │
  │                                  │
  │  Warp 0: HBM → shared mem       │  tile N+1
  │  Warp 1: shared mem → compute   │  tile N
  │  Warp 2: compute → HBM          │  tile N-1
  │                                  │
  │  3 warps, 3 pipeline stages,     │
  │  3 different tiles, same SM      │
  └──────────────────────────────────┘


=== THE DOUBLE BUFFERING INSIDE ===

  Shared memory has 2 slots (ping-pong):

  Slot 0: [A0 | B0 | C0]     Slot 1: [A1 | B1 | C1]
           ↑                           ↑
     loader fills this           loader fills this
     while compute reads         while compute reads
     the other one               the other one

  Tile 0: loader → slot 0,  compute reads slot 0,  storer writes from slot 0
  Tile 1: loader → slot 1,  compute reads slot 1,  storer writes from slot 1
  Tile 2: loader → slot 0,  compute reads slot 0,  storer writes from slot 0
           (ping)              (pong)                (ping)


=== TWO PIPELINES INSIDE THE KERNEL ===

  pipe_lc (loader → compute):        pipe_cs (compute → storer):

  Loader                              Compute
    ↓ "tile is loaded"                  ↓ "result is ready"
  pipe_lc.producer_commit()           pipe_cs.producer_commit()
    ↓                                   ↓
  Compute                              Storer
  pipe_lc.consumer_wait()             pipe_cs.consumer_wait()
    ↓ "okay, I can read"               ↓ "okay, I can write back"

  Why TWO pipelines instead of one?
  - Loader and storer are INDEPENDENT — they don't wait for each other
  - Compute is the bridge: consumes from loader, produces for storer
  - Separate pipelines = maximum overlap between all three


=== HOST DRIVER PATTERN ===

  For each batch b:
  ┌────────────────────────────────────────────────┐
  │ Stream = s[b % 2]  (round-robin across 2)      │
  │                                                │
  │ 1. cudaMallocAsync  → alloc device buffers     │
  │ 2. cudaMemcpyAsync  → upload A, B  (H→D)      │
  │ 3. kernel<<<...>>>  → warp-specialized compute │
  │ 4. cudaMemcpyAsync  → download C   (D→H)      │
  │ 5. cudaFreeAsync    → free device buffers      │
  │                                                │
  │ All enqueued instantly, CPU moves to next batch │
  └────────────────────────────────────────────────┘

  CPU enqueues ALL batches rapidly:
  b=0 → stream 0    b=1 → stream 1    b=2 → stream 0    b=3 → stream 1 ...

  GPU executes:
  Stream 0: [alloc|upload|kernel|download|free]     [alloc|upload|kernel|download|free]
  Stream 1:       [alloc|upload|kernel|download|free]     [alloc|upload|kernel|download|free]
                   ↑ overlaps with stream 0


=== COMBINED: EVERYTHING AT ONCE ===

  Time → ────────────────────────────────────────────────────────►

  Copy H→D:  |batch 1 uploading |batch 2 uploading |batch 3 uploading |
  SMs:       |batch 0 kernel    |batch 1 kernel    |batch 2 kernel    |
  Copy D→H:       |batch -1 done|batch 0 download  |batch 1 download  |
                                 │
                          zoom into "batch 0 kernel":
                                 │
               Warp 0: [load T1][load T2][load T3]
               Warp 1:    [compute T0][compute T1][compute T2]
               Warp 2:        [store T0][store T1][store T2]

  = Every piece of hardware busy at every moment
  = Two levels of pipelining, fully nested

=== WARP SPECIALIZATION vs DOUBLE BUFFERING: RECAP ===

  Double buffering:
  - ALL warps do the SAME job (load → compute → store)
  - Overlap comes from 2 shared memory buffers:
    - All warps load tile N+1 into buffer B
    - Then all warps compute tile N from buffer A
    - Swap buffers, repeat
  - Sync: block-wide __syncthreads() or pipeline consumer_wait

  Warp specialization:
  - DIFFERENT warps have DIFFERENT jobs:
    - Warp 0 = loader (always loading)
    - Warp 1 = compute (always computing)
    - Warp 2 = storer (always storing)
  - Overlap comes from dedicated roles running simultaneously
  - Sync: pipeline producer_commit / consumer_wait between warps


=== WHAT THIS BOOK'S CODE DOES: BOTH ===

  It uses warp specialization (3 warps with dedicated roles)
  AND double buffering (2 shared memory slots, ping-pong)

  Why? They solve different problems:

  Warp specialization alone (1 buffer):
  Warp 0: [load T0]         [load T1]          [load T2]
  Warp 1:          [compute T0]       [compute T1]
  Warp 2:                   [store T0]          [store T1]
                    ↑ loader must WAIT for compute to finish
                      before it can reuse the buffer ❌

  Warp specialization + double buffering (2 buffers):
  Warp 0: [load T0 → buf0][load T1 → buf1][load T2 → buf0]
  Warp 1:                  [compute T0 ← buf0][compute T1 ← buf1]
  Warp 2:                          [store T0][store T1]
                    ↑ loader writes to buf1 WHILE compute reads buf0
                      no waiting! ✅

=== THE BIGGER PICTURE: EVERY LEVEL OF THE SYSTEM HAS IDLE TIME TO FILL ===

  The GPU is a pipeline of pipelines. Each level eliminates a different source of idle time.

  Level 0: No optimization
  ┌──────────────────────────────────────────────────────────────┐
  │ CPU: [upload][wait...][launch kernel][wait...][download]     │
  │ GPU:        [idle]    [compute]      [idle]                  │
  │ Copy:       [idle]                   [idle]                  │
  │                                                              │
  │ Problem: everything waits for everything else                │
  └──────────────────────────────────────────────────────────────┘

  Level 1: CUDA Streams → overlap batches across hardware engines
  ┌──────────────────────────────────────────────────────────────┐
  │ Copy H→D: [batch 1][batch 2][batch 3]                       │
  │ SMs:      [batch 0][batch 1][batch 2]                       │
  │ Copy D→H: [      ][batch 0][batch 1]                        │
  │                                                              │
  │ Solved: CPU↔GPU transfer latency hidden                     │
  │ Remaining problem: INSIDE each kernel, still sequential      │
  └──────────────────────────────────────────────────────────────┘

  Level 2: Warp specialization → overlap pipeline stages within a kernel
  ┌──────────────────────────────────────────────────────────────┐
  │ Inside each kernel:                                         │
  │ Loader:  [tile 1][tile 2][tile 3]                           │
  │ Compute:    [tile 0][tile 1][tile 2]                        │
  │ Storer:        [tile 0][tile 1][tile 2]                     │
  │                                                              │
  │ Solved: HBM latency hidden within a kernel                  │
  │ Remaining problem: loader waits for buffer to free up        │
  └──────────────────────────────────────────────────────────────┘

  Level 3: Double buffering → loader never waits for compute
  ┌──────────────────────────────────────────────────────────────┐
  │ Loader:  [T0→buf0][T1→buf1][T2→buf0][T3→buf1]              │
  │ Compute:     [T0←buf0][T1←buf1][T2←buf0]                   │
  │ Storer:          [T0][T1][T2]                               │
  │                                                              │
  │ Solved: no warp ever waits for a shared memory buffer        │
  └──────────────────────────────────────────────────────────────┘

  Level 4: Thread block clusters + DSMEM → eliminate duplicate loads
  ┌──────────────────────────────────────────────────────────────┐
  │ Leader loads tile once → shares via DSMEM to all blocks      │
  │                                                              │
  │ Solved: multiple blocks needing same tile don't each load it │
  └──────────────────────────────────────────────────────────────┘


=== EACH LEVEL TARGETS A DIFFERENT BOTTLENECK ===

  Bottleneck                          Solution
  ───────────────────────────────     ──────────────────────────
  CPU↔GPU transfer latency           Streams (overlap H2D/D2H with compute)
  HBM→SMEM load latency              Warp specialization (dedicated loader warp)
  Buffer contention in SMEM          Double buffering (2 slots, ping-pong)
  Duplicate HBM loads across blocks  Clusters + DSMEM (load once, share)
  Launch overhead across batches     Persistent kernels (one launch, loop inside)
  Global sync overhead               Cooperative groups / cluster.sync()


=== ONE SENTENCE ===

  Streams feed the machine; warp spec + double buffering keep it full;
  clusters eliminate waste — together they leave zero idle silicon.

cp.async/cuda::memcpy_async -- DMA unit inside the SM handles HBM -> shared memory

  Async load (cp.async / cuda::memcpy_async):
  HBM → shared memory (bypasses registers)
  - DMA unit inside the SM handles it
  - Thread is FREE to do other work while data moves
  - This is what the warp-specialized kernel uses ✅

also, it looks like in double buffering, we create enough shared memory size so that we have 2 buffers instead of 1, right

=== LOAD PATH IN WARP SPECIALIZATION ===

  Two different cases:

  Regular load (without async):
  HBM → registers → shared memory
  - Thread issues load instruction → data goes to register first → thread writes to SMEM
  - Thread is busy during the transfer

  Async load (cp.async / cuda::memcpy_async):
  HBM → shared memory (bypasses registers)
  - DMA unit inside the SM handles it
  - Thread is FREE to do other work while data moves
  - This is what the warp-specialized kernel uses ✅

  So in this code:
  Warp 0 (loader): issues cuda::memcpy_async → HBM directly to SMEM
  Warp 1 (compute): reads from SMEM → registers → ALU/Tensor Cores
  Warp 2 (storer): reads from SMEM → registers → writes to HBM


=== DOUBLE BUFFERING SHARED MEMORY ===

  Yes, exactly. You allocate 2× the SMEM:

  Single buffer:
  SMEM: [A | B | C]                    = 3 × TILE_SIZE × sizeof(float)
        ↑ loader and compute fight over this

  Double buffer:
  SMEM: [A0 | B0 | C0 | A1 | B1 | C1] = 6 × TILE_SIZE × sizeof(float)
         ↑ slot 0        ↑ slot 1
         compute reads   loader writes
         this one        this one
                         (simultaneously, no conflict)

  Trade-off:
  - Uses 2× shared memory → may reduce max blocks per SM (lower occupancy)
  - But eliminates wait time → usually worth it
  - Must fit: 6 × 1024 × 4 = 24KB << 228KB available on Blackwell ✅ (no problem)

=== WHY ASYNC MEMCPY ENABLES TRUE PARALLEL LOAD + COMPUTE ===

  Regular load (ld.global):
  - Thread issues load → thread's register is OCCUPIED waiting for data
  - Thread is stalled (or at least its register is tied up)
  - The warp scheduler can switch to another warp, but that warp's
    registers/ALU slots are still held

  Async load (cp.async / cuda::memcpy_async):
  - Thread says "copy this from HBM to SMEM" → hands it off to DMA unit
  - Thread's registers are NOT involved in the transfer
  - DMA unit (separate hardware inside the SM) does the copy
  - Thread is completely free → compute warp can use the ALU

  ┌──────────────────────────────────┐
  │ SM                               │
  │                                  │
  │  Warp 0 (loader):               │
  │    issues cp.async → hands off   │
  │    to DMA unit, warp does nothing│
  │                                  │
  │  DMA unit: [HBM → SMEM]  ←──── separate hardware, no ALU/register use
  │                                  │
  │  Warp 1 (compute):              │
  │    uses ALU + registers freely   │ ← no contention with the DMA ✅
  │    reads from SMEM (other slot)  │
  │                                  │
  │  Both happening simultaneously   │
  └──────────────────────────────────┘

  Without async:
  - Load goes through registers → registers tied up → fewer available for compute
  - ALU might be idle while waiting for load to finish

  With async:
  - Load bypasses registers entirely → DMA unit handles it
  - ALU + registers fully available for compute warp
  - Two different hardware paths, zero contention  

Async memory transfer helps both:
  - Double buffering: makes the overlap REAL (not just phase alternation)
  - Warp specialization: frees loader warp's registers/ALU for compute warp

async memcpy is the foundation that makes BOTH techniques actually overlap
load and compute simultaneously, rather than just interleaving them.  
