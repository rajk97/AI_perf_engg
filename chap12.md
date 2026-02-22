Dynamic Scheduling, CUDA Graphs, and Device-Initiated Kernel Orchestration:

- We'll understand dynamic, device-side, and graph-based kernel orchestration techniques that keep every SM fed across multi-GPU clusters. 
- When work is not evently distributed across thread blocks, SM's will sit idle. 
- Active cycles/Total elapsed SM cycles give an idea of underutilization

- Atomic Counters: 
2/22/26:

- Atomic counters: foundation for dynamic work queues on GPU
- L2 cache services atomics on-chip (~200 cy) — fast when uncontended
- Contention: multiple threads hit same address simultaneously → hardware replays → waste
- Nsight Compute metrics:
  - atomic_transactions: total L2 atomic ops (including replays)
  - atomic_transactions_per_request: contention ratio (1.0 = ideal, >1.0 = retries)

- Fix: BATCH the atomic — grab N items per atomic instead of 1

  BEFORE (1 item per atomic)          AFTER (32 items per atomic)
  ────────────────────────────         ────────────────────────────
  T0: atomicAdd(&head, 1) ─┐          W0: atomicAdd(&head, 32)
  T1: atomicAdd(&head, 1) ─┤ 32       │ start = result
  T2: atomicAdd(&head, 1) ─┤ atomics  │ __shfl_sync → broadcast to warp
  ...                       │ per      │ all 32 threads process items
  T31: atomicAdd(&head, 1)─┘ warp     └→ 1 atomic per warp

  Contention: 32×             Contention: 1×

  CODE
  ─────────────────────────────────────────────────────────
  // One thread per warp does the atomic
  int start = atomicAdd(&queue_head, batchSize);   // grab 32 items
  // Broadcast to all lanes
  start = __shfl_sync(0xFFFFFFFF, start, 0);
  // Each thread in warp processes one item
  int myIdx = start + laneId;
  if (myIdx < N) process(data[myIdx]);

  ALTERNATIVE 1: PER-BLOCK COUNTERS
  ─────────────────────────────────────────────────────────
  Global counter      Per-block counters (shared memory)
  ┌─────────┐         ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ head=0  │         │ local=0 │  │ local=0 │  │ local=0 │
  │ (L2)    │         │ (SMEM)  │  │ (SMEM)  │  │ (SMEM)  │
  │         │         │ Block 0 │  │ Block 1 │  │ Block 2 │
  └────┬────┘         └────┬────┘  └────┬────┘  └────┬────┘
       │                   │            │            │
  All threads ──→          Threads ──→ SMEM atomic (fast, no L2 contention)
  hammer L2                │            │            │
  (slow)                   └──── periodically merge to global ────┘
                                 (one global atomic per block)

  Flow:
  Thread → atomicAdd(&local_counter, 1)   ← SMEM: ~20 cycles, block-local
  When local_counter hits threshold:
    One thread → atomicAdd(&global_head, threshold)  ← L2: rare, batched
  Result: 1000 threads → ~8 global atomics (1 per block) instead of 1000


  ALTERNATIVE 2: HIERARCHICAL REDUCTION
  ─────────────────────────────────────────────────────────
  Level 0 (thread):     each thread has local count
                        ↓ warp shuffle to sum
  Level 1 (warp):       warp-level total (32→1)
                        ↓ SMEM atomic
  Level 2 (block):      block-level total (warps→1)
                        ↓ global atomic
  Level 3 (global):     final counter in L2

  Thread ──warp shuffle──→ Warp sum ──SMEM atomic──→ Block sum ──L2 atomic──→ Global
  32:1 reduction           8:1 reduction              N:1 reduction
  (registers, free)        (SMEM, ~20 cy)             (L2, ~200 cy)

  Example: 8192 threads (32 blocks × 8 warps × 32 threads)
  Naive:         8192 global atomics
  Per-block:       32 global atomics
  Hierarchical:    32 global atomics, but SMEM atomics also reduced by 8×

  Both alternatives: push contention to cheaper/faster levels of the memory hierarchy

THE PROBLEM: UNEVEN WORK
─────────────────────────────────────────────────────────
Static assignment: thread i → item i (fixed)

Warp 0: [==]────────idle────────   fast threads wait for slow ones
Warp 7: [========================] everyone waits for the slowest warp
        ↑ thread-level imbalance = wasted SIMD lanes (unfixable in SIMT)

THE SOLUTION: DYNAMIC ATOMIC QUEUE
─────────────────────────────────────────────────────────
Global counter: [next = 0]

┌─────────────────────────────────────────────────┐
│ 1. Warp leader: base = atomicAdd(&counter, 32)  │ one atomic per batch
│ 2. __shfl_sync: broadcast base to 32 threads    │ free (register crossbar)
│ 3. Each thread: process(data[base + laneId])     │ all 32 threads work
│ 4. Loop back to step 1 until counter ≥ N         │ no idle time
└─────────────────────────────────────────────────┘

Dynamic result:
Warp 0: [==][==][===][==][====][==]──   fast warps grab more batches
Warp 7: [====][===][====][===]───────   slow warps grab fewer
        ↑ all warps busy until work exhausted, kernel finishes earlier

WHY IT WORKS — THE KEY INSIGHT
─────────────────────────────────────────────────────────
Thread-level imbalance:  SIMT lockstep → can't hide → wasted lanes
Warp-level imbalance:    SM warp scheduler → freely hides → no waste

Dynamic queue moves imbalance from thread level → warp level
→ where the hardware can absorb it via warp scheduling

CONSTRAINTS
─────────────────────────────────────────────────────────
- Only works for INDEPENDENT items (no data dependencies between items)
- Atomic queue itself adds ~200 cy overhead per batch (L2 atomic)
- Batch size tradeoff: too small = contention, too large = tail imbalance
- 8-32 items per batch is the sweet spot on modern GPUs

GPU MEMORY HIERARCHY FOR COMMUNICATION
─────────────────────────────────────────────────────────
Scope          Via              Latency     Use case
─────────────  ───────────────  ──────────  ─────────────────
Threads→Warp   Warp shuffle     ~0 cycles   Broadcast batch base
Warps→Block    Shared memory    ~20 cycles  Per-block counters
Blocks→GPU     L2 / Global      ~200 cycles Global work queue  

  ════════════════════════════════════════════════════════════
  DYNAMIC SCHEDULING — VISUAL DISCOVERY MAP
  ════════════════════════════════════════════════════════════

  1. THE PROBLEM
  ─────────────────────────────────────────────────────────
  Items:  [light][light][HEAVY][light][HEAVY][light][HEAVY][light]

  Static assignment:
  Block 0 → [light][light]  → done ──────idle──────────────
  Block 1 → [HEAVY][light]  → ─────────────────done────────
  Block 2 → [HEAVY][light]  → ────────────────done─────────
  Block 3 → [HEAVY][light]  → ───────────────done──────────
                                           ↑ GPU done here
                                  Block 0 wasted half the time

  2. THE FIX
  ─────────────────────────────────────────────────────────
  Shared queue: [item0|item1|item2|item3|item4|item5|item6|item7]
              ↑ any warp grabs next available

  Block 0: [grab][grab][grab][grab][grab]──queue empty──EXIT
  Block 1: [  grab  ][  grab  ][ grab  ]──queue empty──EXIT
                                       ↑ GPU done here (earlier)

  3. HOW: while(true) + atomicAdd
  ─────────────────────────────────────────────────────────
  Static kernel body:              Dynamic kernel body:

  ┌────────────────┐               ┌──────────────────────┐
  │ get fixed idx  │               │ ┌──────────────────┐ │
  │ compute        │               │ │ atomicAdd → base │ │
  │ return (die)   │               │ │ shfl → broadcast │ │
  └────────────────┘               │ │ compute          │ │
                                │ │ done? loop back  │ │
                                │ └──────────────────┘ │
                                │ queue empty? → return│
                                └──────────────────────┘

  4. THREAD vs WARP IMBALANCE
  ─────────────────────────────────────────────────────────
  Thread imbalance (UNFIXABLE — SIMT lockstep):
  ┌─── Warp ──────────────────────────┐
  │ T0:  done─────idle────────────────│ ← can't do anything else
  │ T31: ─────────────────────done────│ ← everyone waits for this
  └───────────────────────────────────┘

  Warp imbalance (FIXABLE — warp scheduler hides it):
  SM warp scheduler picks whichever warp is ready:
  → Warp 0 done? grab more work
  → Warp 1 stalled? skip, run Warp 2
  → Warp 2 done? grab more work

  5. THE QUEUE MECHANISM
  ─────────────────────────────────────────────────────────
  globalIndex in L2:  [0] → [32] → [64] → ... → [N] STOP

  Warp leader ──atomicAdd(&globalIndex, 32)──→ gets base
    │
  __shfl_sync ──broadcast base──→ all 32 threads know
    │
  T0: base+0    T1: base+1    ...    T31: base+31
    │              │                     │
  process()      process()             process()
    │
  loop back ──→ next atomicAdd

  6. WHY IT WORKS
  ─────────────────────────────────────────────────────────
  Static:                        Dynamic:
  total time = max(block)        total time = total_work / num_SMs

    ████                           ██
    ████                           ██
    ████████                       ██
    ████                           ██
    ████████████  ← slowest        ██  ← all finish together
    ████                           ██
    ████████                       ██
    ████                           ██

  7. CONSTRAINTS
  ─────────────────────────────────────────────────────────
  ✓ Works:   item[i] independent of item[j]    (histogram, map, element-wise)
  ✗ Breaks:  item[i] depends on item[i-1]      (prefix sum, recurrence)

  Queue overhead:  ~200 cy per atomicAdd (L2)
  Batch size:      8-32 = sweet spot (less = contention, more = tail waste)

is it also safe to say that thread level imbalance still exists but maybe minimized due to small batch sizes and hidden with warp scheduling here? but the main alpha comes from dynamic work instead? please tell, thank you

Thread imbalance still exists — SIMT lockstep is physics, can't eliminate it
Minimized — batch of 32 consecutive items means work = idx%256 varies by only 31 across the warp (e.g., 64→95 instead of 0→255), so the spread is smaller than random assignment
Hidden — warp scheduler switches to other warps while the slow warp finishes, so the SM doesn't stall
But yes, the main alpha is the dynamic work distribution — blocks/warps never idle because they keep pulling from the queue. Thread-level gains are a nice side effect, not the point.

Mnemonic: Thread imbalance = minor tax (small spread + scheduler hides it), dynamic queue = the real win.

  WITHOUT CUDA GRAPHS — CPU launches kernels one by one
  ─────────────────────────────────────────────────────────────────────
  CPU:  [==kernel launch==]      [==kernel launch==]      [==kernel launch==]
  GPU:                     [kern]                   [kern]                   [kern]
                        ↑ gap ↑                  ↑ gap ↑                  ↑ gap
                   CPU overhead              CPU overhead             CPU overhead
                   between each              between each             between each

  GPU is IDLE while CPU prepares each launch → short kernels = mostly waiting


  WITH CUDA GRAPHS — one graph launch dispatches all kernels
  ─────────────────────────────────────────────────────────────────────
  CPU:  [==graph launch==]
  GPU:                     [kern][kern][kern]
                        ↑ back-to-back, no gaps

  CPU does ONE call → GPU runs entire pipeline without waiting for CPU between ops
  The shorter the kernels, the bigger the win (less gap overhead per kernel)

  CUDA GRAPHS
  ─────────────────────────────────────────────────────────
  - Capture an entire pipeline (kernels, copies, events, callbacks) into a DAG once
  - Replay it every iteration with one CPU call → near-zero CPU launch overhead
  - Dependencies are baked into the graph → no manual sync between ops
  - Does NOT magically overlap more than streams already do — it eliminates the
 CPU-side cost of enqueuing each op individually every iteration

PyTorch, Inference Engines, and CUDA Graphs:

  - PyTorch: torch.cuda.Graph captures static op sequences; reduce-overhead compiler mode
 auto-wraps eligible segments in graphs (not guaranteed for all paths, may increase memory)
  - vLLM / TensorRT-LLM: pre-capture one graph per batch size at startup
 → bucket/pad inputs to match a captured graph's fixed shape → replay at runtime
  - Constraint: graphs need STATIC shapes (fixed tensor addresses, fixed sizes)
 → dynamic shapes = multiple graphs (one per bucket) or fall back to eager mode

 MEMORY POOLS FOR CUDA GRAPHS — SUMMARY
──────────────────────────────────────────────

- Pre-allocate memory outside the graph, not inside
- Frameworks like PyTorch use static memory pools per graph
- Graph can overlap independent transfers and compute (like streams)
  but won't make any single op faster — it eliminates CPU scheduling overhead
- Dependency graph known upfront = automatic overlap without manual sync

YOUR INSIGHTS (confirmed correct)
──────────────────────────────────────────────

1. "the memory pool basically has same addresses coz its like fixed
    memory and hence graphs with memory operations inside of them
    with fixed memory addresses can safely use this memory pools"

2. "the main alpha here is the blocks instead of carrying a fixed
    amount of work carries a dynamic amount of work"
    (from dynamic scheduling — same principle applies: graphs want
    fixed/static things, dynamic things go outside)

3. "instead of block, its the kernel being an atomic queue is the
    real dynamic scheduling implementation here"
    (and graphs are the opposite — they want everything STATIC
    and predictable, which is why dynamic ops go in eager mode)

CUDA GRAPHS: Core Concept
═══════════════════════════════════════════════════════════

WITHOUT CUDA Graph:                    WITH CUDA Graph:
  CPU          GPU                      CPU          GPU
   │                                     │
   ├─launch A──→ [A]                     ├─capture──→ record A,B,C
   ├─launch B──→    [B]                  │            (one-time setup)
   ├─launch C──→       [C]              │
   │                                     ├─replay───→ [A][B][C]
   3 CPU→GPU round trips                 ├─replay───→ [A][B][C]
   per iteration                         ├─replay───→ [A][B][C]
                                         1 CPU call per iteration!

- CUDA Graph = record a sequence of GPU ops once, replay with ONE launch call — eliminates per-kernel CPU→GPU launch overhead
- Capture: BeginCapture → enqueue kernels/copies/events → EndCapture → Instantiate → Replay in loop
- Only works for REPETITIVE sequences (same ops, same shapes, same pointers each iteration)
- cudaGraphExecUpdate: tweak node parameters (pointers, sizes) without recapturing entire graph
- Partial capture OK: only the repetitive portion needs to be graphed
- PyTorch: torch.cuda.CUDAGraph() + with torch.cuda.graph(g): block to capture, g.replay() to launch
- Gotcha: must use static buffers (fixed memory addresses) — no allocations inside capture
- Gotcha: warm-up pass required before capture to initialize lazy CUDA/cuBLAS/cuDNN setup
- Mnemonic: "Record once, replay forever — CUDA Graph removes the CPU middleman."

- CUDA Graphs: 300 kernel launches → 100 graph replays, ~25% faster end-to-end by eliminating CPU→GPU handshakes
- Zero GPU idle time between kernels (back-to-back execution) vs ~3µs gaps without graphs
- Pitfall: graph is invalid if workload size changes — must recapture or use cudaGraphExecUpdate
- Pitfall: no memory allocation or host-device sync inside capture — allocate everything before capture
- Pointer stability: all tensors must stay at same memory address across replays — no reallocation between iterations
- PyTorch: torch.cuda.graph_pool_handle() creates dedicated memory pool for fixed-address tensors during capture
- Update inputs by copying into static tensors (static_x.copy_(new_data)), never reallocate
- No host-side ops inside capture: print(), RNG, nested captures, memory allocation — graph must be pure deterministic GPU work
- All tensors pre-allocated with fixed addresses + fixed shapes before capture — no resizing or cudaMalloc during capture

Data updating between replays??

SETUP (once):
  static_input  = torch.empty(shape, device='cuda')  ← fixed address
  static_output = torch.empty(shape, device='cuda')  ← fixed address

CAPTURE (once):
  with torch.cuda.graph(g):
      # Graph records: "read from static_input's ADDRESS, write to static_output's ADDRESS"
      static_output = model(static_input)

REPLAY LOOP (every iteration):
  for batch in dataloader:
      static_input.copy_(batch)    ← Copy NEW data into the SAME address
      g.replay()                   ← Graph runs on whatever is at that address now
      result = static_output       ← Read result from fixed address

Dynamic Graph Update: 

- Dynamic update = change kernel parameters, grid/block dims, or pointers in an existing graph without recapturing (~few µs)
- Workflow: capture template graph at MAX expected size → update parameters per iteration for actual size
- Can change: scalar values, pointers, grid dims, swap kernel of same signature
- Cannot change: graph structure (add/remove nodes) → must recapture if topology changes
- Best for: semi-static workloads (same pipeline, few varying parameters like batch size)
- Mnemonic: "Capture the skeleton once, update the muscles each iteration."

Device-Initiated CUDA Graph Launch:
- GPU can launch graphs automatically
- Device-initiated launch = GPU kernel triggers a prerecorded graph entirely on-device, no CPU round-trip
- Setup: capture graph on host → cudaGraphInstantiate with DeviceLaunch flag → cudaGraphUpload to GPU memory (must upload before device launch)
- ~2× lower launch latency vs host-side graph launch, and latency stays flat regardless of graph size/width
- Host-launch latency grows with more nodes/branches (CPU scheduling overhead); device-launch does not
- Launch API from device code: cudaGraphLaunch(graphExec, stream) with special stream values for mode selection
- Fire-and-forget (cudaStreamGraphFireAndForget): child graph runs immediately, concurrently with parent kernel (max 120 per graph execution)
- Tail launch (cudaStreamGraphTailLaunch): child runs after parent finishes
- Sibling launch (cudaStreamGraphFireAndForgetAsSibling): concurrent like fire-and-forget but scheduled as sibling
- Debug with Nsight Systems (child graphs appear as separate GPU timeline streams) + NVTX markers around cudaGraphLaunch calls

- Self-relaunch pattern: scheduler kernel calls cudaGetCurrentGraphExec() to get its own graph handle, then tail-launches itself
- This creates an infinite GPU-resident loop: schedule → work → schedule → work... with zero CPU involvement
- Only 1 pending self-tail-launch allowed at a time (prevents unbounded self-queuing / stack overflow on GPU)
- Tail launches from the SAME graph execute in enqueue order (FIFO) — so worker runs before self-relaunch
- Tail launches from NESTED graphs (child enqueuing more tails) run BEFORE the parent's remaining tails (LIFO/stack insertion)
- Max 255 pending tail launches total per graph
- Sibling launch = fire-and-forget but runs as a peer (not child) of parent graph, in parent's stream environment
- Key difference from fire-and-forget: sibling doesn't delay parent's tail launches (fire-and-forget child could)
- API: cudaGraphLaunch(graphExec, cudaStreamGraphFireAndForgetAsSibling)
- Decision guide: need results from graph → tail launch; side task → fire-and-forget/sibling; independent peer work → sibling
- Practical pattern: GPU kernel branches on data content and directly launches the appropriate prerecorded graph (e.g., LZ vs Huffman) — no CPU round-trip to decide

FIRE-AND-FORGET          TAIL LAUNCH              SIBLING
─────────────────────    ─────────────────────    ─────────────────────
Parent ██████████───→    Parent ██████████        Parent ██████████───→
       ↓                        ↓                        ↓
Child  ████████ (concurrent)   (parent done)     Sibling ████████ (concurrent)
       (is a child)            ↓                        (is a peer)
                         Graph █████████                 
                         (runs after parent)     Tail launches NOT blocked
                                                 by sibling

Atomic Queues and Device-Initiated CUDA Graphs for
In-Kernel Persistent Scheduling:

    ┌──────────────────────────────────────────────────────┐
    │  PERSISTENT SCHEDULER KERNEL (loops on GPU)          │
    │                                                      │
    │   while (work in queue) {                            │
    │       ┌──────────────────────────────────────┐       │
    │       │ idx = atomicAdd(&queueHead, 1)       │       │
    │       │        (claim next token)            │       │
    │       └──────────────────┬───────────────────┘       │
    │                          ▼                           │
    │       ┌──────────────────────────────────────┐       │
    │       │ TAIL-LAUNCH: Decode Graph            │       │
    │       │ (precaptured attention + FFN)        │       │
    │       └──────────────────┬───────────────────┘       │
    │                          ▼                           │
    │       (graph completes → loop back)                  │
    │   }                                                  │
    │                                                      │
    │   CPU: 😴 not involved                               │
    └──────────────────────────────────────────────────────┘

Conditional Graph Nodes:
    
- Conditional graph nodes = embed control flow (IF, IF/ELSE, WHILE, SWITCH) directly into CUDA Graphs, evaluated on GPU
- Controlled by a "condition handle" — a small integer set on-device via cudaGraphSetConditional(handle, flag)
- IF: runs body once if flag ≠ 0 | IF/ELSE: two body graphs, picks one | WHILE: loops body while flag ≠ 0 | SWITCH: N body graphs, runs the i-th one
- Setup: create handle → add upstream kernel that computes & sets handle → add conditional node → populate body subgraph(s)
- Set condition from a single thread to avoid races; ensure memory is flushed before conditional node reads it
- Conditional nodes can nest (e.g., WHILE body contains IF) for multilevel logic — all on GPU
- PyTorch has no Python API for conditional nodes yet — requires custom C++ integration

Dynamic Parallelism: Launch child kernels from the parent kernel based on the evalution of the parent kernel without using CPU resources -- consumes more GPU resources though. 

- Dynamic Parallelism (DP) = parent kernel spawns child kernels on GPU at runtime — no prerecorded graph needed
- Use case: irregular/data-dependent workloads where execution shape isn't known ahead of time (adaptive mesh, graph traversal, recursive problems)
- Diagnostic signal: profiler shows Kernel A → GPU idle gap → Kernel B (idle = CPU deciding next launch)
- Parent kernel doesn't complete until all its children complete (implicit sync, no cudaDeviceSynchronize needed)
- Launch from single thread (threadIdx.x==0 && blockIdx.x==0) to avoid duplicate child launches
- Compile with -rdc=true (relocatable device code)
- Costs: per-launch overhead (~25µs vs ~20µs host-side), extra stack space, max 2048 pending child launches (configurable via cudaLimitDevRuntimePendingLaunchCount)
- Bump stack if needed: cudaDeviceSetLimit(cudaLimitStackSize, newSize)
- Example result: 3 host launches → 1 host launch, ~40% idle → ~5% idle, ~25% faster overall
- Bonus: data locality preserved — intermediate results stay on GPU, no cache eviction from CPU round-trips
- DP is NOT always a win — if child kernel work is too small, device-side launch overhead negates the benefit. Always profile with Nsight Compute.

Decision guide:

CUDA Graphs          → execution shape known ahead of time, repeated many times (amortize capture cost)
Conditional Nodes    → known structure with runtime branching (IF/WHILE/SWITCH)
Device-Initiated Graph → known graphs, GPU decides WHICH to launch
Dynamic Parallelism  → execution shape unknown, emerges from data at runtime

Graphs for plans, DP for surprises

Orchestrate Acrsoss Multiple GPUs and Cluster Nodes(NVSHMEM):

- Node = one physical machine in a cluster (e.g., one DGX with 8 GPUs)
- Inter-node = across machines over InfiniBand (GPUDirect RDMA)
- Intra-node = within one machine over NVLink/NVSwitch (GPUDirect P2P)
- NUMA node is a different concept — refers to CPU socket + local memory affinity within one machine

- Core goal at multi-GPU scale: hide data movement behind compute (overlap is mandatory, not optional)
- Even NVLink is slower than on-device HBM — so P2P transfers MUST overlap with compute on separate streams
- P2P point-to-point: cudaMemcpyPeerAsync on a comm stream (copy engines, zero SM cost)
- Collectives (all-reduce): NCCL on a separate stream — auto-arranges rings/trees to saturate NVLink/NVSwitch
- Inter-node: CUDA-aware MPI auto-uses GPUDirect RDMA over InfiniBand (no CPU staging)
- Intra-node: GPUDirect P2P over NVLink or PCIe
- Pattern: stream 1 = compute, stream 2 = communication — both run concurrently

Fine-Grained GPU-to-GPU Memory Sharing with NVSHMEM: 

- NVSHMEM = GPU-level shared memory library using PGAS (Partitioned Global Address Space) — each GPU is a "PE" (processing element)
- Key capability: GPU kernel directly reads/writes another GPU's memory from device code, no CPU involved
- Pattern: put data → nvshmem_quiet() (ensure completion) → set flag on remote GPU; receiver spins on flag then reads data
- All communication is one-sided RMA (Remote Memory Access) — single hardware transaction over NVLink/PCIe
- vs cudaMemcpyPeerAsync: that's CPU-initiated (host tells copy engine); NVSHMEM is GPU-initiated (kernel code does the write)
- Achieves near-peak wire speed by bypassing CPU + kernel launch overhead
- Danger: it's GPU-level shared-memory programming — you own synchronization and race prevention
- Prefer fine-grained signals (nvshmem_wait_until, nvshmem_signal_*) over global barriers (nvshmem_barrier_all) to avoid stalling all GPUs on the slowest peer

- NVSHMEM kernels can be captured inside CUDA Graphs (operations are inside kernels, not separate graph nodes)
- Transformer pipeline example: GPU 0 computes attention → NVSHMEM puts activations to GPU 1 + signals → GPU 1 starts MLP; GPU 0 immediately starts next batch → both GPUs in perfect tandem, near-100% utilization
- Work-stealing pattern: nvshmem_int_atomic_inc on a global queue counter — each GPU grabs next task dynamically, no host coordination
- Cooperative launch: nvshmemx_collective_launch() starts same kernel on ALL GPUs simultaneously — required for device-side barriers/collectives
- nvshmem_barrier_all() inside cooperative kernel = lockstep execution across all GPUs (use sparingly — stalls on slowest GPU)

Capturing Multi-GPU Collectives with NCCL and CUDA Graphs:
- Without this, there is overhead of host nccl calls everytie -- instead put on a graph and keep replaying. 
- Capture forward + ncclAllReduce + backward into one CUDA Graph → replay with single cudaGraphLaunch per iteration
- All ranks MUST capture and replay the same NCCL sequence with the same communicator (mismatch → deadlock or silent corruption)
- Bucketed all-reduce: split gradients into chunks on streamA, all-reduce each chunk on streamB → compute/comms overlap hides network time
- PyTorch DDP does bucketed all-reduce automatically; wrapping in CUDA Graph further reduces CPU overhead + gives deterministic timing
- Practical rules: allocate gradient/comm buffers BEFORE capture; warm up collectives first; use cudaGraphExecUpdate to patch params without recapture
- At scale (100K+ GPUs), per-step CPU savings compound → tighter sync, higher utilization across cluster
- NCCL = bulk collectives (all-reduce, broadcast). NVSHMEM = fine-grained, async, point-to-point GPU↔GPU sharing

Pattern for N-GPU Scaling:
- Scaling pattern is always the same: dispatch once, overlap transfers with compute, keep CPU off critical path
- Ideal: N GPUs = N× speedup — only if all GPUs stay fed with data and communication is hidden behind compute
- Without overlap, scaling plateaus when comms time = compute time (Amdahl's law on communication)
- As GPU count grows: more aggressive pipelining, less CPU orchestration — offload to async copies, NCCL-in-graphs, NVSHMEM

Roofline-Guided Scheduling and Orchestration Decisions:
- Memory-bound kernel: kernel fusion helps modestly (fewer global mem round-trips); real gains from latency masking + more loads/stores in flight + reduce precision (FP16/FP8/FP4)
- Compute-bound kernel: fusion shines (combine ops → higher FLOPS/byte → shift right on roofline); reduce launch overhead with persistent kernels, CUDA Graphs, device-side launches to keep ALUs fed
- Neither bound (middle of roofline): increase concurrency — parallel streams, concurrent graphs, multiple persistent kernels → aggregate throughput climbs on both axes
- Persistent kernels, graphs, device-initiated launches don't change arithmetic intensity — they reduce idle gaps, pushing actual perf closer to the ceiling
- Always: Nsight Compute to measure FLOPS + bytes → plot on roofline → pick strategy based on which roof you're under → measure again after each change

ROOFLINE MODEL:
                          ┌───────────────── compute ceiling (peak FLOPS)
  Performance             │
  (FLOPS/s)     ─────────/
               /         │
              / ←slope   │
             /  (memory  │
            /   ceiling) │
           /             │
          /              │
         /               │
        ──────────────────────────────→
                  Arithmetic Intensity (FLOPS/byte)


CASE 1: Below the slope
  ×  ← your kernel (below the line)
  /
  You're NOT hitting peak memory bandwidth
  = memory-INEFFICIENT (bad access patterns, uncoalesced, misaligned)
  Fix: coalesce, align, SoA, vectorized loads


CASE 2: On the slope
  /
  × ← your kernel (on the line)
  You ARE hitting peak memory bandwidth
  = memory-BOUND (bandwidth wall reached)
  Can't squeeze more bandwidth, BUT you can:
    → Shift RIGHT: reduce precision (FP16→FP8), fuse ops, reuse data in registers/smem
    → This increases FLOPS/byte → moves you toward the flat (compute) ceiling


CASE 3: Below the flat ceiling
         ─────────────
         × ← your kernel (below the flat part)
  = compute-INEFFICIENT (low occupancy, idle gaps, launch overhead)
  Fix: persistent kernels, CUDA Graphs, increase occupancy


CASE 4: On the flat ceiling
         ──────×──────
  = compute-BOUND (ALUs maxed out)
  Only fix: better algorithm or faster hardware

Chapter 12 done!