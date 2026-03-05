Profiling, Tuning, and Scaling PyTorch:

PROFILING TOOLS — CHAPTER 13 INTRO
──────────────────────────────────────────────

NVTX MARKERS
──────────────────────────────────────────────
- Tags you inject into code: "forward", "backward", "optimizer_step"
- Show up in ALL profilers (PyTorch profiler, Nsight, etc.)
- Lets you correlate events across tools on the same timeline
- PyTorch: torch.profiler.record_function() or torch.cuda.nvtx.range_push()

TOOL MAP — WHAT TO USE WHEN
──────────────────────────────────────────────
Level           Tool              What it tells you
-----------     ---------------   ----------------------------------
Python ops      PyTorch Profiler  Which ops are slow, shapes, memory
                (Kineto)          graph breaks, forward/backward split

Full system     Nsight Systems    CPU+GPU+I/O unified timeline
                (nsys)            data loader stalls, CPU-GPU overlap

Single kernel   Nsight Compute    Hardware counters per kernel
                (ncu)             memory bound? compute bound? why?

GPU memory      PyTorch memory    Per-op peak memory, fragmentation
                profiler          which op allocates the most

CPU side        Linux perf        GIL contention, data loading bottleneck
                                  CPU cache misses, thread scheduling

Multi-GPU       HTA               Distributed trace, idle GPU detection
                                  worker imbalance, comm vs compute

Trace viewing   Perfetto          Web UI for traces, SQL queries
                                  sharing across teams

Metrics         TorchEval         Throughput, latency, accuracy logging

Edge/mobile     ExecuTorch        Lightweight profiling on embedded devices


TYPICAL WORKFLOW
──────────────────────────────────────────────
Step 1: PyTorch Profiler  -->  "which ops are slow?"
Step 2: Nsight Systems    -->  "is it CPU, GPU, I/O, or overlap?"
Step 3: Nsight Compute    -->  "WHY is this specific kernel slow?"
Step 4: Fix and re-profile

Profiling PyTorch to Identify Bottlenecks:

SETUP
──────────────────────────────────────────────
- Load MoE transformer via Hugging Face, move to GPU
- Warm up 5 iterations first (JIT compile, fill caches) — don't profile these
- Profile ONE training iteration with:
  - record_shapes=True     (tensor shapes per op)
  - profile_memory=True    (GPU memory per op)
  - with_stack=True        (Python call stacks)
  - with_flops=True        (FLOP counters)
- Mark phases with NVTX: "forward", "backward", "optimizer_step"

TAKEAWAY
──────────────────────────────────────────────
prof.key_averages().table(sort_by="self_cuda_time_total")
--> instantly shows which ops to optimize first
--> here: reduce expert GEMM count or fuse dispatch/combine

GPU PROJECTION = mapping NVTX ranges to ACTUAL GPU work
──────────────────────────────────────────────

The problem:
 NVTX range "forward" is recorded on the CPU timeline
 But GPU work is ASYNCHRONOUS — kernels launch later, finish later

 CPU timeline:  [====forward====][====backward====]
 GPU timeline:       [kern1][kern2][kern3]  [kern4][kern5]
                ↑ gap                   ↑ some "forward" kernels
                                          might still run during
                                          "backward" on CPU

GPU projection:
 Takes each GPU kernel and asks:
 "which NVTX range was active on the CPU when this kernel was LAUNCHED?"
 Then PROJECTS that kernel's GPU execution time back onto the NVTX range

 So instead of:
   "forward" CPU time = 15ms  (just CPU-side enqueue time)

 You get:
   "forward" GPU projected time = 43ms  (actual GPU compute attributed to "forward")

 This tells you how much GPU work each phase ACTUALLY caused

System Profiling with NSight Systems and NVTX Timelines: 

NSIGHT SYSTEMS + NVTX PROFILING — SUMMARY
──────────────────────────────────────────────

COMMANDS
──────────────────────────────────────────────
nsys profile -t cuda,nvtx python train.py    --> capture trace
nsys stats --report=nvtx_gpu_proj_sum ...    --> print GPU time per NVTX range
Can plug into CI/CD to catch performance regressions automatically

MULTI-GPU NOTE
──────────────────────────────────────────────
- Nsight Systems timeline shows NVLink, NCCL, InfiniBand activity
- Gaps/bubbles in timeline = GPU starved (waiting for comms or data)
- HTA does the same for distributed traces across nodes
- Use both to verify GPUs stay busy in large clusters (NVL72, etc.)

KERNEL ROOFLINE ANALYSIS — NSIGHT COMPUTE (ncu)
──────────────────────────────────────────────

COMMAND
──────────────────────────────────────────────
ncu --kernel-name-regex "matmul" --metrics <hw counters> python train.py
--> targets specific kernels by name, collects hardware counters

KEY METRICS COLLECTED
──────────────────────────────────────────────
gpu__dram_throughput           % of peak HBM bandwidth
lts__throughput                % of peak L2 bandwidth
sm__sass_thread_inst_executed  FP32 instruction count (compute proxy)
sm__warps_active               achieved occupancy
gpu__time_duration             kernel execution time

OCCUPANCY NOTE
──────────────────────────────────────────────
- No universal target — some kernels hide latency fine at 25-50%
- Use stall-reason breakdown + eligible warps per cycle to judge
- Low occupancy is only a problem if warps can't cover memory stalls

COMMANDS
──────────────────────────────────────────────
perf stat -e cycles,instructions,cache-misses,branch-misses python train.py
--> quick CPU health check (IPC, cache/branch miss rates)

perf record -F 2000 -g --call-graph dwarf python train.py
perf report --stdio -n -g -i perf.data
--> detailed call graph: where CPU time goes

RESULTS — CPU TIME BREAKDOWN
──────────────────────────────────────────────
Hotspot                  %      Fix
-----------              ---    --------------------------------
py::forward              45%    torch.compile (eliminate interpreter)
aten::matmul             20%    fused CUDA kernel or compiler
dataloader_iter_next     10%    more num_workers, persistent_workers
ncclAllReduce             9%    bigger bucket_cap_mb, gradient compression
read syscalls             5%    pin_memory, non_blocking, shard formats

KEY RULE: apply ONE optimization at a time, measure, then next

NVIDIA PMU VIA LINUX PERF
──────────────────────────────────────────────
perf stat -e nvidia_nvlink_c2c0_pmu_0/cycles/ python train.py
--> correlate NVLink/C2C activity with CPU behavior
--> only link-level and fabric-level counters (bandwidth, cycles)

+---------------------+---------------------------+
| Linux perf PMU      | Nsight Compute (CUPTI)    |
+---------------------+---------------------------+
| NVLink traffic      | SM occupancy              |
| C2C cycles          | warp stall reasons        |
| fabric requests     | memory bandwidth per kern |
| link utilization    | roofline analysis         |
+---------------------+---------------------------+
| "how busy is the    | "why is this kernel slow" |
|  wire between chips" |                          |
+---------------------+---------------------------+


TORCH.COMPILE — SUMMARY
──────────────────────────────────────────────

WHAT IT DOES
──────────────────────────────────────────────
model = torch.compile(model)

Python code                    Compiled code
-----------                    -------------
op1: add                       ┐
op2: relu          ──compile──→ │ ONE fused kernel
op3: layer_norm                │ (fewer launches,
op4: dropout                   ┘  less memory traffic)

Stack:
TorchDynamo    --> traces Python, captures the graph
AOT Autograd   --> captures backward pass too
TorchInductor  --> generates optimized GPU kernels (via Triton)

Result: 248ms (eager) --> 173ms (compiled) = ~30% faster


EAGER MODE vs COMPILED MODE
──────────────────────────────────────────────

EAGER MODE (default PyTorch):
  Python runs line by line, each op launches a GPU kernel immediately

  Python:  op1 → launch kern1 → op2 → launch kern2 → op3 → launch kern3
  GPU:          [kern1]              [kern2]              [kern3]
                       ↑ gap               ↑ gap
                   Python interpreter overhead between each op

COMPILED MODE:
  Compiler sees the whole graph, fuses ops, launches fewer kernels

  Python:  compiled_model(x) → launch fused_kern
  GPU:     [====fused_kern====]
           ↑ one kernel does what 3 did before
             no Python between ops, no gaps


WHY CACHE? ISN'T IT COMPILED?
──────────────────────────────────────────────

torch.compile is NOT like gcc producing a .exe that lives forever

It compiles AT RUNTIME (JIT = just-in-time):

  Run 1:                              Run 2 (no cache):
  start python                        start python
      |                                   |
  torch.compile(model)                torch.compile(model)
      |                                   |
  [===compile: 30-120 sec===]         [===compile AGAIN: 30-120 sec===]
      |                                   |
  run fast                            run fast

  The compiled kernels live in memory — gone when process exits

With Mega-Cache:
  Run 1:                              Run 2 (with cache):
  start python                        start python
      |                                   |
  torch.compile(model)                torch.compile(model)
      |                                   |
  [===compile: 30-120 sec===]         [load from disk: <1 sec]
      |                                   |
  save_cache_artifacts() --> disk     run fast immediately
      |
  run fast

  Cache = save JIT output to disk so you don't recompile every run
  Unlike gcc, the "binary" is tied to exact model + shapes + GPU + versions
  Change any of those --> must recompile

The cache doesn't make compilation faster — it skips compilation entirely by reloading the already-compiled kernels. But if anything in the kernel is changed, it must recompile.  

PROLOGUE and EPILOGUE
──────────────────────────────────────────────

A typical transformer layer:

  input
    |
    v
  [bias add]           <-- PROLOGUE (setup before the heavy op)
    |
    v
  [GEMM / matmul]      <-- MAIN OP (the expensive part)
    |
    v
  [activation (ReLU)]  <-- EPILOGUE (cleanup after the heavy op)
  [dropout]            <--
  [residual add]       <--
    |
    v
  output

Without fusion (eager):                With fusion (compiled):
  kern1: bias add         --> HBM        ONE kernel:
  kern2: matmul           --> HBM          read input from HBM
  kern3: relu             --> HBM          bias add (registers)
  kern4: dropout          --> HBM          matmul
  kern5: residual add     --> HBM          relu (registers)
                                           dropout (registers)
  5 kernels, 5 HBM round trips            residual add (registers)
                                           write output to HBM

                                         1 kernel, 1 HBM round trip

  Prologue + main + epilogue fused = less memory traffic, fewer launches


COMPILE MODES
──────────────────────────────────────────────
Mode               Compile time    Runtime speed    What it does
-----------        ------------    -------------    ----------------------
"default"          medium          medium           balanced
"reduce-overhead"  medium          fast             uses CUDA Graphs
"max-autotune"     slow            fastest          tries many kernel variants

2/25/26: 

TORCH.COMPILE RESULTS + MODES — SUMMARY
──────────────────────────────────────────────

WHY MoE BENEFITS MORE THAN DENSE MODELS
──────────────────────────────────────────────
MoE: hundreds of small ops (dispatch, combine, per-token activations)
  --> many kernel launches, much Python overhead
  --> compiler fuses them --> big win (~30%)

Dense: one massive GEMM dominates (already using Tensor Cores well)
  --> little to fuse --> small win (<10%)

CUSTOM KERNELS vs COMPILER
──────────────────────────────────────────────
Priority order:
  1. torch.compile first           (free, easy, covers most cases)
  2. Custom Triton/CUDA kernel     (only for ops compiler can't handle)
  3. Register custom op in compile (embed hand-tuned kernel in compiled graph)

You can mix both: compiler optimizes surrounding ops,
custom kernel handles the hot spot — best of both worlds


COMPILATION MODES
──────────────────────────────────────────────
Mode                       Compile   Speed   CUDA    Best for
                           time              Graphs
-----------------------    -------   -----   ------  -------------------------
default                    low       good    maybe   first try, large models
reduce-overhead            medium    fast    yes     small batches, inference
max-autotune               high      best    yes     long training, fixed shapes
max-autotune-no-cudagraphs high      best    no      dynamic shapes, debugging

Mode                        Description                                                          Compile time    Extra memory              Notable features
--------------------------  -------------------------------------------------------------------  --------------  ------------------------  ------------------------------------------
default                     Balanced optimizations (good speed without long compile or extra      Low-Medium      No                        General fusion, basic autotuning
                            mem); includes minor autotuning; may use CUDA Graphs for stable
                            segments

reduce-overhead             Reduces per-iteration overhead (good for small batches); ideal for    Medium          Yes (workspace caching)   Uses CUDA Graphs (if possible) to
                            inference or small batches; automatically skips CUDA Graphs if it                                               eliminate launch overhead
                            detects dynamic shapes to preserve correctness

max-autotune                Maximizes runtime performance (best for long runs); longer compile    High (slow      Maybe (if graphs are      Aggressive Triton autotuning; enables
                            time; best for aggressive tuning for a large amount of SMs and GPU    compile)        used)                     CUDA Graphs on GPU
                            memory

max-autotune-no-cudagraphs  Does everything max-autotune does but without CUDA Graph capture.     High            No                        Same as max-autotune but disables
                            Best for dynamic shapes or for debugging issues masked by CUDA                                                  graphs for flexibility
                            Graphs


                     compile time --->
  default -----> reduce-overhead -----> max-autotune
  least effort                          most tuning
  good enough                           best runtime

KEY CONSTRAINTS
──────────────────────────────────────────────
- CUDA Graphs modes (reduce-overhead, max-autotune):
  need fixed shapes, fixed memory addresses, no dynamic control flow
  --> compiler auto-falls-back to eager if it detects dynamic shapes

- Dynamic token routing (MoE):
  use default or max-autotune-no-cudagraphs first
  switch to max-autotune once shapes stabilize

- max-autotune can sometimes REGRESS latency — always profile each mode

DECISION FLOW
──────────────────────────────────────────────
Start: torch.compile(model)                        (default)
  |
  Still slow?
  |-- small batch / inference --> mode="reduce-overhead"
  |-- long training job       --> mode="max-autotune"
  |-- dynamic shapes          --> mode="max-autotune-no-cudagraphs"
  |
  Still slow on specific op?
  |-- write custom Triton/CUDA kernel, register as custom op

REGIONAL COMPILATION
──────────────────────────────────────────────

Without regional:
  Block 0: [compile] ← 10s
  Block 1: [compile] ← 10s    (same code, compiled again)
  Block 2: [compile] ← 10s    (same code, compiled again)
  ...
  Block 95: [compile] ← 10s
  Total: 96 × 10s = 960s compile time

With regional (torch.compiler.nested_compile_region):
  Block 0: [compile] ← 10s
  Block 1: [reuse]   ← ~0s    (same compiled code)
  Block 2: [reuse]   ← ~0s
  ...
  Block 95: [reuse]  ← ~0s
  Total: 10s compile time

- Same runtime speed, just faster startup
- Auto-recompiles if shapes/dtype/device change (correctness preserved)
- Best for: transformers/MoEs with many identical layers, inference, short jobs

DEBUGGING TORCH.COMPILE ISSUES
──────────────────────────────────────────────

Graph break = compiler gives up on a section, falls back to eager (slow)

DIAGNOSTIC TOOLS
──────────────────────────────────────────────
Tool/command                                What it does
------------------------------------------  ----------------------------------
torch._dynamo.explain(model)                Lists graph breaks + reasons + suggestions
TORCH_LOGS="+dynamo,+inductor"              Verbose logs: breaks, fallbacks, compile phases
torch._dynamo.mark_dynamic(tensor, dim)     Tell compiler to expect variable shape on dim
TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1         Name kernels for profiling
TORCHINDUCTOR_BENCHMARK_KERNEL=1            Auto-generate benchmark harness per kernel

Common graph break causes:
 - Unsupported ops
 - Data-dependent control flow (if tensor.item() > 0: ...)
 - Dynamic shapes without shape guards

Fix options:
 - Replace unsupported op with supported equivalent
 - mark_dynamic for known variable dimensions -- mark regions as dynamic ones so that the compiler knows it
 - Split model: compile static parts, run dynamic parts eagerly
 - Use max-autotune-no-cudagraphs for truly dynamic workloads


PYTORCH ATTENTION MECHANISMS
──────────────────────────────────────────────

API                  What it does                              When to use
-------------------  ----------------------------------------  ----------------------------
SDPA                 Auto-selects fastest attention backend    Default choice, no-hassle
                    (Flash, memory-efficient, or math)        standard attention patterns

FlexAttention        Compiler-generated kernels for custom     Block-sparse, sliding-window,
                    sparse attention patterns                 any pattern SDPA can't handle

FlexDecoding         Optimized autoregressive decoding         LLM inference, long generation
                    + KV cache management, works with         sequences, compile-time
                    torch.compile                             decode optimization

Context Parallel     Shards attention along sequence length    Scaling to very long contexts
                    across multiple GPUs/ranks                across multiple devices
                    (splits QKV by sequence, syncs            (not within-GPU parallelism)
                    during attention)

TORCHAO — PYTORCH ARCHITECTURE OPTIMIZATION
──────────────────────────────────────────────
- Single namespace for quantization, sparsity, and pruning

torchao.quantization:  PTQ, QAT, INT8, FP8 (reduces memory + speeds up compute)
torchao.pruning:       remove weights (smaller model)
torchao.sparsity:      2:4 structured sparsity, block sparsity (hardware-accelerated)

- Integrates with torch.compile: TorchInductor emits kernels that use torchao
- Works for both training and inference

- CUDA streams in PyTorch
  - By default, all ops go on stream 0 (default stream) → sequential execution
  - Create non-default streams with `torch.cuda.Stream()` to run independent work concurrently
  - Use `with torch.cuda.stream(stream):` context manager to enqueue ops on a specific stream

- Overlapping H2D transfer + compute (double-buffering pattern)

  CPU (Python) enqueues everything non-blocking, never waits:
  ┌────────────────────────────────────────────────────────────────┐
  │ 1. compute_stream.wait_stream(transfer_stream) → GPU fence    │
  │ 2. Enqueue batch i+1 H2D copy on transfer_stream              │
  │ 3. Enqueue fwd/bwd of batch i on compute_stream               │
  │ 4. Loop back immediately                                      │
  └────────────────────────────────────────────────────────────────┘

  GPU executes on separate hardware units concurrently:
                     time ────────────────────────────────────►
  transfer_stream:  [H2D batch i+1 (copy engine)] [H2D batch i+2]
  compute_stream:   [fwd/bwd batch i (SMs)]        [fwd/bwd batch i+1]

- Key mechanisms
  - `compute_stream.wait_stream(transfer_stream)` = GPU-side-only fence
    - Compute stream waits for transfer to finish before using that batch
    - Transfer stream is NOT blocked — keeps copying next batch
    - CPU/Python thread is NOT blocked — just inserts a marker and moves on
  - `.to(device, non_blocking=True)` = async DMA copy, doesn't block CPU thread
  - `next(dataloader_iter, None)` = explicit control over when transfers are enqueued
  - One batch always in-flight per stream → decouples loading from compute

- Profiling tip
  - Use Nsight Systems or PyTorch profiler to verify overlap
  - Look for: transfer and compute lanes running in parallel → near 100% GPU utilization
  - Watch for: implicit syncs like `print()` on CUDA tensors or extra `synchronize()` calls that serialize everything

2/26/26: 
  - CUDA Graphs in PyTorch = capture once, replay many times → eliminates per-iteration CPU launch overhead

- Capture pattern

  1. Preallocate static_input, static_output at max required size
  2. Warm up on a dedicated non-default capture_stream (allocates all internal buffers)
  3. capture_stream.synchronize()
  4. Capture: torch.cuda.graph(g, stream=capture_stream) — records ops, doesn't execute
  5. capture_stream.synchronize()

  ┌─────────────────────────────────────────────────────────┐
  │ Preallocate → Warm-up → sync → Capture → sync → Replay │
  └─────────────────────────────────────────────────────────┘

- Replay pattern
  static_input.copy_(new_batch)   ← load new data into same memory
  g.replay()                      ← GPU runs captured op sequence
  result = static_output.clone()  ← clone if you need to keep it (graph overwrites each replay)

- Critical constraints
  - NO new memory allocations during capture — all buffers must exist from warm-up
  - Strict isolation: no CUDA ops on other streams during capture, or you get "operation not permitted"
  - Changing model weights outside the graph won't take effect — must recapture
  - Static shapes only (use max-autotune-no-cudagraphs for dynamic shapes)
  - torch.cuda.empty_cache() before capture = one-time defrag, NOT a regular tool

- Synchronization
  - g.replay() is async — must sync before reading static_output
  - Don't print() or .item() on GPU tensor mid-pipeline → use .cpu().numpy() after sync, outside async regions

- Memory pools
  - Each graph instance gets its own private memory pool by default → no contention between concurrent graphs
  - cudaMallocAsync reuses same addresses on each replay (no new allocations needed)
  - Optional: torch.cuda.graph(pool=...) to share a pool across graph instances

- Shared pool trick (FireworksAI)
  - Multiple batch-size-specialized graphs (1, 2, 4, 8) share one pool
  - Compile in decreasing batch size order → largest variant allocates the pool
  - Smaller variants reuse the larger allocation → saves GPU memory in serving

- CUDA Graphs best practices

  1. No allocations during capture — preallocate all buffers (including optimizer states!) in warm-up
  2. Fixed graph structure — no shape/op changes after capture
     - Multiple shapes? Capture separate graphs per shape, select at runtime
     - Or use max-autotune-no-cudagraphs mode
  3. Capture as much as possible — ideally entire training iteration (fwd + bwd + optimizer + allreduce)
     - MLPerf submissions capture everything into one graph per iteration
     - Can even capture multiple iterations (loop unrolling) if memory allows
  4. Plan for max size — capture with worst-case batch size to avoid graph failure
     - e.g., max batch 64 but sometimes 96 → capture at 96, waste memory instead of breaking
  5. Stream priorities — in multi-GPU, set NCCL to lower-priority stream so compute kernels run first
  6. Graphs are immutable in PyTorch — cudaGraphExecUpdate() exists in CUDA but PyTorch doesn't expose it

- CUDA Graph Trees (torch.compile internal)
  - Used by mode="reduce-overhead" automatically
  - Per-shape piecewise capture: each unique input shape gets its own cached graph
  - Fewer distinct shapes → more cache hits → less capture overhead
  - Multiple subgraphs share a single memory pool (fwd + bwd captures)
  - Enables dynamic subgraph selection at runtime based on shape/batch size

  Shape A → Graph A (cached) ─┐
  Shape B → Graph B (cached) ─┼── shared memory pool
  Shape C → Graph C (cached) ─┘
              ↑ vLLM uses this for variable batch sizes in inference

- Why full-graph capture is hard for LLM inference
  - Variable input sizes, batch sizes, KV cache growth, host-side decisions
  - Piecewise capture (Graph Trees) is the workaround

- Memory profiling intro
  - torch.profiler with profile_memory=True → shows per-op memory allocations
  - PyTorch memory visualizer tool → visual timeline of memory usage
  - Nsight Systems CUDA Memory Inspector → visualizes fragmentation over time

- Tuning the CUDA memory allocator

  Problem: variable-sized allocs (e.g., MoE expert outputs) → fragmentation
  Fix 1: Preallocate fixed max-size buffers, reuse every iteration
  Fix 2: Tune allocator via PYTORCH_ALLOC_CONF:

  ┌─────────────────────────────────────────────────────────────────┐
  │ max_split_size_mb:256     → keep free blocks ≤256MB intact     │
  │                             (don't split into tiny pieces)     │
  │                                                                │
  │ roundup_power2_divisions  → bucket allocation sizes            │
  │   [256:1, 512:2, 1024:4] → fewer unique sizes = more reuse    │
  │                                                                │
  │ backend:cudaMallocAsync   → async free (no sync on dealloc)    │
  │                             good for multi-worker data loading │
  └─────────────────────────────────────────────────────────────────┘

  Monitor: torch.cuda.memory_stats() → fragmentation over time
           torch.cuda.mem_get_info() → free vs total (indirect fragmentation check)

- Activation checkpointing (gradient checkpointing)

  Problem: storing all intermediate activations for backward pass → OOM on large models

  Without checkpointing:
    Forward:  [Layer 1] → save act1 → [Layer 2] → save act2 → ... → save actN
    Memory:   act1 + act2 + ... + actN all in HBM simultaneously → OOM

  With checkpointing:
    Forward:  [Layer 1] → discard act1 → [Layer 2] → discard act2 → ... → output
    Backward: need act3? → rerun Layer 3 forward → compute grad → discard again
    Memory:   only 1 layer's activations at a time → fits in HBM

  Tradeoff:
    ┌──────────────────┬──────────────────────────────────┐
    │ Save memory       │ Costs extra compute (recompute)  │
    │ Larger batch size │ ~30% more FLOPs per iteration    │
    │ Larger models fit │ Modern GPUs have FLOPS to spare  │
    └──────────────────┴──────────────────────────────────┘

  Strategy: checkpoint only the big layers (transformer blocks), skip small ones (layernorm, embedding)
  In FSDP: automated recursive checkpointing available

- Offloading parameters to CPU / NVMe

  When model still doesn't fit after checkpointing → offload inactive params

  ┌────────────────────────────────────────────────────────────────┐
  │              GPU HBM (fast, limited)                          │
  │   [Active layer weights] [Current activations]               │
  │         ↑ prefetch next layer                                │
  │         │ async DMA (.to(device, non_blocking=True))         │
  │─────────┼────────────────────────────────────────────────────│
  │         │    CPU DRAM (slower, larger)                        │
  │   [Inactive expert weights] [Optimizer states]               │
  │         ↑ swap from NVMe if needed                           │
  │─────────┼────────────────────────────────────────────────────│
  │         │    NVMe/SSD (slowest, largest) — last resort       │
  │   [Rarely used params] [Overflow]                            │
  └────────────────────────────────────────────────────────────────┘

  Key: overlap transfer with compute so GPU never stalls:
    GPU:  [compute layer i]──────[compute layer i+1]──────
    DMA:     [prefetch layer i+1]───[prefetch layer i+2]───

  Frameworks: DeepSpeed ZeRO-Infinity (train), ZeRO-Inference (serve) — automate this
  Hardware options:
    - Pin host memory + cudaMemcpyAsync → explicit, predictable
    - Unified Memory (Grace Blackwell NVLink-C2C) → automatic page migration, but unpredictable
    - GPUDirect Storage → GPU reads NVMe directly, no CPU involvement

- SuperOffload (superchip-optimized offloading)
  - Designed for CPU-GPU superchips (Grace Hopper, Grace Blackwell, Vera Rubin)
  - Exploits NVLink-C2C high-bandwidth coherent interconnect between CPU and GPU

  Key innovations:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1. STV (Speculation-Then-Validation)                                │
  │    CPU starts optimizer update DURING backward (speculative)        │
  │    Validates after allreduce finishes → removes optimizer from      │
  │    critical path                                                    │
  │                                                                     │
  │ 2. Heterogeneous optimizer split                                    │
  │    GPU: heavy tensor updates (weight = weight - lr × m/√v)         │
  │    CPU: light state updates (momentum buffers, variance tracking)   │
  │    → both devices busy, no idle cycles                              │
  │                                                                     │
  │ 3. Superchip-aware type casts + copies                              │
  │    Shift FP32↔BF16 conversions toward GPU (faster at math)          │
  │    NVLink-C2C makes CPU↔GPU transfer cheap → changes placement      │
  │    strategy for what goes where                                     │
  │                                                                     │
  │ 4. GraceAdam                                                        │
  │    CPU-optimized Adam using ARM SVE (Scalable Vector Extension)     │
  │    32 vector registers + 16 predicate registers → fast vectorized   │
  │    optimizer on Grace ARM cores                                     │
  └──────────────────────────────────────────────────────────────────────┘

- FSDP (Fully Sharded Data Parallel) = ZeRO Stage-3 in PyTorch
  - Shards parameters + gradients + optimizer states across GPUs

  Without FSDP (DDP):
    GPU 0: [full model + full optimizer states]  ← duplicated on every GPU
    GPU 1: [full model + full optimizer states]

  With FSDP:
    GPU 0: [shard 0 of params/grads/optimizer]  ← each GPU holds 1/N
    GPU 1: [shard 1 of params/grads/optimizer]
    → allgather before fwd/bwd, reduce-scatter after → each GPU only stores its shard

  Built-in automation (no boilerplate needed):
  ┌──────────────────────────────────────────────────────────────┐
  │ auto_wrap_policy       → auto-shard by transformer block     │
  │ CPUOffload             → move params/grads to CPU when idle  │
  │ activation_checkpointing_policy → recompute activations for  │
  │                          specified layers (TransformerBlock)  │
  │ mixed_precision        → BF16 for params/reduce/buffers      │
  │ backward_prefetch      → BACKWARD_PRE = prefetch next shard  │
  │                          during backward (overlap comm+comp) │
  │ use_orig_params=True   → no flattening, better overlap       │
  └──────────────────────────────────────────────────────────────┘
- Why sharding exists
  - Model state (params + grads + optimizer) >> single GPU memory
  - 70B model + Adam = ~1,120 GB total; H100 = 80 GB → must split across GPUs

- FSDP sharding strategies

  ┌─────────────────┬───────────────┬──────────────────┬─────────────────────────┐
  │ Strategy         │ ZeRO Stage    │ What's sharded   │ When to use             │
  ├─────────────────┼───────────────┼──────────────────┼─────────────────────────┤
  │ SHARD_GRAD_OP    │ Stage 2       │ Grads + optimizer │ Few GPUs / slow network │
  │                  │               │ (params = full    │ Least comms, most       │
  │                  │               │  copy everywhere) │ memory per GPU          │
  ├─────────────────┼───────────────┼──────────────────┼─────────────────────────┤
  │ FULL_SHARD       │ Stage 3       │ Params + grads +  │ Fast fabric (NVLink     │
  │                  │               │ optimizer — ALL   │ across nodes), want     │
  │                  │               │ across ALL GPUs   │ smallest memory/GPU     │
  ├─────────────────┼───────────────┼──────────────────┼─────────────────────────┤
  │ HYBRID_SHARD     │ 3 within node │ Shard within node │ Fast intra-node (NVLink)│
  │                  │ replicate     │ Replicate across  │ Slower inter-node (IB)  │
  │                  │ across nodes  │ nodes             │ Trade memory for comms  │
  └─────────────────┴───────────────┴──────────────────┴─────────────────────────┘

- HYBRID_SHARD = spend memory to save comms

  FULL_SHARD (16 GPUs across 2 nodes):
    GPU 0 needs shard 12 → fetch from Node 1 over slow InfiniBand → bottleneck

  HYBRID_SHARD (same 16 GPUs):
    Each node holds full model sharded across its 8 local GPUs (replicated)
    GPU 0 needs any shard → fetch from same node over fast NVLink → no cross-node allgather
    Cross-node traffic = only gradient reduce (smaller, less frequent)

    ┌─── Node 0 (NVLink 900 GB/s) ───┐     ┌─── Node 1 (NVLink 900 GB/s) ───┐
    │ GPU0[s0] GPU1[s1] ... GPU7[s7]  │     │ GPU0[s0] GPU1[s1] ... GPU7[s7]  │
    │      allgather stays HERE        │     │      allgather stays HERE        │
    └──────────────┬───────────────────┘     └──────────────┬───────────────────┘
                   └───── only grad reduce crosses here (InfiniBand 400 GB/s) ──┘

  Cost: each node stores full model (more memory)
  Gain: allgather never crosses slow inter-node fabric

- FSDP built-in automation (from previous section)
  - activation_checkpointing_policy → recompute activations for specified transformer layers
  - CPUOffload → move params/grads to CPU when idle
  - backward_prefetch=BACKWARD_PRE → prefetch next shard during backward
  - mixed_precision → BF16 for params/reduce/buffers
  - Handles uneven per-GPU batch sizes (useful for MoE)

2/28/26:

DEFAULT:
  NCCL asks PyTorch's general allocator for buffers
  → gets regular GPU memory
  → works, but hardware (copy engines, NIC, switch) may need extra steps to access it

CUSTOM ALLOCATOR:
  NCCL gets buffers from its OWN allocator (ncclMemAlloc)
  → memory is pre-registered, page-aligned, hardware-accessible
  → copy engines, NIC (RDMA), SHARP switch can grab data directly
  → = zero-copy

- Blackwell NVLink 5: 1.8 TB/s bidirectional (~900 GB/s each direction) — custom allocators help approach this peak
- CUDAPluggableAllocator: load any custom .so with alloc/free symbols → use as a CUDA allocator in PyTorch
- MemPool wraps the allocator with caching (avoids repeated alloc/free overhead)
- torch.cuda.use_mem_pool(pool): context manager swaps allocator for all allocations inside the block; restores default on exit
- You can mix allocators in one app: default cudaMalloc for compute tensors, NCCL allocator for comm buffers
- Debugging: torch.cuda.memory._record_memory_history() + _dump_snapshot() → visualize in PyTorch memory viewer

DMA (Direct Memory Access) = GENERAL CONCEPT
═══════════════════════════════════════════════
  A dedicated hardware engine moves data — CPU and SMs not involved
  This is NOT a specific technology, it's a category


GPUDirect P2P = DMA WITHIN a node (same machine)
═══════════════════════════════════════════════════

  ┌─────────┐     NVLink / PCIe      ┌─────────┐
  │ GPU A   │ ══════════════════════► │ GPU B   │
  │ HBM     │   copy engine (DMA)    │ HBM     │
  └─────────┘   does the transfer    └─────────┘

  Copy engine = the DMA hardware on the GPU
  Path: GPU memory → wire → GPU memory
  No CPU RAM, no SMs


GPUDirect RDMA = DMA ACROSS nodes (different machines)
═══════════════════════════════════════════════════════

  Node 0                                          Node 1
  ┌─────────┐    ┌─────┐   InfiniBand   ┌─────┐  ┌─────────┐
  │ GPU A   │══► │ NIC │ ════════════► │ NIC │══►│ GPU B   │
  │ HBM     │    └─────┘               └─────┘   │ HBM     │
  └─────────┘                                     └─────────┘

  NIC reads GPU memory DIRECTLY (no CPU RAM staging)
  Path: GPU memory → NIC → network wire → NIC → GPU memory
  "RDMA" = Remote DMA — the DMA happens across a network

  WITHOUT GPUDirect RDMA (old way):
  GPU A → CPU RAM → NIC → wire → NIC → CPU RAM → GPU B
  ^^^^    ^^^^^^^^                      ^^^^^^^^   ^^^^
  2 extra hops through CPU memory on each side

3/2/26

- Peer-to-peer (P2P) DMA = direct GPU-to-GPU transfer, no CPU involved
  - PyTorch auto-enables when moving tensors across devices
  - Verify: torch.cuda.can_device_access_peer(i, j)
  - Uses cudaMemcpyPeerAsync under the hood
  - Works over NVLink/NVSwitch within a node or rack (e.g., NVL72)

- When P2P DMA works (same node/rack, NVLink connected):
  GPU 0 ──NVLink──► GPU 1    (direct, no CPU, no network)
  dst.copy_(src, non_blocking=True)  → uses P2P path automatically

- When P2P DMA doesn't work (cross-node, cross-rack):
  GPU on Node 0 ──?──► GPU on Node 1
  No NVLink between nodes → must go through network fabric
  → NCCL handles this, but by default uses basic transport

- UCX = NCCL plugin for cross-node/cross-rack GPU communication
  - Replaces NCCL's default network transport with UCX
  - Enables: RDMA hardware offload, better pipelining, topology-aware routing
  - Essential when scaling beyond one NVLink domain (multi-rack, multi-node)

  ┌──── NVL72 Rack 0  ────┐         ┌──── NVL72 Rack 1  ────┐
  │ GPU↔GPU via NVLink    │         │ GPU↔GPU via NVLink    │
  │ (P2P DMA, fast)       │         │ (P2P DMA, fast)       │
  └──────────┬────────────┘         └────────── ┬───────────┘
             │                                  │
             └──── InfiniBand + NCCL-UCX ───────┘
                  (RDMA, hardware offload)

  Config:
    NCCL_NET=UCX
    NCCL_PLUGIN_P2P=ucx
    UCX_TLS=rc,self,gdr_copy,cuda_copy

- Summary:
  Within NVLink domain → P2P DMA (direct, fastest)
  Across nodes/racks   → NCCL + UCX plugin (RDMA, hardware offload, topology-aware)

3/3/26:

- Symmetric memory = shared address space across GPUs
- Each GPU can directly read/write other GPUs' buffers (one-sided put/get)
- No CPU handshake, no explicit copy calls, no NCCL setup per op
- Use case: small, frequent, latency-sensitive cross-GPU ops (MoE token shuffles)
- CUDA Graph benefit: no CPU involvement → no D2H gaps → entire MoE forward pass
(including all-to-all) can be captured and replayed in a single graph
- Not a replacement for NCCL — NCCL is still better for large collectives (gradient allreduce)
- Works with Triton + NVSHMEM for custom in-kernel transfers

WHY PINNING EXISTS (non-coherent / PCIe):

  The DMA engine needs a STABLE physical address to copy from.
  
  Regular (pageable) memory:
    OS can swap pages to disk or move them anytime
    → DMA engine starts reading from address 0x1000
    → OS moves the page mid-transfer
    → DMA reads garbage from 0x1000 (page is gone!)
    
    Solution: CPU must first copy to a pinned (locked) staging buffer
    
    [pageable RAM] → CPU copies → [pinned buffer] → DMA copies → [GPU HBM]
                     ↑ extra step                    ↑ safe, address won't change
    
  Pinned memory:
    OS CAN'T move or swap these pages
    → DMA engine reads from stable address → works correctly
    
    [pinned RAM] → DMA copies directly → [GPU HBM]
                   ↑ no extra CPU copy step

WHY COHERENT SYSTEMS DON'T NEED THIS AS MUCH (NVLink-C2C):

  GPU doesn't use DMA to copy from CPU RAM.
  GPU directly ACCESSES CPU memory through the coherence protocol.
  
  [CPU RAM] ←── GPU reads/writes directly via NVLink-C2C ──→ [GPU]
                ↑
          coherence hardware tracks where pages are
          if OS moves a page → coherence protocol updates the mapping
          GPU always follows the correct address → no stale reads
          
  No DMA copy happening → no need to lock pages in place for DMA
  The GPU is just doing memory loads/stores through a coherent link.

- Data input pipeline optimization — preventing GPU idle time

  Problem: GPU finishes batch → waits for next batch → idle time
  
  GPU: [compute]──[idle waiting for data]──[compute]──[idle]
  CPU: [loading]────────────────────────[loading]──────────

- Key DataLoader settings
  ┌─────────────────────────┬────────────────────────────────────────────┐
  │ pin_memory=True          │ Lock host pages → fast DMA to GPU         │
  │                          │ Less critical on NVLink-C2C superchips    │
  │                          │ (coherent path already fast)              │
  ├─────────────────────────┼────────────────────────────────────────────┤
  │ num_workers=N            │ Parallel worker processes for loading     │
  │                          │ Heuristic: 4 per GPU, sweep 4/8/16/32    │
  │                          │ Stop when CPU cores saturate (all 100%)   │
  ├─────────────────────────┼────────────────────────────────────────────┤
  │ persistent_workers=True  │ Don't respawn workers between epochs      │
  │                          │ Avoids process startup overhead           │
  │                          │ Important for tokenization-heavy LLM work │
  ├─────────────────────────┼────────────────────────────────────────────┤
  │ prefetch_factor=N        │ Each worker preloads N batches ahead      │
  │                          │ Total prefetch = prefetch_factor × workers │
  │                          │ Keeps pipeline full, fewer GPU stalls     │
  │                          │ Don't overfetch → wastes CPU memory       │
  ├─────────────────────────┼────────────────────────────────────────────┤
  │ non_blocking=True        │ Async H2D transfer, CPU doesn't wait     │
  └─────────────────────────┴────────────────────────────────────────────┘

- Superchip (NVLink-C2C) note
  - Coherent memory → GPU accesses CPU RAM directly, no DMA pinning needed
  - But still use large pages + verify overlap with Nsight Systems before removing pinning

- Precompute datasets
  - Tokenize once, save to disk → skip tokenization in training loop
  - Tools: Hugging Face Dataset.cache(), WebDataset
  - Also: mixed precision / compression for datasets → reduce I/O bandwidth

- TorchData with DataPipes
  - PyTorch's native data loading library
  - "Composable" = you chain small transform steps like Unix pipes:
    read_files → decode → tokenize → batch → shuffle
    Each step is a DataPipe, they snap together
  - Integrates with PyTorch scheduler to overlap data loading with training
  - Basically a cleaner, more modular replacement for old-style Dataset + DataLoader

- NVIDIA DALI (Data Loading Library)
  - Offloads data preprocessing to GPU instead of CPU
  - CPU-based loaders: [CPU: decode JPEG → resize → augment] → copy to GPU → train
  - DALI:              [GPU: decode JPEG → resize → augment → train] ← all on GPU
  - Especially useful for heavy preprocessing (image/video decode, augmentations)
  - Feeds data through a CUDA pipeline directly → no CPU bottleneck for transforms
  - When to use: image/video workloads where CPU can't keep up with decode/augment  

- Goal: GPU never waits for data → 100% SM utilization
  - Nsight Systems: CPU thread loading next batch IN PARALLEL with GPU training
  - No gaps in GPU timeline = pipeline is working

- Larger batch size benefits (once memory freed by checkpointing/offloading)
  - Higher arithmetic intensity (more compute per memory access)
  - Better GPU utilization (more work per kernel launch)
  - Less communication overhead in multi-GPU (fewer steps per epoch = fewer allreduces)

- Larger batch size risks
  - Too large → optimizer converges to sharp minima → worse generalization
  - Monitor with TorchEval: check if validation loss degrades
  - Gradient accumulation also increases effective batch size: batch_size × accumulation_steps

- Mitigations for large-batch instability
  - Linear learning rate scaling + warm-up period
  - Large-batch optimizers (e.g., LAMB)
  - Retune hyperparameters after changing batch size

- Practical note
  - Periodically retune hyperparameters (LR, batch size, etc.) as you optimize the system
  - What was optimal before may not be optimal after memory/compute changes

- DDP + torch.compile — complete picture

  DDP basics:
    - Each GPU has full model copy, processes different data
    - After backward: allreduce to average gradients across GPUs
    - DDP groups gradients into BUCKETS (default 25 MB each)
    - Buckets allreduce as soon as their grads are ready → overlaps with remaining backward

  Why torch.compile creates graph breaks at bucket boundaries:
  
  torch.compile can't put allreduce INSIDE a compiled graph because:
    1. Compiler generates single-GPU Triton kernels — doesn't model NCCL
    2. Allreduce must run on a SEPARATE comms stream for overlap
    3. Fusing it into one stream would serialize compute and comms

  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Model backward (4 layers, 2 buckets):                                  │
  │   Bucket 2 = [Layer 4, Layer 3 grads]                                  │
  │   Bucket 1 = [Layer 2, Layer 1 grads]                                  │
  │                                                                         │
  │ torch.compile produces:                                                 │
  │   Compiled subgraph 2: fused kernel for Layer 4 + Layer 3 grads        │
  │   ──── graph break (DDP inserts allreduce for bucket 2) ────           │
  │   Compiled subgraph 1: fused kernel for Layer 2 + Layer 1 grads        │
  │   ──── graph break (DDP inserts allreduce for bucket 1) ────           │
  └─────────────────────────────────────────────────────────────────────────┘

  Timeline (what actually runs on GPU):
  
  Compute stream: [subgraph 2: grad L4+L3]────[subgraph 1: grad L2+L1]────
  Comms stream:                     [allreduce bucket 2]────[allreduce bucket 1]
                                     ↑ overlap!              ↑ overlap with
                                     runs while subgraph 1    next fwd pass
                                     computes on SMs

  torch._dynamo.explain() output will show:
    "Graph break: torch.distributed.all_reduce"  ← this is expected, not a bug

- Bucket size tradeoff

  Many small buckets (e.g., 10 MB):
  ┌────────────────────────────────────────────────────────────────┐
  │ Compute: [sg5][sg4][sg3][sg2][sg1]  ← many small subgraphs   │
  │ Comms:     [ar5][ar4][ar3][ar2][ar1] ← lots of overlap points │
  │ + More overlap opportunities                                   │
  │ - Less fusion within each subgraph (too small to optimize)    │
  │ - More kernel launch overhead                                  │
  └────────────────────────────────────────────────────────────────┘

  One giant bucket (all grads):
  ┌────────────────────────────────────────────────────────────────┐
  │ Compute: [════════ one big compiled graph ════════]            │
  │ Comms:                                             [allreduce] │
  │                                                     ↑ NO overlap
  │ + Maximum kernel fusion within the graph            (sequential)
  │ - Zero overlap — allreduce waits for ALL grads                │
  └────────────────────────────────────────────────────────────────┘

  Default 25 MB buckets (sweet spot):
  ┌────────────────────────────────────────────────────────────────┐
  │ Compute: [subgraph 2 (fused)]────[subgraph 1 (fused)]        │
  │ Comms:              [allreduce b2]────[allreduce b1]          │
  │ + Good fusion within each subgraph                             │
  │ + Good overlap between compute and comms                       │
  │ → Profile to find optimal bucket size for your hardware        │
  └────────────────────────────────────────────────────────────────┘

═══ WITH BUCKETS (2 subgraphs) ═══

  time(ms):  0         5         10        15        20        25
             |─────────|─────────|─────────|─────────|─────────|

  Compute    [███ grad L4+L3 ███]
  stream:    (subgraph 2)        [███ grad L2+L1 ███]
                                  (subgraph 1)

  Comms                          [═══ allreduce ═══]
  stream:                         bucket 2          [═══ allreduce ═══]
                                                     bucket 1

             |─────────|─────────|─────────|─────────|─────────|
                                  ↑                   ↑
                          subgraph 1 compute     allreduce b1
                          OVERLAPS with          OVERLAPS with
                          allreduce b2           next forward pass

  Wall time: ~20ms  ✓


═══ WITHOUT BUCKETS (1 giant graph) ═══

  time(ms):  0         5         10        15        20        25        30
             |─────────|─────────|─────────|─────────|─────────|─────────|

  Compute    [███████████ grad L4+L3+L2+L1 ███████████]
  stream:    (one big fused graph)                      IDLE zzz...

  Comms                                                 [════ allreduce ════]
  stream:                                                all grads at once

             |─────────|─────────|─────────|─────────|─────────|─────────|
                                                        ↑
                                                  GPU SMs doing NOTHING
                                                  while allreduce runs
                                                  = wasted 10ms

  Wall time: ~30ms  ✗  (10ms slower)


═══ THE DIFFERENCE ═══

  Buckets:     [compute][compute]       = 20ms total
                     [comms][comms]        (comms hidden behind compute)

  No buckets:  [════ compute ════][comms] = 30ms total
                                   ↑ exposed, not hidden

┌──────────┬────────────────┬──────────────┐
│          │ Model weights   │ Data          │
├──────────┼────────────────┼──────────────┤
│ DDP      │ Full copy       │ Own batch     │
│ FSDP     │ 1/N shard       │ Own batch     │
└──────────┴────────────────┴──────────────┘
  Both process DIFFERENT data on EVERY GPU.

DDP:
  ┌─── GPU 0 ───┐     ┌─── GPU 1 ───┐
  │ FULL model   │     │ FULL model   │
  │ FULL optim   │     │ FULL optim   │
  │ Batch A      │     │ Batch B      │
  └──────┬───────┘     └──────┬───────┘
         │                    │
  Forward:                    │
         │  no comms          │  ← both GPUs already have all weights
         │  (independent)     │
         │                    │
  Backward:                   │
         │  allreduce grads   │  ← average grads across GPUs (once)
         └────────────────────┘
  
  Optimizer: each GPU updates its full model copy identically


FSDP:
  ┌─── GPU 0 ───┐     ┌─── GPU 1 ───┐
  │ 1/2 model    │     │ 1/2 model    │
  │ 1/2 optim    │     │ 1/2 optim    │
  │ Batch A      │     │ Batch B      │
  └──────┬───────┘     └──────┬───────┘
         │                    │
  Forward (per layer):        │
         │  allgather weights │  ← borrow missing shards
         │  compute layer     │
         │  discard borrowed  │  ← free memory immediately
         │                    │
  Backward (per layer):       │
         │  allgather weights │  ← borrow shards again (discarded in fwd)
         │  compute grads     │
         │  reduce-scatter    │  ← send each grad piece to its owner
         │  discard borrowed  │
         └────────────────────┘

  Optimizer: each GPU updates ONLY its owned shard


One-liner:
  DDP  = store everything, zero fwd comms, allreduce grads once
  FSDP = store a slice, gather before every layer, scatter after every layer

- FSDP + torch.compile — per-block wrapping is key

  Same principle as DDP buckets: break into subgraphs so comms overlap with compute

  ┌─────────────────────┬──────────────────────────────────────────────┐
  │ Wrap full model in   │ Wrap each transformer block in FSDP          │
  │ one FSDP instance    │ (recommended)                                │
  │ (BAD)                │ (GOOD)                                       │
  ├─────────────────────┼──────────────────────────────────────────────┤
  │ One big compiled     │ One compiled graph per block                 │
  │ graph                │                                              │
  │ Comms at the end     │ Comms between graphs (allgather/reduce-      │
  │ only → no overlap    │ scatter) → overlaps with next block's compute│
  │ All weights in       │ Only one block's weights in memory at a time │
  │ memory at once       │                                              │
  └─────────────────────┴──────────────────────────────────────────────┘

  Per-block timeline:
  Compute: [compiled B0][compiled B1][compiled B2][compiled B3]
  Comms:        [ag B1]       [ag B2]      [ag B3]
                 ↑ overlap     ↑ overlap    ↑ overlap

- How to set it up
  - transformer_auto_wrap_policy → auto-wraps each TransformerBlock
  - use_orig_params=True → no flattening, better overlap and optimizer compat
  - TorchDynamo auto-inserts graph breaks at FSDP shard boundaries
  - Mirrors DDP's bucketed approach but with allgather/reduce-scatter instead of allreduce  
═══ Setup: 4 transformer blocks, 2 GPUs ═══

  Full model:
  [Embedding] → [Block 0] → [Block 1] → [Block 2] → [Block 3] → [Output Head]

  Each block = 10 GB weights (attention + FFN + norms)


═══ How FSDP shards it ═══

  Each block's weights split in half across 2 GPUs:

  GPU 0 stores:                        GPU 1 stores:
  ┌──────────────────────┐             ┌──────────────────────┐
  │ Embed shard (0.5 GB) │             │ Embed shard (0.5 GB) │
  │ Block 0 shard (5 GB) │             │ Block 0 shard (5 GB) │
  │ Block 1 shard (5 GB) │             │ Block 1 shard (5 GB) │
  │ Block 2 shard (5 GB) │             │ Block 2 shard (5 GB) │
  │ Block 3 shard (5 GB) │             │ Block 3 shard (5 GB) │
  │ Head shard (0.5 GB)  │             │ Head shard (0.5 GB)  │
  ├──────────────────────┤             ├──────────────────────┤
  │ Total: 21 GB stored  │             │ Total: 21 GB stored  │
  └──────────────────────┘             └──────────────────────┘
  vs. 42 GB if full model on each GPU (DDP)


═══ FORWARD PASS — one block at a time ═══

  Step 1: Block 0
  ┌──────────────────────────────────────────────────────────────┐
  │ GPU 0 has 5 GB shard, GPU 1 has 5 GB shard                  │
  │                                                              │
  │    allgather: GPU 0 ←→ GPU 1 exchange shards                │
  │    Both GPUs now have FULL 10 GB Block 0 weights             │
  │                                                              │
  │ GPU 0: fwd(Block 0, batch A) → activations A0               │
  │ GPU 1: fwd(Block 0, batch B) → activations B0               │
  │                                                              │
  │    Drop borrowed 5 GB shard → back to 5 GB each             │
  └──────────────────────────────────────────────────────────────┘

  Step 2: Block 1 (OVERLAPPED with Block 0 drop)
  ┌──────────────────────────────────────────────────────────────┐
  │    allgather Block 1 shards (can start during Block 0 compute)
  │    Both GPUs now have FULL 10 GB Block 1 weights             │
  │                                                              │
  │ GPU 0: fwd(Block 1, activations A0) → activations A1        │
  │ GPU 1: fwd(Block 1, activations B0) → activations B1        │
  │                                                              │
  │    Drop borrowed shard                                       │
  └──────────────────────────────────────────────────────────────┘

  ...Block 2, Block 3 same pattern...

  Forward timeline (GPU 0):
  time:   0      5     10     15     20     25     30     35
          |──────|──────|──────|──────|──────|──────|──────|
  Comms: [ag B0]     [ag B1]      [ag B2]      [ag B3]
          gather      gather       gather       gather
  Comp:       [fwd B0]     [fwd B1]      [fwd B2]      [fwd B3]
               ↑             ↑              ↑             ↑
          uses full      uses full     uses full     uses full
          10 GB weights  10 GB         10 GB         10 GB

  Memory:  15 GB   10 GB   15 GB   10 GB   15 GB   10 GB
           peak    drop    peak    drop    peak    drop
           (5 own + 10 full block)

  Peak forward memory: ~15 GB (own shards + one full block)
  vs. 42 GB if everything loaded at once


═══ BACKWARD PASS — reverse order, allgather + reduce-scatter ═══

  Step 1: Block 3 (last block first in backward)
  ┌──────────────────────────────────────────────────────────────┐
  │    allgather Block 3 weights (need them again for grads)     │
  │                                                              │
  │ GPU 0: bwd(Block 3, saved activations) → grad_W3            │
  │ GPU 1: bwd(Block 3, saved activations) → grad_W3            │
  │                                                              │
  │    reduce-scatter grad_W3:                                   │
  │      GPU 0 gets averaged grad for its shard of W3            │
  │      GPU 1 gets averaged grad for its shard of W3            │
  │    Drop borrowed weights                                     │
  └──────────────────────────────────────────────────────────────┘

  Step 2: Block 2 (OVERLAPPED with Block 3 reduce-scatter)
  ┌──────────────────────────────────────────────────────────────┐
  │    allgather Block 2 weights                                 │
  │    bwd(Block 2) → grad_W2                                   │
  │    reduce-scatter grad_W2                                    │
  │    Drop borrowed weights                                     │
  └──────────────────────────────────────────────────────────────┘

  ...Block 1, Block 0 same pattern...

  Backward timeline (GPU 0):
  time:   0      5     10     15     20     25     30     35     40
          |──────|──────|──────|──────|──────|──────|──────|──────|
  Comms: [ag B3]     [rs B3 + ag B2]    [rs B2 + ag B1]    [rs B1 + ag B0]
  Comp:       [bwd B3]          [bwd B2]          [bwd B1]          [bwd B0]

  ag = allgather weights (borrow)
  rs = reduce-scatter grads (send each piece to owner)
  Both overlap with next block's compute!


═══ After backward — optimizer step ═══

  GPU 0: has averaged grads for its shards only → updates its shards
  GPU 1: has averaged grads for its shards only → updates its shards
  No comms needed for optimizer step!


═══ Benefits summary ═══

  ┌────────────┬──────────────────────────────────────────────────┐
  │ Memory     │ Only 1 block fully materialized at a time        │
  │            │ Peak = own shards + 1 full block (not all blocks)│
  ├────────────┼──────────────────────────────────────────────────┤
  │ Forward    │ allgather per block, overlapped with prev compute│
  │            │ Drop borrowed weights immediately after use      │
  ├────────────┼──────────────────────────────────────────────────┤
  │ Backward   │ allgather + reduce-scatter per block             │
  │            │ Both overlapped with next block's compute        │
  ├────────────┼──────────────────────────────────────────────────┤
  │ Optimizer  │ Each GPU updates only its owned shards           │
  │            │ No comms needed                                  │
  └────────────┴──────────────────────────────────────────────────┘

- FSDP + torch.compile wrap-up

  - Prefetching: while block N computes, block N+1's weights allgathered async → hides comms
  - Training loop unchanged — auto_wrap_policy handles per-block FSDP nesting automatically
  - torch._dynamo.explain() will show graph breaks at each FSDP boundary → expected, not a bug

  Per-block benefits:
  ┌────────────┬──────────────────────────────────────────────┐
  │ Memory     │ One block's full weights at a time            │
  │ Compute    │ Each block = one compiled, fused graph        │
  │ Comms      │ allgather/reduce-scatter overlapped with      │
  │            │ next block's compute via prefetch              │
  │ Code       │ Specify block class once, FSDP handles rest   │
  └────────────┴──────────────────────────────────────────────┘    

3/4/26:

- torch.compile + TP/PP — separation of concerns

  ┌──────────────────────────────────────────────────────────────┐
  │            What torch.compile optimizes                      │
  │                                                              │
  │  Python code → TorchDynamo → AOT Autograd → TorchInductor   │
  │                                                    ↓         │
  │                                    Fused Triton/cuBLAS/cuDNN │
  │                                    kernels for MATH ops      │
  │                                                              │
  │  Scope: compute WITHIN each GPU's segment                    │
  │  (matmul, activation, layernorm, dropout, etc.)              │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │            What distributed strategy handles                 │
  │                                                              │
  │  DDP:  allreduce grads (bucketed)                            │
  │  FSDP: allgather weights + reduce-scatter grads (per block)  │
  │  TP:   allreduce/allgather within layer (per matmul)         │
  │  PP:   send/recv activations between stages                  │
  │                                                              │
  │  Scope: NCCL collectives BETWEEN GPUs                        │
  │  torch.compile does NOT touch these                          │
  └──────────────────────────────────────────────────────────────┘

  How they coexist:

  ┌─────────────────────────────────┐
  │ Compiled graph (fused compute)  │──── graph break
  └─────────────────────────────────┘         ↓
                                      [NCCL collective]  ← handled by DDP/FSDP/TP/PP
                                              ↓
  ┌─────────────────────────────────┐
  │ Compiled graph (fused compute)  │──── graph break
  └─────────────────────────────────┘         ↓
                                      [NCCL collective]
                                              ↓
                                          ...repeat

  Compiler fuses math inside each segment
  Distributed strategy handles comms between segments
  Neither touches the other's job

- Practical checklist
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. torch.compile optimizes compute, not comms               │
  │ 2. NCCL collectives left as-is (ordering, schedule, stream) │
  │ 3. Graph breaks at collective boundaries → expected          │
  │ 4. Always test with AND without compile → compare results   │
  │ 5. Profile with Nsight Systems → verify overlap preserved   │
  │ 6. Check torch._dynamo.explain() for unexpected breaks      │
  │    (allreduce breaks = normal, other breaks = investigate)   │
  └─────────────────────────────────────────────────────────────┘

- TorchTitan, AsyncTP, AutoParallel, SimpleFSDP

  ┌──────────────┬─────────────────────────────────────────────────────────────┐
  │ TorchTitan    │ Reference implementations for large-scale training         │
  │               │ Composes FSDP + TP + PP + AsyncTP recipes                  │
  │               │ Uses DTensor as the core sharding primitive                │
  ├──────────────┼─────────────────────────────────────────────────────────────┤
  │ AsyncTP       │ Overlaps TP collectives with compute using dual streams    │
  │               │ SM-wave aware schedule: stagger allgather so it overlaps   │
  │               │ with next matmul wave                                      │
  │               │                                                            │
  │               │ Traditional TP:                                            │
  │               │   [matmul]──[allgather]──[matmul]──[allgather]             │
  │               │              ↑ idle SMs                                    │
  │               │                                                            │
  │               │ AsyncTP:                                                   │
  │               │   [matmul wave 1]──[matmul wave 2]──                       │
  │               │        [allgather]──────                                   │
  │               │         ↑ overlapped with next wave's compute              │
  │               │                                                            │
  │               │ Enabled via torch.compile — fuses matmuls with             │
  │               │ allgather/reduce-scatter                                   │
  ├──────────────┼─────────────────────────────────────────────────────────────┤
  │ AutoParallel  │ Automatically plans FSDP + TP + PP combinations            │
  │               │ Heuristic-based: considers memory and comms costs          │
  │               │ per workload                                               │
  │               │ Built on DTensor (PyTorch's native sharding primitive)     │
  │               │ Reduces manual parallelization tuning                      │
  ├──────────────┼─────────────────────────────────────────────────────────────┤
  │ SimpleFSDP    │ FSDP reimplemented to be torch.compile-friendly            │
  │               │ Uses DTensor + selective activation checkpointing           │
  │               │ TorchInductor buckets and reorders IR nodes for            │
  │               │ better compute-comms overlap                               │
  │               │ Results: ~28% less memory, up to ~69% faster than          │
  │               │ traditional FSDP2 eager path                               │
  └──────────────┴─────────────────────────────────────────────────────────────┘

  How they relate:
  ┌──────────────────────────────────────────────────┐
  │ TorchTitan (reference recipes)                    │
  │  ├── AsyncTP (overlapped TP collectives)          │
  │  ├── AutoParallel (auto plan FSDP+TP+PP)          │
  │  ├── SimpleFSDP (compile-friendly FSDP)           │
  │  └── All built on DTensor (sharding primitive)    │
  └──────────────────────────────────────────────────┘

═══ Normal TP (one matmul at a time) ═══

  TP allreduce/allgather happens AFTER matmul finishes:

  SMs:        [███ matmul ███]  [idle]  [███ matmul ███]  [idle]
  NVLink:                       [ag]                      [ag]
                                 ↑ SMs waiting              ↑ SMs waiting
  
  Problem: SMs sit idle during allgather


═══ What is a "wave"? ═══

  A matmul launches thousands of thread blocks.
  GPU can only run ~160 blocks at once (160 SMs on Blackwell).
  If matmul needs 320 blocks → runs in 2 WAVES:

  Wave 1: blocks 0-159   (fills all SMs)
  Wave 2: blocks 160-319 (fills all SMs again)

  SMs:  [wave 1: all SMs busy][wave 2: all SMs busy] → matmul done


═══ AsyncTP trick: start allgather during LAST wave ═══

  Insight: during the last wave of a matmul, some SMs finish early
  and sit idle. Use that idle time to start the allgather!

  Normal TP:
  SMs:     [wave1][wave2][idle][wave1][wave2][idle]
  NVLink:                [ag]                [ag]
                          ↑ wasted time

  AsyncTP (dual streams):
  Stream 1 (compute): [wave1][wave2]     [wave1][wave2]
  Stream 2 (comms):          [allgather]──┘     [allgather]──
                              ↑                   ↑
                        starts during         starts during
                        wave 2 of prev        wave 2 of prev
                        matmul (SMs not       matmul
                        all busy at tail)

  "Dual streams" = one stream for matmul, one stream for allgather
  "SM-wave aware" = knows when the last wave has leftover SMs → starts comms there
  "Stagger" = allgather starts before matmul fully finishes

- HTA (Holistic Trace Analysis) — Meta's multi-GPU profiling tool

  Problem: N GPUs = N separate traces → impossible to manually compare

  ┌──────────────────────────────────────────────────────────────────┐
  │ torch.profiler (each rank) → JSON traces → HTA merges + aligns  │
  │                                                                  │
  │  Rank 0: [fwd]────[bwd]────[idle]────[allreduce]────            │
  │  Rank 1: [fwd]────[bwd]──────────[bwd]──[allreduce]──           │
  │  Rank 2: [fwd]────[bwd]────[idle]────[allreduce]────            │
  │                              ↑              ↑                    │
  │                     HTA spots this:  straggler + sync issue      │
  └──────────────────────────────────────────────────────────────────┘

- What HTA shows
  - Unified timeline: all GPUs aligned by time with NVTX markers
  - GPU idle times per rank → find stragglers
  - Sync gaps: which rank is waiting for which (e.g., allreduce bottleneck)
  - Load imbalance: rank 0 finishes early, sits idle while others compute
  - Suggestions for improving overlap

- Verifying optimizations with HTA
  Before (no overlap):
    [backward]──[gap]──[allreduce]──[next compute]

  After (bucketed overlap):
    [backward]──────────────────
            [allreduce]──────    ← gap shrinks/disappears

- Workflow
  1. torch.profiler.schedule → record traces for a few iterations per rank
  2. Save JSON traces to shared location
  3. Load into HTA → unified timeline + analysis

- Note: TensorBoard profiler plugin deprecated (as of 2025)
  Use Perfetto for single-trace timelines, HTA for multi-GPU aggregation

- CI performance regression testing

  Optimize once is not enough → must protect gains as code/PyTorch versions evolve

  Workflow:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Code commit → CI triggers → TorchBench runs model → JSON output │
  │                                                        ↓        │
  │                              compare_perf.py(baseline.json, results.json)
  │                                                        ↓        │
  │                              ≥5% slower? → FAIL build             │
  │                              OK? → pass                           │
  └──────────────────────────────────────────────────────────────────┘

  What to track in CI:
  ┌────────────────────────────┬────────────────────────────────────┐
  │ Throughput                  │ tokens/sec or samples/sec          │
  │ Memory                     │ torch.cuda.max_memory_allocated()  │
  │ Data loading time          │ time the actual data pipeline      │
  │ Correctness                │ torch.allclose() with strict tols  │
  └────────────────────────────┴────────────────────────────────────┘

  Avoid false alarms:
  - Run multiple iterations before failing
  - Require regression sustained over 3+ runs
  - Use statistical smoothing (PyTorch's own approach)
  - Use consistent hardware / reserved cloud instances

- TorchBench
  - Open source suite of PyTorch model benchmarks
  - Includes torch.compile benchmarks for compiler perf tracking
  - Can add your own model: fork TorchBench, add model, run in CI

- Correctness testing
  - Custom kernels must match PyTorch's reference ops
  - Test edge cases (large values → overflow, random seeds)
  - torch.allclose() for numerical accuracy

- PyTorch HUD (Performance Dashboard)
  - Public web UI tracking nightly benchmark results across models + hardware
  - Shows: throughput, compilation time, memory, FLOPS utilization over time
  - ~5% regression threshold → flagged in red → investigation
  - Trend lines catch gradual decay from many small commits
  - Tracks vLLM, common LLMs across NVIDIA/AMD GPUs and CPUs
  - Open source (pytorch/test-infra) → can mimic for your own models
  - Correlates perf changes directly with GitHub commits

- Key takeaway
  ┌─────────────────────────────────────────────────────────────────┐
  │ Performance is multidimensional:                                │
  │   20% faster compute + 200% more memory = net negative          │
  │                                                                 │
  │ Track all dimensions in CI, not just speed                      │
  │                                                                 │
  │ Even PyTorch updates can regress your workload —                │
  │ catch early, report upstream, PyTorch team is responsive        │
  └─────────────────────────────────────────────────────────────────┘

- MLPerf logging — structured performance tracking

  Format: JSON log entries with :::MLL prefix, timestamped to millisecond
  Library: open source MLPerf Logging on GitHub

  Example breakdown per training step:
  ┌─────────────────────────┬──────────┬────────┐
  │ Component                │ Time     │ %      │
  ├─────────────────────────┼──────────┼────────┤
  │ Forward pass             │ 10.5 ms  │ 43.8%  │
  │ Backward pass            │ 9.0 ms   │ 37.5%  │
  │ All-reduce (grad sync)   │ 4.0 ms   │ 16.7%  │
  │ Other overhead           │ 0.5 ms   │ 2.1%   │
  ├─────────────────────────┼──────────┼────────┤
  │ Total step               │ 24.0 ms  │ 100%   │
  └─────────────────────────┴──────────┴────────┘
  → tells you exactly where to optimize (e.g., 16.7% in allreduce → try async overlap)

- Why structured logging matters
  - Pairs performance with accuracy (can't claim speed without meeting accuracy target)
  - Makes results reproducible and debuggable
  - Track trends over long runs (did memory fragmentation slow things down on day 7?)
  - Communicate bottlenecks to team in standardized format

- MLPerf compliance
  - Scripts verify: no hyperparameter changes after start, correct epochs, target accuracy met
  - Ensures fair apples-to-apples comparison across submissions

- Practical takeaway (even without competing)
  ┌────────────────────────────────────────────────────────────────┐
  │ Log JSON per epoch/step:                                       │
  │   throughput, latency, fwd/bwd/comms breakdown,                │
  │   GPU utilization (nvidia-smi), memory usage                   │
  │                                                                │
  │ Plot over time → spot regressions, fragmentation, bottlenecks  │
  │                                                                │
  │ Study MLPerf submissions → best practices for LLMs + clusters  │
  └────────────────────────────────────────────────────────────────┘  

Problem: FP16 has a tiny range → gradients underflow to zero

  FP32 range: ±3.4 × 10^38     (huge)
  BF16 range: ±3.4 × 10^38     (same exponent as FP32!)
  FP16 range: ±6.5 × 10^4      (tiny — max value ~65,504)

  During backward pass, gradients can be very small:
    gradient = 0.00001 → fits in FP32 and BF16
    gradient = 0.0000001 → fits in FP32 and BF16
    gradient = 0.00000001 → FP16 rounds this to 0.0 (underflow!)
    ↑ lost gradient = model stops learning

Solution for FP16: GradScaler

  Before backward:
    loss = loss × 1024       ← scale UP the loss
                               (makes all gradients 1024× bigger)
  
  Backward pass:
    gradient = 0.00000001 × 1024 = 0.00001  → fits in FP16 now! ✓
  
  Before optimizer step:
    gradient = gradient / 1024   ← scale back DOWN
                                   (undo the scaling)
  
  Additional check:
    if any gradient = inf or NaN → skip this step, reduce scale factor
    ↑ overflow detection (scaled gradient too big for FP16)

  ┌──────────────────────────────────────────────────────┐
  │ FP16 training with GradScaler:                       │
  │                                                      │
  │ loss × scale → backward → grads/scale → optimizer    │
  │      ↑                        ↑                      │
  │  prevent underflow      restore true magnitude       │
  │                                                      │
  │ + check for inf/NaN each step                        │
  │ + dynamically adjust scale factor                    │
  └──────────────────────────────────────────────────────┘

Why BF16 doesn't need this:
  BF16 has SAME exponent bits as FP32 (8 bits) → same range
  gradient = 0.00000001 → BF16 represents this fine (no underflow)
  No scaling needed, no inf checks, no skipped steps

  ┌──────────┬──────────┬───────────┬────────────────────┐
  │ Format    │ Exponent │ Range     │ Grad scaling needed│
  ├──────────┼──────────┼───────────┼────────────────────┤
  │ FP32      │ 8 bits   │ ±3.4e38  │ No                 │
  │ BF16      │ 8 bits   │ ±3.4e38  │ No                 │
  │ FP16      │ 5 bits   │ ±6.5e4   │ YES                │
  └──────────┴──────────┴───────────┴────────────────────┘

  BF16 tradeoff: less precision (7 bits mantissa vs FP16's 10)
  but same range → no underflow → no scaling complexity

Key Takeaways (continued)

- Save compiled artifacts to reuse
  - `torch.compiler.save_cache_artifacts()` and `load_cache_artifacts()` persist compilation output
  - For multinode fleets, set `TORCHINDUCTOR_CACHE_DIR` to a shared mount path
  - All nodes read the same mega-cache — eliminates cold-start recompilation on new nodes

- Avoid synchronization gotchas
  - `tensor.item()` on a CUDA tensor forces GPU→CPU sync — avoid in hot loops
  - `tensor.cpu()` without `non_blocking=True` also blocks
  - Use `wait_stream()` + events for inter-stream coordination, not `.item()` or `.cpu()`
  - Never use `time.time()` to profile GPU code — it inserts an implicit sync
  - Use `torch.cuda.Event(enable_timing=True)` for accurate GPU timing without sync penalty

- Utilize the Tensor Cores
  - Wrap forward + loss in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
  - `autocast` picks compute dtype per-op — does NOT change stored weight dtype
  - Numerically sensitive ops (softmax, layernorm) stay in FP32 automatically

- Use TF32 over FP32 to activate Tensor Cores
  - `torch.set_float32_matmul_precision("high")` — enables TF32 for FP32 matmuls
  - Maps to `torch.backends.cuda.matmul.fp32_precision` under the hood
  - "highest" = true FP32, disables TF32 — useful for debug only
  - Verify with Nsight Compute: `sm__inst_executed_pipe_tensor` metric or SpeedOfLight view

- Verify Tensor Cores are actually being used
  - `torch.profiler` → find the kernel → Nsight Compute → SpeedOfLight section
  - Look for Tensor pipeline utilization > 0% in Compute Workload Analysis
  - If Tensor pipe shows 0% — wrong dtype, wrong shape, or op not eligible

- Fuse small operations
  - Spot many sub-1ms ops in a row with `torch.profiler` (linear → gelu → dropout)
  - `torch.compile` fuses automatically; for misses, write custom Triton/CUDA kernels
  - Each fusion saves a few percent — they compound across the full model

- Reduce memory fragmentation
  - Reuse buffers, stick to constant shapes, preallocate at max size
  - Tune `PYTORCH_ALLOC_CONF`: `max_split_size_mb`, `roundup_power2_divisions`
  - Fewer distinct allocation sizes = less fragmentation = fewer surprise OOMs

- Use activation checkpointing
  - `torch.utils.checkpoint` — recompute activations in backward instead of storing them
  - ~30% extra FLOPs but massive memory savings — mandatory for 10B+ param models
  - Can mix precision: keep critical activations in FP32, recompute others in BF16

- Offload memory to CPU or NVMe
  - Parameters, gradients, optimizer states, activations all compete for VRAM
  - Stream offloaded data back with async copies overlapped with compute
  - Monitor interconnect throughput — don't saturate the PCIe/NVLink link

- Reduce input pipeline stalls
  - Gaps before iterations in profiler = data loader bottleneck
  - `num_workers` = utilize all CPU cores, `pin_memory=True`, `prefetch_factor` > 1
  - Preprocess (tokenize) offline so loader does minimal work at runtime

- Profile and possibly offload CPU-side transforms
  - Python tokenization can be surprisingly slow — vectorize or use C++ libs
  - GPU-side preprocessing with NVIDIA DALI for heavy transforms (decode, augment)

- Optimize multi-GPU and multinode communication
  - DDP overlaps allreduce with backward by default
  - Tune `bucket_cap_mb` — larger buckets (50 MB vs default 25 MB) reduce per-message overhead
  - Place frequently communicating ranks on the same node/switch

- Monitor network bandwidth
  - If saturated, explore activation compression or gradient compression
  - Overlap allreduce with backward compute to hide communication time

- Avoid bit rot over time
  - Minor refactor can introduce hidden syncs; PyTorch upgrade can change op implementation
  - CI performance benchmarks on every commit (or daily/weekly)
  - Dashboards + alerts — treat perf as a first-class metric alongside accuracy

- Update baselines on hardware changes
  - B200 → GB300 = different strengths/weaknesses — recalibrate thresholds
  - Old baseline on new hardware gives false positives/negatives

- Use TorchBench + PyTorch HUD for regression tracking
  - Watch PyTorch nightly perf benchmarks — catch upstream regressions before upgrading
  - On slowdown: investigate → root-cause → revert/adapt → report upstream
  - Maintain correctness tests alongside perf tests — don't silently break accuracy

Sync gotcha cheat sheet

  Code pattern              Blocks GPU?   Blocks CPU?   Fix
  ─────────────────────────────────────────────────────────────
  tensor.item()             yes            yes          avoid in hot path
  tensor.cpu()              yes            yes          add non_blocking=True
  time.time() around GPU    yes            yes          use cuda Events
  wait_stream()             yes            no           correct pattern ✓
  wait_event()              yes            no           correct pattern ✓
  torch.cuda.synchronize()  yes            yes          only for debug

Tensor Core activation decision tree

  FP32 workload?
  ├─ yes → torch.set_float32_matmul_precision("high") → TF32 on Tensor Cores
  └─ no → mixed precision?
       ├─ yes → autocast(dtype=bfloat16) → BF16 on Tensor Cores
       └─ no → already BF16/FP16 → Tensor Cores used if shapes are aligned
                                     (M, N, K divisible by 8 for BF16)

  Verify: Nsight Compute → SpeedOfLight → Tensor pipe util > 0%

Performance maintenance lifecycle

  Code commit
      │
      ▼
  CI perf benchmark (TorchBench / custom scripts)
      │
      ▼
  Compare against baseline ──── within threshold? ──── yes → merge ✓
      │                                                       
      no                                                      
      │                                                       
      ▼                                                       
  Investigate root cause                                      
      │                                                       
      ├── your code? → fix before merge                       
      └── upstream? → report to PyTorch/Triton/NVIDIA         
                      pin old version until fix lands          

  Hardware change (B200 → GB300)?
      └── re-establish baseline + recalibrate alert thresholds

Cache artifacts, kill syncs, fuse small ops, checkpoint big layers, benchmark every commit — performance is a feature, not a phase.

Chapter 13 is done!