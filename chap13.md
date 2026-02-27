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
  
    
