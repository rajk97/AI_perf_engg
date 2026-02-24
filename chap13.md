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

