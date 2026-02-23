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