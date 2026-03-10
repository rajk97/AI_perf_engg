PyTorch Compiler, OpenAI Triton, and XLA Backends

TorchDynamo — bytecode capture and graph extraction

  What it does:
  - Stage 1 of torch.compile — hooks into CPython frame evaluation (PEP 523)
  - Intercepts Python bytecode, collects tensor ops into an FX Graph instead of executing them one-by-one
  - Unsupported code → graph break → that part runs eager, Dynamo resumes after

  What it outputs:
  - An FX Graph — IR where each node is an ATen primitive (aten::sin, aten::add, etc.)

  What happens next:
  - FX Graph → AOT Autograd (traces joint forward + backward graph)
  - Joint graph → TorchInductor (or XLA) → fused Triton/CUDA kernels
  - Dynamo only captures — all optimization is downstream

The pipeline (this is the core flow)

  Python code
      │
      ▼
  TorchDynamo (PEP 523 hook)
      │ captures tensor ops as nodes
      ▼
  FX Graph (ATen IR)              ← Dynamo's output
      │
      ▼
  AOT Autograd                    ← traces forward + backward together
      │
      ▼
  Joint fwd+bwd FX Graph
      │
      ▼
  TorchInductor (or XLA)          ← kernel fusion + device codegen
      │
      ▼
  Fused Triton/CUDA kernels       ← what actually runs on GPU

FX Graph example

  Python:   z = x.sin() + y; return z.sum()

  FX IR:    Placeholder(x) → sin(x) → add(sin,y) → sum(add) → return
            each node = one ATen op, no Python overhead between them

Graph breaks — what interrupts capture

  Dynamo captures op by op into FX Graph until it hits:
  - Complex control flow (data-dependent if/else)
  - Non-PyTorch library calls (print, numpy, custom C code)
  - Unsupported Python constructs

  Result: graph splits into multiple smaller FX Graphs with eager gaps between them
  Fix: refactor code, or mark safe ops with torch._dynamo.allow_in_graph()

Guards and recompilation

  On first compile, Dynamo records guards: shape=[32,512], dtype=float32, etc.
  On next call:
    guards pass   → reuse cached compiled graph (fast)
    guard fails   → recompile (full pipeline again — expensive)

  Dynamic shapes:
    dynamic=None (default) → specializes first, 1 extra recompile on first varying input
    mark_dynamic(tensor, dim) → tell Dynamo upfront → 0 extra recompiles

Stances — Dynamo's strictness control

  Stance               Behavior                        Use for
  ───────────────────  ──────────────────────────────   ──────────────────
  default              compile what it can, else eager  normal usage
  fail_on_recompile    ERROR on any recompile/break     dev — find problems
  eager_on_recompile   run eager on recompile           prod — graceful
  force_eager          ignore all torch.compile          debugging

Eager vs compiled (why this matters)

  Eager:     op1 → launch → op2 → launch → op3 → launch → op4 → launch
             4 Python dispatches, 4 kernel launches, 4 global memory round-trips

  Compiled:  op1+op2+op3+op4 fused → 1 launch
             0 Python overhead, intermediates stay in registers/shared memory      

Graph breaks — causes and fixes

- Graph break = Dynamo can't capture the op → current FX Graph ends, eager gap, new graph starts after
- Python `if` on a tensor value (e.g., `if x.sum() > 0:`) → forces tensor to Python bool → graph break
- Fix: replace with `torch.where(mask, f(x), g(x))` — stays in tensor ops, Dynamo captures the whole sequence
  - mask = x.sum(dim=1, keepdim=True) > 0
  - out = torch.where(mask, f(x), g(x))
- `torch.cond(pred, true_fn, false_fn)` — newer alternative, traces and compiles both branches
  - Requires: bool scalar predicate, same output structure/dtypes, consistent shapes
  - Data-dependent branches may still cause breaks
- Each new PyTorch release expands what Dynamo can capture without breaking
- Rule: stay in PyTorch tensor ops → no graph break → one big fused kernel

AOT Autograd — joint forward+backward tracing

- Takes Dynamo's FX Graph (forward only) as input
- Traces it through autograd to record backward ops → produces a joint fwd+bwd graph
- The joint graph enables cross-pass optimizations:
  - Fuse forward + backward elementwise ops into one kernel
  - Reuse intermediate buffers across fwd/bwd (reduce peak memory)
  - Common sub-expression elimination across both passes
- Guarantees same numerical results as eager autograd
- Runs automatically under torch.compile when gradients are involved (training)
- The joint graph is then sent to TorchInductor (or XLA) for kernel codegen

Pipeline so far

  Python code
      │
      ▼
  TorchDynamo → FX Graph (forward only)
      │
      ▼
  AOT Autograd → joint fwd+bwd FX Graph     ← this step
      │
      ▼
  TorchInductor → fused Triton/CUDA kernels


PrimTorch IR — simplified operator set

Mnemonic: "2,000 ops in, 250 prims out — PrimTorch makes every mutation explicit so the compiler can fuse fearlessly."

- PyTorch has 2,000+ ops in its full API — too many for a compiler backend to support
- PrimTorch IR reduces this to ~250 primitives (arithmetic, reductions, copy, reshape, etc.)
- All high-level/complex ops decompose into these 250 primitives

- Key transformation: in-place ops → functional + explicit copy
  - x.add_(y) becomes:  z = add(x, y) → copy_(x, z)
  - No hidden mutations — everything is explicit dataflow
  - Compiler can now reason about the graph without worrying about aliasing or side effects
  - Enables more aggressive fusion and memory planning

- Output: FX Graph containing only ATen IR + PrimTorch IR ops, ready for backend codegen
- Benefit for new hardware: implement 250 primitives, get all 2,000+ PyTorch ops for free

Pipeline so far:

  TorchDynamo → FX Graph (forward, ~2000 ATen ops)
      │
      ▼
  AOT Autograd → joint fwd+bwd FX Graph
      │
      ▼
  PrimTorch IR → same graph but ~250 primitives, no in-place mutations   ← this step
      │
      ▼
  TorchInductor (or XLA) → fused Triton/CUDA kernels

TorchInductor — backend code generation

- Final stage of torch.compile — takes joint fwd+bwd FX Graph (ATen + PrimTorch IR)
- Lowers abstract ops to loop-level IR (loops over tensor indices)
- Groups FX nodes into fused loop blocks → each group becomes one kernel
- Generates: Triton kernels (GPU), C++/OpenMP (CPU)
- Supports symbolic shapes → dynamic dimensions without recompile
- IR is inspectable — can debug and extend
- AOTInductor: compile via `torch.export()` once offline → save as .so artifact → deploy with zero compile overhead
- XLA: alternative backend for TPUs (Google), MTIA (Meta), Inferentia/Trainium (AWS)

torch.compile() and torch.export() architecture
================================================

                        ┌─────────────────────────────────┐  ┌──────────────────────────────┐
                        │         torch.compile()         │  │       torch.export()         │
                        │                                 │  │                              │
  ┌────────┐            │  1. Compiler frontend            │  │                              │
  │ Models │────────────┼──►┌──────────────┐               │  │                              │
  └────────┘            │   │ TorchDynamo  │───────────────┼──┼──►┌────────┐                 │──► XLA
       │                │   └──────┬───────┘               │  │   │ Export │                 │
       │                │          ▼                        │  │   └───┬────┘                 │──► TVM
       │                │   ┌──────────────┐               │  │       │                      │
       │                │   │AOT Dispatcher│───────────────┼──┼───────┘                      │──► ONNX
       │                │   └──────┬───────┘               │  │       │                      │
       │                │          │                        │  │       ▼                      │──► Others
       ▼                │          │                        │  │  ┌────────────┐              │
  ┌──────────┐          │          │                        │  │  │AOTInductor │              │
  │ PyTorch  │          │          ▼                        │  │  └─────┬──────┘              │
  │  eager   │          │  2. Graph compiler                │  │        │                     │
  │ library  │          │   ┌──────────────┐◄───────────────┼──┼────────┘                     │
  └──────────┘          │   │TorchInductor │               │  │                              │
                        │   └──────┬───────┘               │  │                              │
                        │          │                        │  │                              │
                        │          ▼                        │  │                              │
                        │  3. Kernel generator              │  │                              │
                        │   ┌────────┬────────┬────────┐   │  │                              │
                        │   │CPP/CPU │ Triton │ Halide │   │  │                              │
                        │   └────┬───┴────┬───┴────┬───┘   │  │                              │
                        │        │        │        │       │  │                              │
                        │        ▼        ▼        ▼       │  │                              │
                        │   ┌──────────────────────────┐   │  │                              │
                        │   │     HW compilers         │   │  │                              │
                        │   └──────────────────────────┘   │  │                              │
                        └─────────────────────────────────┘  └──────────────────────────────┘


Two paths through the stack:

  Path A: torch.compile() — JIT (compile at runtime)
  ────────────────────────────────────────────────────
  Model → TorchDynamo → AOT Dispatcher → TorchInductor → Triton/CPP/Halide → HW compilers
          (FX Graph)    (joint fwd+bwd)   (loop IR)       (kernel code)       (binary)

  Path B: torch.export() — AOT (compile once, deploy anywhere)
  ────────────────────────────────────────────────────────────
  Model → TorchDynamo → AOT Dispatcher → Export → AOTInductor → TorchInductor → Triton/CPP
          (FX Graph)    (joint fwd+bwd)   (save)   (AOT compile)  (loop IR)      (binary artifact)

  Export can also target: XLA, TVM, ONNX, others (non-Inductor backends)


Key:
  TorchDynamo    = captures Python bytecode into FX Graph (PEP 523)
  AOT Dispatcher = AOT Autograd + PrimTorch IR (joint fwd+bwd, 250 primitives)
  TorchInductor  = loop-level IR → fused kernel codegen
  AOTInductor    = ahead-of-time compilation via torch.export()
  Triton         = GPU kernel language (NVIDIA, AMD)
  CPP/CPU        = C++/OpenMP for CPU targets
  Halide         = image/array processing compiler (alternative codegen)
  HW compilers   = nvcc (NVIDIA), hipcc (AMD), gcc/clang (CPU)

TorchInductor — Triton codegen, autotuning, and optimizations

Inductor writes Triton, Triton writes PTX via LLVM — no NVCC, custom kernels on the fly, optionally wrapped in CUDA Graphs for replay.

- Inductor generates Triton code (Python DSL) from its loop-level IR
- Triton compiles to NVIDIA PTX directly via LLVM NVPTX — does NOT use NVCC
- This produces custom kernels on the fly, tailored to your specific model

- Loop-level IR is Python — inspectable and extensible
  - Each op becomes a Pointwise node: inner_fn (how to compute one element) + ranges (loop bounds)
  - Inductor iterates the ranges, calls inner_fn, generates corresponding Triton → PTX

- `torch.library.wrap_triton` with `triton_op` — register custom Triton kernels as first-class PyTorch ops
  - Gets autograd + faketensor support → Inductor can optimize it as part of the model graph

- Autotuning (what max-autotune mode does under the hood)
  - Benchmarks different kernel variants: block sizes, tile sizes, etc.
  - Picks the fastest variant per kernel, caches the config for subsequent runs
  - Increases initial compile time but produces highly optimized runtime kernels

- CUDA Graphs integration
  - Inductor wraps generated kernels into CUDA Graphs for replay with minimal CPU overhead
  - Triggered by reduce-overhead and max-autotune modes
  - Requires static shapes — NOT used when dynamic=True
  - max-autotune-no-cudagraphs = autotuning without CUDA Graph capture

- Other low-level optimizations
  - Index simplification (reduce complex index arithmetic in loops)
  - Common-subexpression elimination in generated code
  - Memory planning (reuse buffers, reduce allocations)

- Practical guidance
  - Start with default mode
  - Use max-autotune for large/critical workloads — significant compile time but faster runtime
  - Small models may not benefit much from max-autotune

Inductor codegen pipeline

  Loop-level IR (Python)
      │
      ▼
  Triton code (Python DSL)
      │
      ▼
  LLVM NVPTX               ← no NVCC involved
      │
      ▼
  NVIDIA PTX (GPU binary)
      │
      ▼
  (optional) CUDA Graph capture for replay

Compiler modes vs features used

  Mode                           Autotuning   CUDA Graphs   Compile time
  ─────────────────────────────  ───────────  ────────────  ────────────
  default                        basic         no            fast
  reduce-overhead                basic         yes           medium
  max-autotune                   aggressive    yes           slow
  max-autotune-no-cudagraphs     aggressive    no            slow

  dynamic=True with any mode → CUDA Graphs disabled (shapes must be static for capture)

Starting at 618: 12:30 || 22.5 pages to go 

TorchInductor routing:
  Your model → torch.compile
                  ├── large GEMMs ──────→ cuBLAS/CUTLASS (hand-tuned)
                  ├── fusable patterns ──→ Triton kernels (auto-generated)
                  └── caches best path per shape

  Transformer example:
  [layernorm + residual] ← fused Triton → [GEMM] ← cuBLAS → [activation] ← fused Triton

- Inductor fuses elementwise chains into one kernel; delegates large GEMMs to cuBLAS
- NVIDIA TE: NOT auto-used — must call TE modules explicitly; torch.compile fuses around them
- FlexAttention: compiles to fused Triton kernels (~85-90% FlashAttention perf, more flexible)
- Triton auto-enables warp specialization + TMA on Hopper/Blackwell when beneficial
- First compile = slow (autotune); subsequent runs = cached + fast  

Dynamic Shapes and Variable Sequence Lengths:

"Symbolic = algebra, not padding. One kernel fits a range, not one size."

HOW IT WORKS:
═══════════════════════════════════════════════════════════════

  1st call: seq_len=73
    Compiler: "I'll assume seq_len is static = 73"
    → compiles kernel A (for exactly 73)

  2nd call: seq_len=100
    Compiler: "shape changed! recompile once"
    → sets guard: seq_len ≤ 256 (dynamic from now on)
    → compiles kernel B with symbolic dim "S" where S ≤ 256
    → kernel B works for ANY seq_len ≤ 256

  3rd call: seq_len=42  → reuses kernel B ✅ (42 ≤ 256)
  4th call: seq_len=200 → reuses kernel B ✅ (200 ≤ 256)
  5th call: seq_len=500 → guard violated! recompile → new guard S ≤ 1024

  - No padding — kernel handles the exact input size using symbolic dimensions
- SymPy represents unknown dims as math symbols → generated code has flexible grid/block sizes
- Guards (e.g., seq_len ≤ 256) define the valid range for each compiled kernel
- Recompile only when guard is violated; cache grows over time for different ranges
- dynamic=True from start avoids the first recompile, but disables CUDA Graphs
- Prefer torch._dynamo.mark_dynamic() on just the varying dims

- Symbolic = compiler keeps dimensions as variables (S), not constants → one kernel handles any size in range
- SymPy does algebra through optimization passes so S stays symbolic (e.g., ceil(S/32) not ceil(73/32)=3)
- At launch time, S is just a kernel argument — no recompilation needed

- Guard = valid range for a symbolic dim (e.g., S ≤ 256); kernel reused for any S within range
- Guard fails → recompile with wider range → cache grows over time
- Data-dependent control flow = branches on tensor VALUES (not shapes) → can't symbolize → forces specialization + frequent recompiles
- Tip: bucket inputs by size to limit distinct shapes; use dynamic shapes for remaining variability
- Dynamic shapes consistently outperform padding (less wasted compute, less compile time)

TRADEOFF:
  Fixed shapes + padding       vs       Dynamic shapes (symbolic)
  ─────────────────────────             ─────────────────────────
  CUDA Graphs work ✅                   CUDA Graphs disabled ❌
  No recompiles ✅                      Occasional recompiles ❌
  Wasted compute on padding ❌          No wasted compute ✅
  Best when lengths vary <20%           Best when lengths vary widely

  - dynamic=True disables CUDA Graphs (graphs need fixed shapes + fixed memory addresses)
- mode="reduce-overhead" = CUDA Graphs → only with stable shapes
- mode="default" or "max-autotune-no-cudagraphs" → for variable lengths
- If lengths vary ≤20% → pad to fixed size + CUDA Graphs is often faster
- If lengths vary widely → dynamic shapes avoids massive padding waste
- Dynamic shapes = slightly higher memory (extra guards + generalized code)
- Always profile both approaches for your workload

Disabling the PyTorch Compiler and Reverting Back to Eager Mode

- @torch.compiler.disable decorator → disable compilation for a specific function
- torch.compiler.set_stance() → context manager for region-scoped control (eager within a block)
- torch.compile(model, backend="eager") → revert entire model to eager mode
- Use cases: A/B testing compiled vs eager, isolating issues, skipping untraceable code, keeping graph focused on compute

The "SymPy symbolic tracing" sounds fancy, but the end result is just: don't hardcode the size, pass it as an argument.

Performance Hints and Debugging Generated Code

- TORCH_LOGS="perf_hints" → shows missed optimizations (why fusion/CUDA Graphs failed)
- TORCH_LOGS="output_code" → prints generated kernel source code
- TORCH_COMPILE_DEBUG=1 → full debug dir with FX graph, IR dumps, Triton sources, PTX
- Use these to verify fusion, warp specialization, Tensor Core usage (mma.sync in PTX)
- Spot inefficiency → write custom Triton kernel for that specific pattern

Debugging Numerical Correctness and Accuracy

- torch.compile CAN introduce numerical differences vs eager mode (rare but possible)
- Causes: kernel fusion reorders FP math, mixed precision (BF16/FP16) in fused ops, RNG sequence changes
- Debug: minifier → smallest repro script; REPRO_AFTER="aot" + REPRO_LEVEL=4 → compare each compiler stage vs eager
- Random mismatch: fallback_random=True forces eager-matching RNG (slower)
- FP mismatch: test in full FP32 + set_float32_matmul_precision('highest') to isolate
- Determinism: torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8 (cuBLAS split-K is nondeterministic by default)
- torch._dynamo.explain() → overview of graph breaks and subgraphs

Explaining and Minimizing Graph Breaks

WHAT GRAPH BREAKS LOOK LIKE:
═══════════════════════════════════════════════════════════════

  YOUR CODE:
    x = a / (abs(a) + 1)      ─┐
                                ├── Graph 1 (compiled, fast)
    print("woo")               ─┘ ← BREAK (side effect)
                                   ← eager Python runs print()
    if b.sum() < 0:            ─┐
        b = -b                  ├── Graph 2 (compiled, fast)
                               ─┘ ← BREAK (data-dependent branch)
                                   ← eager Python picks branch
    return x * b               ─── Graph 3 (compiled, fast)


  IDEAL (no breaks):           REALITY (with breaks):
  ┌────────────────────┐       ┌──────┐ eager ┌──────┐ eager ┌──────┐
  │  ONE BIG GRAPH     │       │Graph1│►print►│Graph2│►if/el►│Graph3│
  │  (fully optimized) │       └──────┘       └──────┘       └──────┘
  └────────────────────┘       ↑ each gap = Python overhead + lost fusion

- Graph break = compiler gives up, falls back to eager Python for that line
- Each break = lost fusion opportunity + Python overhead between graphs
- Common causes: print(), data-dependent if, unsupported ops, collective comms (FSDP)
- Debug: dynamo.explain(model)(inputs) → shows break count, exact lines, reasons
- Fix print: wrap in `if not torch._dynamo.is_compiling()`
- Fix data-dependent if: refactor to torch.where() or accept the break
- Goal: one big graph for the whole model/training step  

Handling Data-Dependent Branches (Avoiding Graph Breaks)

PROBLEM:
  if b.sum() < 0:    ← Python if on tensor value = GRAPH BREAK
      b = -b

FIX 1: torch.cond() — captures BOTH branches as graph subroutines
  b = torch.cond(b.sum() < 0, lambda b: -b, lambda b: b, (b,))
  ✅ no graph break
  ❌ restrictions: same input/output shape+dtype, no side effects, tensor predicate only

FIX 2: torch.where() — pure tensor masking, no branches at all
  b = torch.where(b.sum() < 0, -b, b)
  ✅ no graph break, no restrictions, simpler
  ❌ computes BOTH -b and b always (minor waste)

- Python `if` on tensor values = graph break (compiler must pick one path)
- torch.cond() = both branches captured in graph, GPU picks at runtime (like GPU if/else)
- torch.where() = mask-based selection, no branching at all — simpler and preferred
- Use torch.where() when possible; torch.cond() when branches are complex/different ops

WHAT TRIGGERS RECOMPILATION:
═══════════════════════════════════════════════════════════════

  1. Dynamic shape guard fails       (seq_len exceeded range)
  2. Data-dependent branch changes    (if b.sum()<0 took other path)
  3. New tensor dtype/device seen
  4. Python global variable changed

- fail_on_recompile = crash on any guard violation instead of silently recompiling → catches graph breaks early in dev/CI

Minimize Graph Recompilations
- Symptom: iterations stay slow after warmup → likely excessive recompiles
- Debug: TORCH_LOGS="graph_breaks,recompiles,guards" → see which guard is failing
- Common cause: Python-level value (seed, timestamp, counter) changing every iter → guard on exact value → recompile
- Fix varying constants: pass as tensor (compiler guards shape/dtype, not value)
- Fix varying shapes: mark_dynamic(tensor, dim) → symbolic from start, skips discovery recompile
- Safety net: set_stance("eager_on_recompile") → caps recompiles, falls back to eager after N failures

- Constants: recompilations are totally unnecessary — value changes but nothing structural (shape/memory) changes. As a tensor, value flows as runtime data, not a compile-time constant.
- mark_dynamic: tells compiler to use SymPy from the start — it already knows the dim is dynamic, no wasted discovery recompile.
- Both = "tell the compiler upfront what varies, so it doesn't learn the hard way."

Mark Functions and Code Blocks as Safe with allow_in_graph:

- allow_in_graph = tell Dynamo "trust me, this function is pure" → skip safety checks → no graph break
- Use as decorator (@torch._dynamo.allow_in_graph) or context manager
- You're promising: no side effects, same input → same output, depends only on tensor inputs
- If you're wrong → silent wrong results (no error, just bad math)
- Last resort — fix the actual graph break cause first, use this only when you're sure it's safe

Tips for Handling Graph Breaks

  In-place operations:
    - x.relu_() creates aliasing ambiguity → compiler can't track which version of x each node sees → may break
    - Fix: rewrite to out-of-place x = x.relu()

  Python data structures:
    - Avoid Python lists/loops for tensors → use torch.stack(), preallocate instead
    - Remove print, logging, math.*, I/O from perf-critical paths

  General rule:
    - Stay in PyTorch tensor ops, avoid Python-native equivalents

  Data-dependent control flow:
    - if tensor.sum() > 0: → unknown at compile time → graph break
    - Fix: torch.where()/masks (preferred), torch.cond() for complex branches
    - If neither works → accept the break

  DDP (Distributed Data Parallel):
    - Graph breaks at all-reduce buckets are INTENTIONAL — needed for compute/comms overlap
    - Each bucket compiled separately so gradient sync overlaps with next bucket's backward
    - Can't eliminate these breaks — they're necessary for efficient distributed training

  FSDP (Fully Sharded Data Parallel):
    - Wrap each transformer block as its own FSDP submodule
    - Dynamo breaks at each submodule boundary → allows shard comms to overlap with compute
    - Same idea as DDP buckets — intentional breaks for overlap


  Compiled FSDP memory benefits:
    - AOT Autograd + Inductor fuse fwd/bwd passes, reuse buffers across shards
    - Only active parameter slices + minimal intermediates resident per GPU → lower peak memory vs DDP/eager
    - Wrap submodules individually (each transformer block) — otherwise falls back to one big bucket (less overlap, less memory savings)
    - Always test on small config first — debugging compiled FSDP at scale is very complex

  Custom CUDA/Triton ops:
    - Unknown custom CUDA C++ extensions → Dynamo can't reason about them → graph break
    - Fix: rewrite in Triton + register via torch.library.triton_op() → compiler can optimize it
    - Check if third-party libraries already provide Triton/Dynamo wrappers before writing your own

Debugging Compiler Phases, Graph Breaks, and Performance

  TORCH_LOGS (env var) — pick what to debug:
    - graph_breaks    → when/where graphs split
    - recompiles      → what triggers recompilation
    - guards          → guard evaluations (shape, dtype, etc.)
    - perf_hints      → missed optimization opportunities
    - output_code     → generated kernel source code
    - dynamo          → verbose TorchDynamo internals
    - aot_graphs      → verbose AOT Autograd internals
    - inductor        → verbose TorchInductor internals
    - dynamic         → dynamic shape decisions

  Deeper debugging:
    - TORCH_COMPILE_DEBUG=1 → full debug dir (FX graph, IR, Triton source, HTML report)
    - TORCHDYNAMO_REPRO_AFTER + REPRO_LEVEL → dump graph per stage, compare vs eager
    - TORCH_TRACE=<dir> + tlparse → stack-frame tree of compilation events
    - Perfetto UI → trace timeline visualization (low overhead, usable in production)

  Tip: start with just "graph_breaks" — output gets very verbose quickly

Writing Custom Kernels with OpenAI Triton

- Triton = Python-native DSL for writing GPU kernels + JIT compiler that compiles to PTX (no CUDA C++ needed)
- TorchInductor uses Triton as its backend codegen — understanding Triton lets you inspect/customize what Inductor generates
- Custom Triton kernels can beat Inductor-generated code with domain-specific knowledge (complex sparsity, novel layers)
- Tradeoff: custom kernels require ongoing maintenance + potential rewrites for new hardware
- NVIDIA competing with Python-centric alternatives (cuTile, CuTe Python, CUTLASS Python) — but Inductor still uses Triton as primary path
- Use torch.compile first; write custom Triton only when Inductor can't produce optimal code for your specific pattern

Triton Programming Model:

  Triton kernel code                      What actually happens
  ─────────────────                       ─────────────────────
  pid = tl.program_id(0)                  → "which block am I?"
  offsets = pid * BLOCK + tl.arange(0,BLOCK)  → "my chunk of data"
  x = tl.load(x_ptr + offsets, mask=mask) → load entire chunk at once
  result = x + y                          → operate on entire chunk
  tl.store(out_ptr + offsets, result)     → store entire chunk

  You wrote: 0 threads, 0 warps, 0 shared memory, 0 barriers
  Triton handles all of that under the hood

- @triton.jit decorator = define a Triton kernel (Python function → GPU code)
- tl.program_id(axis) = block index (like CUDA blockIdx.x)
- tl.arange(0, BLOCK_SIZE) = vectorized range for the whole block (not one thread's element)
- tl.load/tl.store with mask = guarded vectorized memory access (handles out-of-bounds)
- BLOCK_SIZE: tl.constexpr = compile-time constant (how many elements per program)
- One Triton program = one CUDA thread block (CTA) — threads/warps inside are invisible
- No guaranteed one-element-to-one-thread mapping — Triton compiler decides thread layout

## Triton Launch, Shared Memory & PyTorch Registration (pp. 640-642)

### Launching Triton Kernels
- grid function returns tuple of #program instances: triton.cdiv(n_elements, BLOCK_SIZE)
- Launched as: kernel[grid](args..., BLOCK_SIZE=1024)
- num_warps controls threads per CUDA block — NOT the same as BLOCK_SIZE
- mask = offsets < n_elements prevents OOB access (Triton's version of CUDA's if(idx<N))

### Under the Hood
- Each Triton "program" → one CUDA thread block
- tl.arange → per-lane indices; compiler maps vectorized index space across threads
- Triton auto-coalesces memory, auto-vectorizes arithmetic — you write tile-level, it handles thread-level

### Accessing Shared Memory in Triton
- No explicit shared-memory allocator (unlike CUDA's __shared__)
- Instead: tl.make_tensor_descriptor(...) stages tiles into shared memory
- Pipelined via tl.range(..., num_stages=...) → lowers to cp.async + TMA + barriers
- Pattern: load tile from A & B into SMEM → reuse for many computes → reduce HBM traffic

### Registering Custom Triton Kernels with PyTorch
- Problem: raw Triton kernels are opaque to torch.compile → causes graph breaks
- Solution: @triton_op("lib::op_name", mutates_args=()) registers kernel as a PyTorch op
- wrap_triton(kernel) makes it inlineable/fuseable within the torch.compile graph
- Call via: torch.ops.my_triton_lib.vector_add(a, b)

### Training Support (Autograd)
- Forward-only: just @triton_op is enough
- For training: register backward via vector_add.register_autograd(backward, setup_context=...)
- Alternative: torch.autograd.Function (but register_autograd preferred for torch.compile)

Flow:  @triton.jit kernel
            │
       @triton_op ──► registers name + mutation metadata with PyTorch
            │
       wrap_triton ──► makes kernel visible to torch.compile graph
            │
       torch.compile ──► can now fuse/reorder/inline the Triton kernel

"TOP-W: Triton_Op registers the Path, Wrap_triton opens the gate" — two steps to make Triton visible to the compiler.     

3/8/26: 
Start at 643

Triton kernel tuning, autotuning, and advanced implementations

- If Triton can't do what you need → fall back to CUDA C++ (e.g., CUTLASS), register as PyTorch extension with autograd

- Kernel launch parameters
  - Default: 4 warps (128 threads) per block
  - Modern GPUs (larger shared mem, register files): push to 8 warps (BLOCK_SIZE >= 2048) or 16 warps (BLOCK_SIZE >= 4096)
  - Compute-heavy kernels → more warps helps (higher occupancy)
  - Memory-bound kernels → more warps hides latency, but too many → contention/cache thrashing
  - Manual tuning is tedious → use the autotuner

- Triton autotuner (`@triton.autotune`)
  - Decorate kernel with a list of `triton.Config` objects (BLOCK_SIZE, num_warps, num_stages, tile size)
  - First invocation: JIT-compiles and benchmarks every config → caches the fastest one
  - Subsequent calls with same input shape → reuse cached config (zero tuning cost)
  - New input shape → re-tunes, caches separately
  - Warm up with realistic/production inputs to get optimal configs
  - Custom `key_fn` for advanced per-shape cache control

- The occupancy trade-off
  - Larger tiles + more warps → higher arithmetic intensity, fewer blocks per SM (less occupancy)
  - Smaller tiles + fewer warps → more concurrent blocks per SM, less data reuse
  - Autotuner finds the sweet spot automatically

- Warp specialization in Triton
  - Split warps into producer (memory) and consumer (compute) roles
  - Producer prefetches next tile via TMA while consumer computes current tile
  - Enable with `tl.range(..., warp_specialize=True)` + `num_stages > 1`
  - Or via autotune config: `num_consumer_groups=2`, `num_buffers_warp_spec=3`
  - Especially effective for large-K GEMMs — keeps memory subsystem and ALUs busy simultaneously
  - TorchInductor can emit warp-specialized Triton code automatically for its generated kernels

Autotuner flow

  First call with shape [M, N, K]
      │
      ▼
  Benchmark Config 1: BLOCK=128, warps=4, stages=1  → 1.2 ms
  Benchmark Config 2: BLOCK=256, warps=8, stages=2  → 0.8 ms  ← winner
  Benchmark Config 3: BLOCK=512, warps=16, stages=3 → 0.9 ms
      │
      ▼
  Cache: shape [M,N,K] → Config 2

  Subsequent calls with same shape → Config 2 reused instantly
  New shape [M2, N2, K2] → re-benchmark all configs → cache new winner

Warp specialization overlap

  Without warp specialization:
    warp 0-7:  load tile 0 ──── compute tile 0 ──── load tile 1 ──── compute tile 1
                idle                                  idle

  With warp specialization:
    producer warps:  load tile 1 ──── load tile 2 ──── load tile 3 ────
    consumer warps:       compute tile 0 ──── compute tile 1 ──── compute tile 2
                     ▲
                     └── memory and compute fully overlapped

3/9/26: 

Software pipelining in Triton

  Pattern: prefetch-next → compute-current → swap

  num_stages=2 (double buffer):
    ┌──────────┐┌──────────┐┌──────────┐
    │ load T1  ││ load T2  ││  idle    │
    │compute T0││compute T1││compute T2│
    └──────────┘└──────────┘└──────────┘
    2 buffers in shared memory

  num_stages=3 (triple buffer):
    ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
    │ load T2  ││ load T3  ││ load T4  ││  idle    │
    │ load T1  ││ load T2  ││ load T3  ││  idle    │
    │compute T0││compute T1││compute T2││compute T3│
    └──────────┘└──────────┘└──────────┘└──────────┘
    3 buffers — hides more latency, costs more shared memory

  - Don't overwrite current tile reference before compute finishes — compiler may reorder and break overlap
  - More stages = diminishing returns if compute-bound or shared memory is tight

Triton Proton profiler

  - Separate profiling package for Triton kernels
  - Emits NVTX ranges → visible in Nsight Systems timelines
  - Usage: proton.start("name", hook="triton") → with proton.scope("name", metadata) → finalize
  - Outputs hierarchical timing table with derived metrics (TFLOPS, bandwidth)
  - Supply metadata (FLOPS count) → shows how close you are to hardware peak
  - Workflow: Proton summary → pinpoint kernel → Nsight Systems timeline → Nsight Compute deep dive
  - Note: for many shapes/precisions, cuBLASLt/CUTLASS may match or beat custom Triton kernels

PyTorch XLA backend

  - Alternative to TorchInductor — targets TPUs (Google), MTIA (Meta), Inferentia/Trainium (AWS)
  - Activate: torch.compile(..., backend="openxla")
  - Captures whole-program static graph ahead of time (not incremental like Inductor)
  - Optimized for static shapes — new shapes trigger full whole-graph recompilation (expensive)
  - Fix: pad inputs or use fixed-size buckets to avoid recompilation
  - Caches compiled graphs per shape signature — improves after warm-up
  - Same principles apply: minimize graph breaks, use distributed strategies (DP, MP)
  - Not commonly used with NVIDIA GPUs — use TorchInductor for those

  TorchInductor vs XLA

  Feature               TorchInductor             XLA
  ────────────────────  ────────────────────────  ─────────────────────────
  Target HW             NVIDIA/AMD GPUs, CPU      TPUs, MTIA, Inferentia
  Dynamic shapes        symbolic (incremental)    recompile per new shape
  Compilation style     incremental mid-run       whole-graph ahead of time
  Kernel codegen        Triton / C++              XLA HLO → device runtime
  Default for NVIDIA    yes                       no    

PyTorch XLA backend

  - Alternative to TorchInductor — targets TPUs (Google), MTIA (Meta), Inferentia/Trainium (AWS)
  - Activate: torch.compile(..., backend="openxla")
  - Captures whole-program static graph ahead of time (not incremental like Inductor)
  - Optimized for static shapes — new shapes trigger full whole-graph recompilation (expensive)
  - Fix: pad inputs or use fixed-size buckets to avoid recompilation
  - Caches compiled graphs per shape signature — improves after warm-up
  - Same principles apply: minimize graph breaks, use distributed strategies (DP, MP)
  - Not commonly used with NVIDIA GPUs — use TorchInductor for those

it said the xla will trigger whole program compilation for dynamic shape. how is that different from the torch inductor also triggering recompilation?

TorchInductor vs XLA

Feature               TorchInductor             XLA
────────────────────  ────────────────────────  ─────────────────────────
Target HW             NVIDIA/AMD GPUs, CPU      TPUs, MTIA, Inferentia
Dynamic shapes        symbolic (incremental)    recompile per new shape
Compilation style     incremental mid-run       whole-graph ahead of time
Kernel codegen        Triton / C++              XLA HLO → device runtime
Default for NVIDIA    yes                       no

Chapter 14 — key takeaways

- torch.compile mode selection
  - "default" for quick startup, "max-autotune" for max performance
  - Always do warm-up iterations before measuring
  - Short jobs / small models: compile overhead may not pay off

- Set performance flags early (before any computation)
  - torch.set_float32_matmul_precision("high") → TF32 fast path
  - torch.backends.cuda.matmul.allow_tf32 = True
  - torch.backends.cudnn.allow_tf32 = True
  - enable_flash_sdp(True), enable_mem_efficient_sdp(True)

- Minimize graph breaks
  - Inspect: torch._dynamo.explain() or TORCH_LOGS="graph_breaks"
  - Fix: remove prints, refactor Python if → torch.cond / torch.where
  - Move non-critical Python processing out of forward()
  - Goal: long, purely tensor-in tensor-out code path

- Dynamic shapes
  - dynamic=True forces all dims symbolic upfront — one compiled model handles many shapes
  - mark_dynamic(tensor, dim) for selective dims only
  - Trade-off: disables CUDA Graphs, adds extra guards
  - Hybrid: bucket inputs by size + dynamic=True for remaining variability

- Profile for recompilation guards
  - TORCH_LOGS="graph_breaks,guards,recompiles" → find which guard triggers
  - Common culprits: Python random values, changing tensor rank, mixed device/dtype

- Avoid recompilations
  - Well-tuned loop = zero recompiles after first few iterations
  - Continued recompiling = something changes every iteration (debug prints, counters, mixed CPU/GPU tensors)
  - Use set_stance() + guard logging to catch

- Tune memory usage
  - Compiled mode can use more memory (larger fused kernels, guard buffers)
  - OOM fixes: smaller BLOCK_SIZE, disable certain fusions, compile submodules separately
  - Free large intermediates promptly — they persist longer in compiled graph lifecycle

- Combine with distributed training
  - DDP/FSDP have intentional graph breaks at communication points — these are expected
  - FSDP + torch.compile: wrap submodules for shard-wise compilation
  - Graph breaks for gradient pre/post-reduction are handled — focus on forward/backward being compiled

- TORCH_LOGS="perf_hints" for missed optimizations
  - Tells you if CUDA Graphs weren't used (e.g., input mutation), or if an op fell back to eager
  - Often suggests the workaround directly

- Debug with small inputs first
  - Small tensors → fast correctness checks → then scale up
  - TORCH_LOGS="output_code" to inspect generated kernels on small cases

- Custom kernels only for true bottlenecks
  - Profile first — TorchInductor already fuses most things
  - Custom Triton only where Inductor falls short (atypical fusion, custom activation)
  - Weigh maintenance cost: custom kernels need updates on hardware changes
  - Consider filing issues for Inductor to support the pattern natively

Chapter 14 — key takeaways (continued)

- Triton best practices
  - Coalesced memory accesses, avoid shared memory bank conflicts (pad if needed)
  - Mask tl.load() / tl.store() at boundaries
  - Block/tile sizes = multiples of 32 (warp-aligned)
  - Tile sizes that fit L1/shared memory carve-out
  - Start with num_warps ∈ {4, 8}, num_stages ∈ {2, 3, 4}, then autotune
  - Check existing community implementations (FlashAttention, etc.) before writing from scratch

- Cache hint caution
  - Some PTX load hints (non-coherent, L1 no-allocate, 256B L2 prefetch) are undocumented
  - Can break across driver versions and GPU generations
  - Prefer Triton's portable knobs: eviction_policy="evict_last", cache_modifier=
  - Profile when migrating to different CUDA drivers or hardware
  - Conflicting cache hints are extremely hard to debug

- Stay up to date
  - Each PyTorch/Triton release: more operator support, fewer graph breaks, free speedups
  - Especially important as GPU hardware evolves rapidly

- Conclusion
  - Use profilers together: Nsight Systems + Nsight Compute + PyTorch profiler + Triton Proton
  - Iterative tuning: fix one bottleneck → next one emerges (GPU kernel → CPU overhead → input pipeline)
  - Compiler handles a big piece, but also optimize data loading, I/O, and algorithmic choices

Chapter 14 — full pipeline recap

  Python model
      │
      ▼
  TorchDynamo ─── FX Graph (captures tensor ops, PEP 523)
      │
      ▼
  AOT Autograd ── joint fwd+bwd graph (cross-pass fusion)
      │
      ▼
  PrimTorch IR ── 2000+ ops → 250 primitives (no in-place mutations)
      │
      ▼
  TorchInductor ─ loop-level IR → fused kernels
      │
      ├──► Triton → LLVM NVPTX → PTX (GPU)
      ├──► C++/OpenMP (CPU)
      └──► Halide (alternative)
               │
               ▼
          (optional) CUDA Graph capture for replay

  Alternative path: torch.export() → AOTInductor → deploy .so artifact
  Alternative backend: XLA → TPUs, MTIA, Inferentia



  
