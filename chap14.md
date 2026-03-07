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