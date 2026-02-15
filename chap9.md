Increasing CUDA Kernel Efficiency and Arithmetic Intensity: 

- No. of FLOPS per byte of data transferred. 
- Roofline Model: Kernel Performance(FLOPS/sec) against Arithmetic Intensity (FLOPSs/byte)
- Some improvements: Improve the algorithm, reuse data, fuse operations, and increase batch sizes to raise arithmetic intensity without changing the algorithm's result. 

Multilevel Microtiling and Software Prefetching: 
-  Microtiling is simple -- load tile from global DRAM to shared memory and then do vectorized loads of microtiles into registers using stuff like float4 and <half2>. 

OLD WAY (manual tiling):
  Developer manually moves data at every level:

  DRAM → (you load) → Shared Memory → (you load) → Registers → Tensor Core
                ↑                            ↑
          Your code: global load       Your code: float4 load
          coalesced into shmem         from shmem to registers


NEW WAY (Blackwell/modern GPUs):
  Hardware + compiler handle the inner levels:

  DRAM → (you load) → Shared Memory → (hardware moves) → TMEM → Tensor Core
              ↑                              ↑
        You still do this            tcgen05 instructions
        (cp.async or TMA)            handle this automatically

WHAT IS TMEM?
═══════════════════════════════════════════════════════════

  TMEM = Tensor Memory
  A dedicated memory space ONLY Tensor Cores can access
  
  DRAM (big, slow)
    └→ Shared Memory (fast, programmer-visible)
         └→ TMEM (fastest, Tensor Core private, compiler-managed)
              └→ Tensor Core (compute)

  You don't read/write TMEM directly — compiler & hardware manage it
  It's like registers, but specifically shaped for matrix fragments

WHAT IS tcgen05?
═══════════════════════════════════════════════════════════

  tcgen05 = Tensor Core Generation 05 (Blackwell's Tensor Core instructions)
  
  These instructions do TWO things at once:
    1. Move data: Shared Memory → TMEM
    2. Compute: Matrix multiply-accumulate (MMA)
  
  "Implicitly stage" = the instruction handles data movement FOR you
  You just say "multiply these tiles" and hardware fetches from shmem into TMEM


WHAT ARE cp.async AND TMA?
═══════════════════════════════════════════════════════════

  cp.async = Copy Async: DRAM → Shared Memory without blocking the thread
  TMA = Tensor Memory Accelerator: hardware unit that does DRAM → Shared Memory
        with built-in address calculation (no thread math needed)
  
  Both handle the DRAM → Shared Memory step
  tcgen05 handles the Shared Memory → TMEM step

- Unified memory eases development but may not produce the best performance. Expert users often prefer explicit cudaMemcpy or pinned memory allocations to fully avoid page migration overheads.
- Physical memory is divided into fixed-size CHUNKS called pages.
Typical page size: 4 KB (CPU) or 64 KB (GPU unified memory)


Tiling with Thread Block Clusters: 
- CUDA thread-block clusters from Cooperative Groups allow multiple thread blocks to share data using distributed shared memory -- can be used to batch load data using multiple blocks and TMA for tiling purposes -- using something called multicast. 

WITHOUT CLUSTERS (traditional):
═══════════════════════════════════════════════════════════

  4 thread blocks each need the SAME tile of matrix A:

  DRAM:  [A tile]
            │
            ├──→ CTA 0 loads A tile into its SMEM  (128 bytes from DRAM)
            ├──→ CTA 1 loads A tile into its SMEM  (128 bytes from DRAM)
            ├──→ CTA 2 loads A tile into its SMEM  (128 bytes from DRAM)
            └──→ CTA 3 loads A tile into its SMEM  (128 bytes from DRAM)

  Total DRAM traffic: 4 × 128 = 512 bytes
  Same data loaded 4 times! 😢


WITH CLUSTER (2×2, multicast):
═══════════════════════════════════════════════════════════

  4 CTAs in a cluster share via DSMEM + TMA multicast:

  DRAM:  [A tile]
            │
            └──→ TMA loads ONCE → multicasts to all 4 CTAs' SMEM simultaneously
                      │
                 ┌────┼────┬────┐
                 ▼    ▼    ▼    ▼
               CTA0  CTA1 CTA2 CTA3
               SMEM  SMEM SMEM SMEM

  Total DRAM traffic: 1 × 128 = 128 bytes
  4× reduction! ✅

- 4 Thread blocks -- 4x the DRAM memory load speed 
- On B200, default is 8, but you can increase the no. of thread blocks per cluster to be 16 -- comes at a cost though 

Kernel Fusion: 
- Fusion = merge multiple kernels into one so intermediates stay in registers, never hit DRAM → higher AI (FLOPs/byte)
- Tradeoff: more fusion = more registers/thread → can reduce occupancy or spill to local memory — always profile
- torch.compile / TorchInductor auto-fuses elementwise ops; manual fusion for complex patterns (reductions, norms)
- Vertical fusion = chain sequential ops on same data (sin→sqrt); Horizontal fusion = combine parallel ops across data
- Micro-opt: replace divide with rsqrtf * multiply — faster instruction, but only matters if compute-bound
- Rule of thumb: if data is read more than once by same block, stage it in shared memory to kill redundant global loads
- CUTLASS, Triton, TorchInductor help write fused kernels with Tensor Cores + TMA + TMEM

Structured Sparsity: 
- 2:4 structured sparsity = exactly 2 of every 4 weights are zero → hardware skips zeros, doubles Tensor Core throughput
- Applied post-training (pruning) for inference only — training gradients don't benefit
- Sparse Tensor Cores operate on half-width data, doing 2× work per cycle on nonzero elements
- Increases AI by cutting memory traffic ~50% (don't load zeros) while keeping same useful FLOPs
- PyTorch: to_sparse_semi_structured() converts dense → 2:4 sparse format for Sparse Tensor Cores
- Needs large matmuls + large batches to amortize index/compression overhead — small batches see less benefit
- Only 2:4 pattern is hardware-accelerated — arbitrary sparsity gets no special acceleration
- Apply AFTER basic optimizations (coalescing, tiling, fusion) are already in place

Recomputation vs. Memory Trade-Off: 
- Calculate x^2 twice instead of storing and reading it as memory is expensive. 

PyTorch and Arithmetic Intensity: 
- PyTorch auto-fuses elementwise ops via torch.compile and uses cuDNN/cuBLAS for tiled matmuls — you get tiling, fusion, shared memory for free
- SDPA dispatches to FlashAttention/cuDNN automatically — control with sdpa_kernel(SDPBackend.FLASH_ATTENTION)
- Prefer high-level ops (torch.matmul, nn.functional) over many small kernels — libraries call optimized fused kernels
- Custom/nonstandard ops may not get auto-fused — manual optimization still needed for those

2/15/26: 
- On variable length sequences, use PyTorch's nested/rugged tensors
- Batch of 3 sequences, lengths: [3, 5, 2]

With padding (standard approach, pad to max_len=5):
┌───────────────────────────┐
│ seq1: [A B C 0 0]         │  ← 2 zeros = wasted memory loads
│ seq2: [D E F G H]         │  ← full, no waste
│ seq3: [I J 0 0 0]         │  ← 3 zeros = wasted memory loads
└───────────────────────────┘
Total: 15 elements loaded, only 10 are useful = 33% waste
Attention computes on zeros too = wasted FLOPS

With Nested/Ragged Tensors — Pack Tightly:
┌───────────────────┐
│ [A B C D E F G H I J]  ← one contiguous buffer, NO zeros
└───────────────────┘
+ metadata: offsets = [0, 3, 8, 10]  ← "seq1 starts at 0, seq2 at 3, seq3 at 8"

- Especially valuable for LLM attention with mixed sequence lengths (e.g., batch has lengths [32, 512, 128]).
- Still maturing in PyTorch — verify operator coverage for your workload and profile both memory traffic and kernel time.

Mixed Precision and Utilizing Tensor Cores:
- NVIDIA GPU's have TF32, FP16, FP8, FP4, INT8 in Tensor Cores 
- Mixed precision solves memory-bound primarily because smaller datatypes = fewer bytes moved for the same FLOPS = higher AI.
- Tensor Cores raise the compute roof (more peak FLOPS), TMA+TMEM feed them without stalls — together they make it nearly impossible to stay memory-bound.

Blackwell SM:
  ┌──────────────────────────────────────┐
  │  Register File (256 KB)              │ ← general purpose: local vars, addresses, loop counters
  │                                      │
  │  TMEM (256 KB)                       │ ← dedicated: Tensor Core accumulators ONLY
  │                                      │
  │  Shared Memory (228 KB)              │ ← shared across block: tiles, communication
  │                                      │
  │  CUDA Cores (ALUs)                   │ ← scalar FP32/INT ops
  │  Tensor Cores                        │ ← matrix ops (MMA)
  └──────────────────────────────────────┘
Inside one SM:
  ┌─────────────────────────────────────┐
  │  4 Warp Schedulers                  │
  │     │                               │
  │     ├──→ CUDA Cores (128 per SM)    │  ← scalar ops: add, multiply, compare
  │     │    one thread = one operation  │     e.g., x = a + b
  │     │                               │
  │     ├──→ Tensor Cores (4 per SM)    │  ← matrix ops: 16×16 MMA in one instruction
  │     │    one warp = one matrix op   │     e.g., D = A×B + C (whole tile at once)
  │     │                               │
  │     ├──→ Load/Store Units           │
  │     └──→ Special Function Units     │  ← sin, cos, sqrt, etc.
  └─────────────────────────────────────┘

- Each Blackwell SM has 256 KB TMEM dedicated to Tensor Core accumulators — frees registers for other work, boosts occupancy.
- TMA asynchronously copies tiles from global → shared memory, Tensor Core instructions implicitly move data from shared memory → TMEM — no SM intervention needed.
- Lower precision (FP16/FP8/FP4) = fewer bytes per element = higher arithmetic intensity for same FLOPS = shifts kernel from memory-bound toward compute-bound.
- AI boost is multiplicative: FP16 = 2× AI, FP8 = 4× AI, FP4 = 8× AI compared to FP32.
- In Nsight Compute, successful mixed precision shows: memory stalls drop, dependency/pipeline stalls rise — that means you shifted the bottleneck from memory to compute.
- Verify with Nsight Compute Roofline chart (AI should shift right) and Speed of Light panel (memory util down, compute util up).

- CUTLASS = CUDA Templates for Linear Algebra Subroutines. It's NVIDIA's open-source C++ template library for writing high-performance GEMM (matrix multiply) and convolution kernels.

Without CUTLASS:
  You manually write: TMA loads → shared mem staging → TMEM allocation → PTX MMA instructions → pipeline sync
  = hundreds of lines of expert-level CUDA

With CUTLASS:
  You configure a template: "I want FP16 GEMM, 128×128 tiles, 2-stage pipeline"
  CUTLASS generates the optimized kernel with TMA + TMEM + Tensor Core instructions automatically

- It's what cuBLAS uses under the hood. PyTorch → cuBLAS/cuDNN → CUTLASS-level code → Tensor Cores.
- CUDA Pipeline API (<cuda/pipeline>) = formal producer/consumer sync for overlapping async memory copies with Tensor Core compute.
- cuda::memcpy_async copies global → shared memory without blocking the SM — the SM can compute on previous tiles while the next tile loads.
- Double buffering (ping-pong): two shared memory buffers — load into one while computing from the other → memory latency hidden behind compute.
- Tensor Cores work without pipelining, but without it the SM stalls waiting for data — pipelining is what makes them run at full throughput.
- You don't manually manage TMEM — hardware and libraries (CUTLASS) handle operand movement between shared memory and TMEM automatically.

TF32 and Automatic Mixed Precision(PyTorch):

  - TF32 = FP32 exponent (8-bit range) + FP16 mantissa (10-bit precision) → runs on Tensor Cores, not CUDA cores → higher TFLOPS, same dynamic range as FP32.
  - torch.set_float32_matmul_precision('high') → torch.matmul silently routes FP32 inputs through TF32 Tensor Cores.
  - AMP autocast: GEMM/conv → FP16/BF16 Tensor Cores, accumulation → FP32, sensitive ops (layernorm, softmax) → FP32.
  - BF16 exponent = 8 bits (same as FP32) → no overflow → no GradScaler needed. FP16 exponent = 5 bits → overflow risk → needs GradScaler.
  - Format cheat sheet:
      FP32: 1 sign + 8 exp + 23 mantissa = 4 bytes
      TF32: 1 sign + 8 exp + 10 mantissa = 19 bits (internal only, stored as FP32)
      BF16: 1 sign + 8 exp + 7 mantissa = 2 bytes
      FP16: 1 sign + 5 exp + 10 mantissa = 2 bytes
  - Throughput ladder: FP32 CUDA cores < TF32 Tensor Cores < BF16/FP16 Tensor Cores < FP8 < FP4 (each step ~2× TFLOPS).

- When using reduced precision or 2:4 sparsity, use higher batch sizes to amortize overheads and maximize Tensor Core utilization.

BF16/FP16, FP8, and FP4 Reduced Precision:
- BF16/FP16 on Tensor Cores ≈ 4× FP32 throughput — Tensor Cores issue many FMAs per cycle on half-sized elements.
- FP16 exponent = 5 bits → small gradients underflow to zero → needs loss scaling (static or dynamic GradScaler).
- BF16 exponent = 8 bits (same as FP32) → no underflow → no loss scaling needed → simpler training workflow.
- BF16 preferred for training on modern GPUs — FP32-like stability without GradScaler complexity.
- Loss scaling = multiply loss by a large factor before backward pass → keeps tiny gradients above FP16's underflow threshold → unscale after.

- BF16 has 4x FLOPS/s compared to FP32
Think of the MMA as:  A[M×K] × B[K×N] = C[M×N]

FP32 (TF32):   M=16, K=8,  N=8   → 16×8 × 8×8  = 1024 FMAs
FP16:           M=16, K=16, N=16  → 16×16 × 16×16 = 4096 FMAs
                     ↑       ↑
                   K doubled  N doubled  (2× × 2× = 4×)

                        FP32          FP16
Bytes per element:      4             2          → 2× fewer bytes
FLOPS (same algorithm): same          same       → same useful work
AI:                     F/B           F/(B/2)    → 2× AI

Peak TFLOPS (hardware): 80T           300T+      → ~4× peak throughput

- FP8 = half the bytes of FP16 → 2× weights per HBM transaction → 2-3× BF16/FP16 TFLOPS with FP32/TF32 accumulation.
- NVFP4 = 4-bit format with two-level micro-scaling (per-microblock + higher-level scale) to retain accuracy at extreme compression.
- B200 NVFP4 peak = 10 PFLOPS vs FP32 peak = 80 TFLOPS → ~125× theoretical throughput gain per weight.
- B300 (Ultra) = 15 PFLOPS NVFP4 → 50% more than B200.
- FP4 elements are tiny → 256 KB TMEM fits huge tiles (256×256) → more on-chip reuse → less DRAM traffic.
- Low-precision accumulation happens automatically: kernel reads FP4 from HBM → Tensor Cores do FP4×FP4 → accumulate into FP32/BF16.
- Each precision drop (FP32→FP16→FP8→FP4) roughly doubles ops/byte → doubles AI → pushes kernel from memory-bound → compute-bound.
- Accuracy must be validated per model — NVFP4 needs calibration to confirm precision drop is acceptable.

- Smaller precision = more elements per fixed-width datapath = more FMAs per Tensor Core instruction = higher TFLOPS.
- Smaller precision = fewer bytes per element = more data per HBM transaction = higher AI.
- Accumulation stays in FP32 because summing thousands of small products in low precision loses significant digits.

INT8 Reduced Precision and DP4A Instructions for Inference: 
- INT8 = 1 byte/element vs FP32 = 4 bytes → 75% less memory traffic for weights → 4× more data per HBM transaction.
- DP4A = SIMD dot-product on CUDA cores: 4 INT8 MACs per instruction vs 1 FP32 FMA → 4× throughput even on regular cores.
- INT8 also runs on Tensor Cores via integer MMA instructions → even higher throughput than DP4A on CUDA cores.
- Primarily for inference — training needs higher precision for gradients.
- TMA + TMEM keep INT8 data feeding Tensor Cores without stalls → compute-bound, not memory-bound.

Using CUTLASS for Optimal Arithmetic Intensity and Tensor Core Performance:
- "CUTLASS is a C++ template library that generates high-performance Tensor Core kernels from a configuration — you specify WHAT you want (precision, tile size, pipeline), it builds HOW to do it."
- CUTLASS does automatic optimizations like shared‐memory tiling, asynchronous memory transfers, and double buffering. 
- All you have to do is to make sure your kernel uses CUTLASS optimally
GEMM: General Matrix Multiply

Without cp.async (old way):

DRAM → L2 → L1 → Registers → Registers → Shared Memory
                   ↑            ↑     ↑
              cache hit    load into   store from
                          register    register to SMEM

Thread does: register = global_load(addr);   // data lands in register
             shared_mem[idx] = register;      // thread writes to SMEM

 With cp.async (bypass L1):

DRAM → L2 → ─────────────────────────→ Shared Memory  -- using DMA not TMA
             skips L1, skips registers 

cp.async:  thread issues the copy instruction (still uses thread to set up address/size)
TMA:       dedicated hardware unit does everything — address calc, copy, no thread involvement at all
           DRAM → L2 → Shared Memory (hardware handles it entirely)                        

Method              Who sets up the copy?     Who moves the data?      Thread busy?
─────────────────   ──────────────────────    ─────────────────────    ────────────
Regular load/store  Thread (address + load)   Thread (via registers)   YES — fully blocked
cp.async            Thread (issues instruction) Hardware DMA engine     Partially — thread issues cmd, then free
TMA                 Hardware unit             Hardware unit             NO — thread just triggers it, done

- cp.async - DMA 
- cp.async.bulk.tensor - TMA
- CUTLASS also leverages thread block clusters

Hand-tuned MMA kernel:
  - Written by an expert in raw PTX/CUDA
  - Manually manages TMA descriptors, TMEM allocation, pipeline stages, barrier sync
  - Hundreds of lines of low-level code
  - Squeezes every last drop of performance

CUTLASS GEMM kernel:
  - Generated from C++ templates
  - You configure: tile size, precision, pipeline depth, TMA policy
  - CUTLASS generates the kernel with all the same optimizations
  - Much less code, much easier to maintain

Hand-tuned MMA:   ~98-100% Tensor Core utilization
CUTLASS GEMM:     ~95-98% Tensor Core utilization
                   ↑
                  "a few percent" difference

CUTLAS subset of CUBLAS subset of PyTorch

Agent: Iteratively decide if CUTLASS is sufficient or you generate an optimized MMA kernel yourself.

Inline PTX and SASS Tuning for Microoptimizations: 

- PTX = virtual assembly language (intermediate), SASS = actual native GPU assembly — PTX compiles down to SASS.
- CUDA compiler is already good — PTX/SASS tuning is for squeezing the last few percent in extreme cases.
- Use cases: cache hints, memory fences, special registers (SM ID, lane ID), instruction reordering, instructions with no CUDA C++ intrinsic.
- Inline PTX via asm() in CUDA C++ — mix C++ and assembly, compiler incorporates it into final SASS.
- Rarely needed — only when compiler doesn't generate the optimal instruction sequence you know is better.
- Can also access features not yet exposed in higher-level CUDA API (e.g., new prefetch instructions like cp.async.bulk.prefetch.global).

- volatile keywork -- compiler won't optimize away or reorder accesses to this variable — useful for shared memory flags in producer-consumer patterns.
- Inline PTX example: cp.async.bulk.prefetch.L2.global prefetches data 32 floats ahead into L2 — hides future load latency.
- asm volatile prevents compiler from optimizing away the prefetch instruction.
- Can control cache level: .L2 (prefetch to L2 only) or .L1 (prefetch to L1+L2) — choose based on reuse pattern.
- ld.global.cg = load with "cache global" modifier → cache in L2, bypass L1. ld.global.ca = cache in both L1+L2 (default).
- Inline PTX also useful for manual instruction scheduling — e.g., issue two loads back-to-back then use both results → overlap latencies.
- For modern GPUs, prefer cp.async.bulk.prefetch.tensor.L2 over undocumented built-ins.

- L2 is 500x larger than L1 

CPU (e.g., x86):
  Compiler reorders:     YES (at compile time)
  Hardware reorders:     YES (at runtime, OoO execution engine)
  → CPU has a big reorder buffer, dependency tracking, speculative execution
  → hardware dynamically finds independent instructions and runs them in parallel

GPU:
  Compiler reorders:     YES (at compile time, PTX → SASS)
  Hardware reorders:     NO (in-order execution per warp)
  → GPU issues instructions in the order the compiler arranged them
  → no speculative execution, no reorder buffer

- CPU-GPU superchips (shared memory): use membar.sys / __threadfence_system() + cache operators in PTX for cross-chip coherence — not exposed in high-level CUDA.
- PTX is forward compatible across GPU generations; SASS changes per architecture — write PTX, not SASS.
- asm("mov.u32 %0, %%smid;" : "=r"(smid)) → get SM ID with no C++ intrinsic — useful for debugging and manual SM partitioning in persistent kernels.
- Gains from inline PTX/SASS are incremental (~5-10%) on already-optimized code — e.g., two independent load streams to reduce load-to-use latency bubbles.
- Compute-bound: use fast math instructions (__sinf()) instead of precise ones — PTX lets you force this when compiler is conservative.
- PTX lets you access new hardware features immediately — no need to wait for CUDA C++ intrinsics to catch up.

CPU-GPU race condition: 

Without fence:
  CPU: write data=42    → in CPU cache
  CPU: write flag=1     → in CPU cache
  GPU: read flag        → might see 1 (from memory)
  GPU: read data        → might see OLD value (GPU cache has stale copy) ❌
                           reordering could also mean GPU sees flag=1 before data=42 is visible

With __threadfence_system() / membar.sys:
  CPU: write data=42
  CPU: write flag=1
  CPU: memory fence     → forces writes to be visible SYSTEM-WIDE
  
  GPU: membar.sys       → ensures GPU sees ALL prior CPU writes before proceeding
  GPU: read flag        → sees 1 ✅
  GPU: read data        → sees 42 ✅ (fence guarantees ordering)

- Developer is responsible when: 
- You're on a unified memory superchip (Grace-Hopper, Grace-Blackwell)
- AND you're doing fine-grained CPU↔GPU communication (polling, lock-free queues)
- AND you're bypassing cudaMemcpy / standard sync mechanisms

99% of developers: use cudaMemcpy / streams / PyTorch → hardware handles it.
1% of developers:  write persistent kernels polling CPU flags → need manual fences.  

Newer arch --- SASS will fail, PTX is forward compatible but perf needs tuning on new arch.

Thrust:
  - High-level C++ template library (like STL for GPU)
  - Provides: sort, reduce, scan, transform, copy, unique, etc.
  - You write: thrust::sort(device_vec.begin(), device_vec.end());
  - It handles: kernel launch, block size, memory management
  - Think of it as: "I want to sort on GPU" → Thrust does it optimally

CUB:
  - Lower-level building blocks (block-level and warp-level primitives)
  - Provides: BlockReduce, BlockScan, WarpReduce, DeviceRadixSort, etc.
  - You write: inside YOUR kernel, use CUB primitives for common patterns
  - Think of it as: "I'm writing a kernel and need an efficient reduction inside it"

- So, you can not use L1 cache, use it for data passage from L2 to registers, and also leverage the 256B L2 cache line size to maximize throughput. 
- L1 is both a cache (storage) and a datapath (routing) — data always passes through L1 hardware to reach registers.
- .no_allocate = data passes through L1's datapath but doesn't allocate a cache line → no evictions, hot data stays cached.
- .ca (default) = data passes through AND allocates a cache line → may evict useful data to make room.

L1 is both a CACHE and a DATA PATH:

  ┌─────────────────────────────────┐
  │            L1 unit              │
  │                                 │
  │  ┌───────────────────────┐      │
  │  │  Cache storage        │      │     ← the actual SRAM that holds cached lines
  │  │  [slot][slot][slot]   │      │        (limited capacity, eviction needed)
  │  └───────────┬───────────┘      │
  │              │                  │
  │  ┌───────────┴───────────┐      │
  │  │  Data path / crossbar │──────┼────→ Register File
  │  └───────────────────────┘      │     ← data always flows through here
  │              ↑                  │        regardless of caching policy
  │              │                  │
  └──────────────┼──────────────────┘
                 │
                L2

  .ca (cache all):        data flows through crossbar → register  AND  stored in cache slots
  .no_allocate:           data flows through crossbar → register  BUT  NOT stored in cache slots
                                                                       ↑
                                                              this is the difference!



