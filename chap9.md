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
