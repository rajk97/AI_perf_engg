- Warp can be a reserved one or stalled one or nothing 
- Keeping more warps in flight is known as high occupancy on the SM --> when one warp stalls, another is ready to run --> keeps GPU's compute units busy. 
- Each SM has limited registers/shared memory--> if per-thread requirement of registers is high -- low occupancy 

- Your code complexity → Determines register need
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      High register need                Low register need
      = fewer warps fit                 = more warps fit
      = low occupancy                   = high occupancy
              │                               │
              │                               ▼
              │                         ✅ Good (if no spilling)
              ▼
      ❌ Bad (latency not hidden)
      
      OR if you force low registers on complex code:
      → Spilling → ❌ Bad (memory stalls)

Your code needs:    64 registers per thread
You forced:         32 registers max
Excess:             32 registers worth of data → SPILLED

Spilled where?      "Local memory" (which is actually slow global memory!)

Key insight: A warp is not 32 independent processors. It's one unit that executes one instruction on 32 data elements simultaneously (SIMT = Single Instruction, Multiple Threads).

Warp Scheduler
      │
      ▼
  ┌─────────────────────────────────────────────┐
  │           ONE Instruction Fetch              │
  │           ONE Instruction Decode             │
  │           ONE "Execute this" signal          │
  └─────────────────────────────────────────────┘
      │
      ▼
  ┌──────┬──────┬──────┬──────┬─────────┬──────┐
  │Core 0│Core 1│Core 2│Core 3│   ...   │Core31│  ← 32 CUDA cores
  │ T0   │ T1   │ T2   │ T3   │         │ T31  │  ← 32 threads
  └──────┴──────┴──────┴──────┴─────────┴──────┘
         ALL execute the SAME instruction
         but on DIFFERENT data

There's only ONE instruction decoder per warp — it's cheaper hardware-wise. All 32 threads must do the same thing at the same time.

What Happens With Divergence

if thread_id < 16:
    A()  # Threads 0-15 want this
else:
    B()  # Threads 16-31 want this

Cycle 1-10:  Execute A() → Threads 0-15 active, Threads 16-31 MASKED (do nothing)
Cycle 11-20: Execute B() → Threads 16-31 active, Threads 0-15 MASKED (do nothing) --> Serialization --> Warp Divergence!!! -> Bad for performance 

Choosing Threads per-Block and Blocks-per-Grid Sizes

- Thread block size --> Total no. of threads in the block should be multiple of warp size (32) --> to fully utilize warps
- 

Hardware 32-Thread Warp Size
It's a fixed hardware design choice by NVIDIA. The GPU physically executes threads in groups of exactly 32 — this is baked into the silicon.

FIXED HARDWARE LIMITS
---------------------
Limit                   What It Means                              Example (Ampere)
---------------------------------------------------------------------------------------
Threads per warp        Threads that execute in lockstep           Always 32 (fixed)
Threads per SM          Max threads resident (loaded/ready)        2048
Warps per SM            = Threads per SM / 32                      64 warps
Schedulers per SM       Warps that can issue instructions/cycle    4

- "Blocks are assigned to SM at launch. Block's warps stay resident until block finishes. Then next block loads."

Kernel Launch:
   "Run 1000 blocks, each block = 256 threads (8 warps)"

SM says:
   "I can hold 64 warps max. That's 8 blocks × 8 warps = 64 warps."
   
┌─────────────────────────────────────────────────────────────┐
│ SM                                                          │
│                                                             │
│  Block 0 (8 warps) ──┐                                      │
│  Block 1 (8 warps) ──┼── All 64 warps RESIDENT together    │
│  Block 2 (8 warps) ──┤   (loaded, ready to run)            │
│  ...                 │                                      │
│  Block 7 (8 warps) ──┘                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

When Block 0 FINISHES completely → Block 8 loads in
When Block 1 FINISHES completely → Block 9 loads in
...

Occupancy = min(thread limit, block limit, register limit, shared mem limit). Maximize warps without spilling

- 1 thread block - 1 SM 
- 1 SM has fixed Max_threads_per_SM(2048), Max_warps_per_SM(64), Max_blocks_per_SM(32), Registers_per_SM(65536)
- Thread block size should be multiple of warp size(32)
- It should also be choosen to maximize occupancy without register spilling

-
Fewer registers/thread → More blocks fit → Higher occupancy → BUT may spill
More registers/thread  → Fewer blocks fit → Lower occupancy → BUT no spill

Sweet spot = highest occupancy WITHOUT spilling

Scenario A: Your code genuinely needs few registers

Code is simple → needs 20 registers/thread → No spill ✅
More blocks fit → High occupancy ✅
GOOD!

Scenario B: You FORCE fewer registers than code needs

Code is complex → needs 64 registers/thread
You force: --maxrregcount=32
Compiler: "Fine, I'll spill the extra 32 to local memory" → SPILL ❌
More blocks fit → High occupancy
BUT each thread runs slow due to spilling
BAD!

- How much shared data per SM? 
- 227 KB of shared memory for all resident thread blocks on SM 

B200 Limits(Thread/Block level): 
- Warp size: 32
- Max threads per block: 1024
- Max. warps per thread block: 32

Per-SM limits/SM-resident limits(B200): 

- Max. resident warps per SM - 64
- Max. resident threads per SM - 2048 -> Smaller blocks don't increase peak occupancy — they make occupancy more stable by smoothing out block transitions(like one big block leaving is 1/2 of occupancy gone)
- Max. active blocks per SM - 32

Too big blocks → warp limit. Too small blocks → block limit. Sweet spot = 64-256 threads/block

- CUDA grid limits: 
Maximum blocks(per grid) in X, Y, or Z -> X: 2, 147, 483, 647 || Y & Z: 65,535
Maximum concurrent grids -> 128 grids

- We usually hit thread/block limit first before grid limit

GPU
 └── Multiple Grids (up to 128 concurrent)    ← Multiple kernels running
      └── Each Grid has many Blocks           ← One kernel's work
           └── Each Block has many Threads    ← Assigned to one SM
                └── Threads grouped into Warps (32 each)

┌─────────────────────────────────────────────────────────────┐
│                          GPU                                │
│                                                             │
│  Grid A (Kernel 1)          Grid B (Kernel 2)              │
│  ┌──────────────────┐       ┌──────────────────┐           │
│  │ Block A0  A1  A2 │       │ Block B0  B1  B2 │           │
│  │ Block A3  A4  A5 │       │ Block B3  B4  B5 │           │
│  └──────────────────┘       └──────────────────┘           │
│         │                          │                        │
│         └──────────┬───────────────┘                        │
│                    ▼                                        │
│         SM resources shared across grids                    │
│         (blocks from BOTH kernels on same SM)               │
└─────────────────────────────────────────────────────────────┘

CUDA GPU Backward and Forward Compatibility Model: 
- CUDA is forward compabile if PTX is included in binary -> kernel today will work with tomorrow's arch without any changes

Your CUDA Code (.cu)
        │
        ▼
┌───────────────────┐
│      NVCC         │ (NVIDIA Compiler)
└───────────────────┘
        │
        ├────────────────────┬─────────────────────┐
        ▼                    ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│     PTX       │    │     SASS      │    │    cubin      │
│ (Intermediate)│    │ (Machine Code)│    │  (Binary)     │
└───────────────┘    └───────────────┘    └───────────────┘

Term	What It Is	Analogy
PTX	Portable intermediate code (human-readable assembly-like)	Java bytecode
SASS	Actual GPU machine code for specific architecture	Native x86 binary
cubin	Compiled binary for ONE specific GPU architecture (contains SASS)	.exe for one CPU
fatbin	Bundle containing MULTIPLE cubins + PTX for different architectures	Universal binary (Mac)

PTX (Portable):
   Compile once → GPU driver JIT-compiles to SASS at runtime
   ✅ Works on FUTURE GPUs (forward compatible)
   ❌ Slightly slower first launch (JIT overhead)

SASS (Specific):
   Pre-compiled for exact GPU (e.g., sm_90 = Hopper)
   ✅ Fastest, no JIT needed
   ❌ Won't run on different architecture

fatbin contains:
├── cubin for sm_90 (Hopper)     ← Runs optimized on Hopper
├── cubin for sm_100 (Blackwell) ← Runs optimized on Blackwell  
└── PTX for compute_90           ← JIT-compiles on FUTURE GPUs

Result: Fast on known GPUs, still works on unknown future GPUs.

Format	Speed	Portability
PTX	Slower first run (JIT compiles to SASS)	✅ Works on future GPUs
SASS/cubin	Fast (pre-compiled)	❌ Only works on target architecture
fatbin	Fast + portable	✅ Best of both worlds

CUDA Programming Refresher: 
- Kernel function --> __global__/<<<>>> launch syntax --> need to input blocks_per_grid and threads_per_block
- "CUDA errors are lazy — GPU sets a fault flag, host only sees it on next sync. Always call cudaGetLastError() + cudaDeviceSynchronize() after kernel launch to catch errors early --> Event if one thread in a warp faults, entire warp faults."
- No bounds check? Either it's hidden somewhere, or the dev guarantees N is always a perfect multiple of block size.

Configuring Launch Parameters: Blocks per Grid and Threads per Block 
- 256 threadsperblock is good starting point
- Choose 32 multiples for threadsperblock
-  Latency hiding: Many warps = while one waits for memory, others compute. 64 warps can hide ~400 cycle latency by taking turns
- Occupancy: For GPUs like Blackwell, 256-512 threads/block is good to maximize occupancy while respecting register/shared-memory limits
-  Resource balanced: 256 threads/block = 8 warps = hits warp limit (64) before block limit (32). More warps = better stall hiding. -- too small block size -> hits the block limit of 32 first -- lower occupancy 
- blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock

2D and 3D Kernel Inputs: 

- dim3 type for 2D/3D grids and blocks

Asynchronous Memory ALlocation and Memory Pools: 

Problem: cudaMalloc is slow because it talks to the OS every time.

Solution: Pre-grab a chunk of GPU memory, manage it yourself (or let CUDA manage it).

cudaMalloc = Go to bank every time you need cash

cudaMallocAsync = Withdraw once, keep cash in wallet, reuse

Gold: Use cudaMallocAsync + cudaFreeAsync with memory pools not cudaMalloc/cudaFree

Memory pool = pre-allocated GPU memory chunk. Alloc/free from pool = fast pointer math, no OS calls."

- GPU memory exists, but OS controls access for safety/sharing. Pool = ask OS once, self-manage after

Who OWNS the memory?       → GPU (physically)
Who CONTROLS access?       → OS + GPU Driver
Who ALLOCATES?             → OS + GPU Driver (on your behalf)
Who USES it?               → Your CUDA kernels

- A memory pool recycles freed memory buffers and avoids repeated OS calls to allocate new memory. 
- PyTorch has a custom memory caching allocator, configured with PYTORCH_ALLOC_CONF -- similar to CUDA's memory pool
- Use cudaMallocAsync/cudaFreeAsync with streams — they allocate from a memory pool, respect stream ordering, and avoid expensive global device sync->You CAN use async APIs with default stream (0), but explicit non-blocking streams avoid hidden synchronization
- Use cudaStreamNonBlocking — avoids legacy default-stream barriers; frees only wait for THEIR stream to finish, not the whole GPU.
- Stream = ordered queue of GPU ops. Multiple streams = parallel execution. Async alloc on stream = no global sync, just stream-local ordering
- The memory pool lives in the GPU VRAM. 
- There are commands to set reserve memory, return memory proactively. 

Understanding GPU Memory Hierarchy: 

GPU MEMORY HIERARCHY - BLOCK DIAGRAM
=====================================

                    GPU 0                                         GPU 1
    +-------------------------------+             +-------------------------------+
    |  +--------+       +--------+  |             |  +--------+       +--------+  |
    |  |   SM   |  ...  |   SM   |  |             |  |   SM   |  ...  |   SM   |  |
    |  |+------+|       |+------+|  |             |  |+------+|       |+------+|  |
    |  || SMEM ||       || SMEM ||  |             |  || SMEM ||       || SMEM ||  |
    |  |+------+|       |+------+|  |             |  |+------+|       |+------+|  |
    |  ||  L1  ||       ||  L1  ||  |             |  ||  L1  ||       ||  L1  ||  |
    |  |+------+|       |+------+|  |             |  |+------+|       |+------+|  |
    |  || TMEM ||       || TMEM ||  |             |  || TMEM ||       || TMEM ||  |
    |  |+------+|       |+------+|  |             |  |+------+|       |+------+|  |
    |  +---+----+       +---+----+  |             |  +---+----+       +---+----+  |
    |      |               |        |             |      |               |        |
    |      +-------+-------+        |             |      +-------+-------+        |
    |              |                |             |              |                |
    |        +-----+-----+          |   NVLINK    |        +-----+-----+          |
    |        |    L2     |          |<----------->|         |  L2     |           |
    |        +-----+-----+          |  (fast GPU  |        +-----+-----+          |
    |              |                |   to GPU)   |              |                |
    |        +-----+-----+          |             |        +-----+-----+          |
    |        |   DRAM    |          |             |        |   DRAM    |          |
    |        |  (HBM)    |          |             |        |  (HBM)    |          |
    |        +-----+-----+          |             |        +-----+-----+          |
    +--------------|----------------+             +--------------|----------------+
                   |                                             |
                   |                   PCIe                      |
                   |            (slower, to CPU)                 |
                   +----------------------+----------------------+
                                          |
                              +-----------+-----------+
                              |   CPU Host Memory     |
                              |       (DDR)           |
                              +-----------------------+

- TMEM(for Blackwell GPUs+) -- 256 KB per-SM on-chip memory used by Tensor Core Instructions -> transparently communicates with the Tensor Cores at tens of terabytes/sec of bandwidth.      
- TMEM = dedicated warp-level memory for whole tiles. No fragmentation, no register pressure, Tensor Cores access directly

OLD (Registers):                    NEW (TMEM):
+--------+                          +--------+
| Tensor |                          | Tensor |
| Cores  |                          | Cores  |
+---+----+                          +---+----+
    |                                   |
    v                                   v
+--------+                          +--------+
|Reg T0  | ← A fragment 0           | TMEM   | ← Whole tile A
|Reg T1  | ← A fragment 1           |   A    |   (no fragments!)
|Reg T2  | ← A fragment 2           +--------+
|  ...   |                          | TMEM   | ← Whole tile C
|Reg T31 | ← A fragment 31          |   C    |   (contiguous)
+--------+                          +--------+
  ↑ Complex coordination              ↑ Direct access

- Thread - Sequence of instructions -- on GPU, a tiny worker with its own private scratchpad (registers)

┌─────────────────────────────────────────────────────────┐
│                SM Register File (64K registers)         │
│                                                         │
│  ┌─────────┬─────────┬─────────┬─────────┬──────────┐  │
│  │ Thread 0│ Thread 1│ Thread 2│   ...   │Thread 2047│  │
│  │ R0-R31  │ R0-R31  │ R0-R31  │         │  R0-R31  │  │
│  └─────────┴─────────┴─────────┴─────────┴──────────┘  │
│                                                         │
│  Hardware PARTITIONS the register file among threads    │
└─────────────────────────────────────────────────────────┘
         ↑
    ONE physical SRAM block, logically divided 

MNEMONICS
=========

"64K-255-32"
  - 64K total registers per SM
  - 255 max per thread
  - 32 banks for parallel access

"Spill = Kill"
  - Register spilling kills performance

"Storage ≠ Speed"
  - 256 KB storage, TB/s bandwidth (parallel access)

"Banking = Parking"
  - 32 parking spots (banks), 32 cars (threads)
  - Same spot = wait, different spots = all park at once

GPU REGISTERS - COMPLETE SUMMARY
=================================


1. WHAT ARE REGISTERS?
======================

    NOT this:                           THIS:
    ---------                           -----
    "Each thread has                    "One shared SRAM block,
     its own box"                        partitioned among threads"
    
    [T0] [T1] [T2] [T3]                 +---------------------------+
      |    |    |    |                  |   Register File (SRAM)    |
      v    v    v    v                  |   64K slots × 32 bits     |
    [R] [R] [R] [R]                     |   = 256 KB per SM         |
                                        +-------------+-------------+
                                                      |
                                        Hardware partitions at launch


2. REGISTER FILE STRUCTURE
==========================

    +-----------------------------------------------------------+
    |                 SM Register File (256 KB)                 |
    |                                                           |
    |  64K registers = 65,536 slots of 32 bits each             |
    |                                                           |
    |  Partitioned at kernel launch:                            |
    |  +---------+---------+---------+-----+---------+          |
    |  |Thread 0 |Thread 1 |Thread 2 | ... |Thrd 2047|          |
    |  |Slots    |Slots    |Slots    |     |Slots    |          |
    |  |0-31     |32-63    |64-95    |     |N-N+31   |          |
    |  +---------+---------+---------+-----+---------+          |
    |                                                           |
    |  Example: 32 registers/thread, 2048 threads               |
    |  Check: 32 × 2048 = 65,536 = 64K ✓                        |
    +-----------------------------------------------------------+


3. WHY REGISTERS ARE FAST
=========================

    Memory Type        Location              Latency      
    -------------------------------------------------------
    Registers          Inside SM             ~0 cycles    
    Shared Memory      Inside SM             20-30 cycles 
    L2 Cache           On GPU chip           ~200 cycles  
    HBM (DRAM)         On GPU (off-chip)     400-800 cycles
    
    Registers are WIRED directly to ALUs - no fetch needed!
    
    Instruction: ADD R3, R1, R2
                      |
                      v
    +----------+    +----------+    +----------+
    | Fetch R1 |--->|   ALU    |--->| Store R3 |
    | Fetch R2 |--->| R1 + R2  |    |          |
    +----------+    +----------+    +----------+
                ALL IN ONE CYCLE (pipelined)


4. BANKING FOR PARALLEL ACCESS
==============================

┌─────────────────────────────────────────────────────────┐
│              Register File SRAM (256 KB)                │
│                                                         │
│    All registers physically in same chip area           │
│    No register is "closer" to any thread                │
│                                                         │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐    │
│  │Slot │Slot │Slot │Slot │     │Slot │Slot │Slot │    │
│  │  0  │  1  │  2  │  3  │ ... │65533│65534│65535│    │
│  └──┬──┴──┬──┴──┬──┴──┬──┴─────┴──┬──┴──┬──┴──┬──┘    │
│     │     │     │     │           │     │     │        │
└─────┼─────┼─────┼─────┼───────────┼─────┼─────┼────────┘
      │     │     │     │           │     │     │
      ▼     ▼     ▼     ▼           ▼     ▼     ▼
   ┌─────────────────────────────────────────────────┐
   │              Crossbar / Banking Logic           │
   │  (Routes each thread to its assigned slots)     │
   └─────────────────────────────────────────────────┘
      │     │     │           │     │     │
      ▼     ▼     ▼           ▼     ▼     ▼
   Thread Thread Thread ... Thread Thread Thread
     0      1      2         2045   2046   2047

    Problem: 32 threads need registers simultaneously
    Solution: 32 banks with independent access ports
    
    +-------+-------+-------+-------+-----+-------+
    |Bank 0 |Bank 1 |Bank 2 |Bank 3 | ... |Bank 31|
    +-------+-------+-------+-------+-----+-------+
    |Slot 0 |Slot 1 |Slot 2 |Slot 3 |     |Slot 31|
    |Slot 32|Slot 33|Slot 34|Slot 35|     |Slot 63|
    |Slot 64|Slot 65|Slot 66|Slot 67|     |Slot 95|
    |  ...  |  ...  |  ...  |  ...  |     |  ...  |
    +---+---+---+---+---+---+---+---+-----+---+---+
        |       |       |       |             |
        v       v       v       v             v
      Port 0  Port 1  Port 2  Port 3  ...  Port 31
    
    Bank number = Slot number % 32
    
    32 threads accessing 32 different banks = 1 cycle (parallel)
    32 threads accessing same bank = 32 cycles (serialized) BAD!


5. REGISTER LIMITS
==================

    Hardware Limits:
    +--------------------------------+----------------+
    | Limit                          | Value          |
    +--------------------------------+----------------+
    | Max registers per thread       | 255            |
    | Total registers per SM         | 64K (65,536)   |
    | Register size                  | 32 bits        |
    | Total register file size       | 256 KB         |
    +--------------------------------+----------------+
    
    Why 255 max?
    - Instruction encoding uses 8 bits for register address
    - 2^8 = 256, minus 1 reserved = 255


6. REGISTERS VS OCCUPANCY TRADEOFF
==================================

    More registers/thread = Fewer threads fit = Lower occupancy
    
    Registers/Thread    Threads on SM    Warps    Occupancy
    ----------------------------------------------------------------
    32                  2048             64       100%
    64                  1024             32       50%
    128                 512              16       25%
    255                 257              8        12.5%
    
    Formula: Max threads = 65,536 ÷ registers_per_thread


7. REGISTER SPILLING
====================

    When thread needs more registers than allocated:
    
    Code needs: 64 registers
    You force:  --maxrregcount=32
    
    +------------+          +----------------+
    | Registers  |  SPILL   | Local Memory   |
    | (32 slots) | -------> | (HBM - SLOW!)  |
    +------------+          +----------------+
         |                        |
         | ~0 cycles              | 400-800 cycles
         v                        v
       FAST                     SLOW!
    
    Spilling = Registers overflow to slow memory = BAD


8. BANDWIDTH CALCULATION
========================

    Storage ≠ Bandwidth!
    
    Storage:   64K registers = 256 KB (how much fits)
    Bandwidth: ~TB/s (how fast we access)
    
    Per cycle:
      4 schedulers × 1 instruction each
      Each instruction: ~3 operands (2 read + 1 write)
      Per warp: 32 threads
      Each operand: 4 bytes
    
      = 4 × 3 × 32 × 4 bytes = 1.5 KB per cycle
    
    At 2 GHz:
      1.5 KB × 2 billion = 3 TB/s per SM
    
    All 2048 threads touch registers simultaneously = HIGH bandwidth


9. KEY MENTAL MODELS
====================

    Thread:
      "A sequence of instructions with an ASSIGNED partition 
       in the shared register file"
    
    Register file:
      "One big SRAM with 32 doors (banks). 
       Each thread knows which slots are theirs."
    
    Zero latency:
      "Registers are flip-flops wired to ALUs. 
       No memory fetch - values already there."
    
    Banking:
      "32 parallel access ports. 32 threads, 32 doors = 
       all enter at once. Same door = wait in line."
    
    Spilling:
      "Overflow to slow memory. Every spill = 
       hundreds of wasted cycles."


10. DECISION FLOWCHART
======================

    START: How many registers does my kernel need?
                          |
                          v
            +-------------+-------------+
            | Compile with -v flag      |
            | nvcc --ptxas-options=-v   |
            +-------------+-------------+
                          |
                          v
                   Registers/thread
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
        <32             32-64           >64
          |               |               |
          v               v               v
      High             Good            Low
      occupancy        balance         occupancy
          |               |               |
          v               v               v
      Maybe too        Usually         Consider:
      simple?          optimal         - Simplify code
                                       - Accept low occupancy
                                       - Check for spilling


MNEMONICS
=========

"64K-255-32"
  - 64K total registers per SM
  - 255 max per thread
  - 32 banks for parallel access

"Spill = Kill"
  - Register spilling kills performance

"Storage ≠ Speed"
  - 256 KB storage, TB/s bandwidth (parallel access)

"Banking = Parking"
  - 32 parking spots (banks), 32 cars (threads)
  - Same spot = wait, different spots = all park at once

- For scheduling/execution - we talk warps 
- For resource allocation - we talk blocks 

- Shared memory is per-block 

┌─────────────────────────────────────────────────────────────────┐
│                             SM                                  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Shared Memory Pool (256 KB total)              │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │  │
│  │  │  Block 0    │ │  Block 1    │ │  Block 2    │  ...   │  │
│  │  │  SMEM: 64KB │ │  SMEM: 64KB │ │  SMEM: 64KB │        │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Block 0:                Block 1:              Block 2:        │
│  ┌──────────────┐        ┌──────────────┐     ┌──────────────┐ │
│  │Warp 0 (32 T) │        │Warp 0 (32 T) │     │Warp 0 (32 T) │ │
│  │Warp 1 (32 T) │        │Warp 1 (32 T) │     │Warp 1 (32 T) │ │
│  │Warp 2 (32 T) │        │Warp 2 (32 T) │     │Warp 2 (32 T) │ │
│  │    ...       │        │    ...       │     │    ...       │ │
│  └──────────────┘        └──────────────┘     └──────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         SM (Blackwell)                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Register File: 256 KB                       │   │
│  │              (64K × 32-bit registers)                    │   │
│  │              FIXED - cannot be resized                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         Unified L1 + Shared Memory: 256 KB               │   │
│  │         CONFIGURABLE split (carveout)                    │   │
│  │                                                          │   │
│  │    Option A: 128 KB L1  +  128 KB Shared                 │   │
│  │    Option B: 64 KB L1   +  192 KB Shared                 │   │
│  │    Option C: 32 KB L1   +  228 KB Shared (max SMEM)      │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Total on-SM SRAM: 256 KB (regs) + 256 KB (L1/SMEM) = 512 KB   │
└─────────────────────────────────────────────────────────────────┘

Every SM on the GPU has IDENTICAL allocation.

- For joint shared + L1 cache, max available - 256 KB per SM
- Max. shared memory allocatable - 227 KB per SM 
- Maximum shared memory per block - 227 KB per block 

- Access cost = 20-30 cycles 

SM's Shared Memory Pool (228 KB)
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Block 0    │  │   Block 1    │  │   Block 2    │          │
│  │   SMEM       │  │   SMEM       │  │   SMEM       │          │
│  │   64 KB      │  │   64 KB      │  │   64 KB      │          │
│  │              │  │              │  │              │          │
│  │  PRIVATE!    │  │  PRIVATE!    │  │  PRIVATE!    │          │
│  │  Can't see   │  │  Can't see   │  │  Can't see   │          │
│  │  other       │  │  other       │  │  other       │          │
│  │  blocks      │  │  blocks      │  │  blocks      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  Total used: 192 KB    Remaining: 36 KB                        │
└─────────────────────────────────────────────────────────────────┘

Block 0's threads: Can read/write Block 0's SMEM ✅
Block 0's threads: CANNOT access Block 1's SMEM ❌

Thread Block Clusters (Hopper+)
  - Distributed Shared Memory (DSMEM)
  - Blocks in same cluster CAN share
  - But cluster is limited (~16 blocks max)

Throughput TB/s per SM(bank-conflict free--> serialized access)  

TMEM: 
- Again 256 KB per-SM on-chip memory used by Tensor Core Instructions -> transparently communicates with the Tensor Cores at tens of terabytes/sec of bandwidth.
1. PURPOSE:    Feeds Tensor Cores (matrix multiply)
2. SIZE:       256 KB per SM
3. ACCESS:     NOT pointer-addressable (can't do TMEM[i])
4. MANAGEMENT: TMA handles data movement automatically
5. CONTENTS:   Accumulator (C matrix) lives here

How Data Flows for C = A × B

Global Memory (HBM)
       │
       │ TMA (automatic)
       ▼
Shared Memory (SMEM)
       │
       │ TMA (automatic)
       ▼
Tensor Memory (TMEM)
       │ No TMA - direct hardware connection 
       ▼
 Tensor Cores
   compute
    A × B
       │
       ▼
Result in TMEM (accumulator)

- TMEM exists because Tensor Cores are TOO FAST to share registers with CUDA cores. Dedicated memory = no contention = full throughput.

2/6/26:

Constant Memory Cache: 

1. __constant__ space (GLOBAL):
   - 64 KB total for entire GPU
   - Lives in global memory (HBM)
   - Read-only during kernel execution
   - Shared by ALL SMs

2. Constant cache (PER-SM):
   - 8 KB per SM
   - Fast on-chip SRAM
   - Caches reads from __constant__ space
   - Each SM has its own copy

WHAT:     
  - __constant__: 64 KB global read-only memory
  - Constant cache: 8 KB per-SM cache (fronts __constant__)

BEST FOR: Small lookup tables shared by all threads
FAST:     Cache hit + all 32 threads read SAME address (broadcast)
SLOW:     Cache miss OR threads read DIFFERENT addresses (serialize)

BENEFIT:  Cache avoids repeated global memory traffic

L2 Cache: 

- 126 MB GPU-wide buffer that glues all SMs to off-chip HBM3e -> latency 200 cycles --> bandwidth 10s of TB/s. 
- Structure your global loads into 128-byte aligned, coalesced segments that map cleanly to caache lines. This avoids split transactions and maximizes use of L2 and DRAM bandwidth. 

Global Memory(HBM): 

1. LOCATION:    Off-chip (not on SM)

2. SIZE:        180 GB (Blackwell)

3. BANDWIDTH:   ~8 TB/s (fast throughput)

4. LATENCY:     400-1000+ cycles (slow access)

5. WHAT LIVES HERE:
   - Global memory (your tensors, weights, etc.)
   - Local memory (spills, oversized arrays)
   - Constant memory backing store

6. SPILL PENALTY:
   - Registers overflow → "local memory" → actually HBM
   - Same slow latency as global memory
   - "Local" = private scope, NOT local location

7. KEY INSIGHT:
   - High bandwidth ≠ low latency
   - Wide highway, but far away

- NSight Compute for kernel tuning. 
- Die = a single piece of silicon chip cut from a wafer.
- Wafer (circular silicon disk, ~300mm)
┌─────────────────────────────────────┐
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐    │
│  │Die│ │Die│ │Die│ │Die│ │Die│    │
│  └───┘ └───┘ └───┘ └───┘ └───┘    │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐    │
│  │Die│ │Die│ │Die│ │Die│ │Die│    │
│  └───┘ └───┘ └───┘ └───┘ └───┘    │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐    │
│  │Die│ │Die│ │Die│ │Die│ │Die│    │
│  └───┘ └───┘ └───┘ └───┘ └───┘    │
└─────────────────────────────────────┘

Each die = one chip (e.g., one GPU, one CPU)
Wafer is cut up → individual dies → packaged

┌─────────────────────────────────────────────────────────────────┐
│                     B200 "Single GPU"                          │
│                                                                 │
│   ┌─────────────────┐   10 TB/s   ┌─────────────────┐          │
│   │     Die 0       │◄──────────►│     Die 1       │          │
│   │   (half GPU)    │  chip-to-  │   (half GPU)    │          │
│   │                 │   chip     │                 │          │
│   │  80 SMs         │ interconn  │  80 SMs         │          │
│   └────────┬────────┘            └────────┬────────┘          │
│            │                              │                    │
│     ┌──────┴──────┐                ┌──────┴──────┐             │
│     │ 4× HBM3e    │                │ 4× HBM3e    │             │
│     │ stacks      │                │ stacks      │             │
│     └─────────────┘                └─────────────┘             │
│                                                                 │
│   Total: 160 SMs, 8 HBM stacks, appears as ONE GPU             │
└─────────────────────────────────────────────────────────────────┘

- Point of coherency: PoC = where all relevant threads 'agree' on data. Narrower scope = faster sync. Choose smallest scope that covers your communication needs

Level               PoC At                Who Sees the Update?
------------------------------------------------------------------------
Thread              Registers             Just this thread
Thread-block (CTA)  Shared Memory         All threads in same block
Cluster             Distributed SMEM      All blocks in same cluster
Device              L2 Cache / HBM        All threads on GPU
System              System memory         All GPUs + CPU

Unified Memory: 
- Unified Memory = single memory address space shared by CPU + GPU
- How It Works

cudaMallocManaged(&ptr, size);

         CPU                              GPU
    ┌──────────┐                     ┌──────────┐
    │  Page A  │ ───── migrate ────► │  Page A  │
    │  Page B  │ ◄─── on-demand ──── │  Page B  │
    └──────────┘                     └──────────┘
                     │
                     ▼
              Page-fault triggers
              automatic migration

GPU kernel runs
       │
       ▼
Access data at ptr[1000]
       │
       ▼
┌──────────────────────────────┐
│  Page fault!                 │
│  Data is on CPU, not GPU     │
│                              │
│  GPU STALLS while waiting    │
│  for page migration...       │
│                              │
│  (could be ms of latency)    │
└──────────────────────────────┘
       │
       ▼
Data arrives, kernel continues

Migration Speed by Hardware
Hardware                  CPU ↔ GPU Bandwidth    Page Fault Pain
------------------------------------------------------------------------
PCIe Gen4                 ~32 GB/s               HIGH (slow migration)
Early NVLink              ~300 GB/s              MEDIUM
Grace Hopper (NVLink-C2C) ~900 GB/s              LOW (but still >0)

Solutions: Avoid Surprise Stalls

SOLUTION 1: Prefetch (move data BEFORE kernel)

  cudaMemPrefetchAsync(ptr, size, gpuId, stream);
  myKernel<<<...>>>();
  
  Timeline:
  ─────────────────────────────────────────────►
  [Prefetch data async] [Kernel runs - no stalls!]
        └── overlapped with other work


SOLUTION 2: Memory Advice (tell driver where data lives)

  cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, gpuId);
    └── "This data mostly used on GPU"
  
  cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, gpuId);
    └── "This data is read-only, can duplicate"

"Pages = coarse-grained (KB-MB), where data LIVES, OS-managed, migration is slow. Caches = fine-grained (bytes), hardware-managed, speeds up ACCESS. Both work together."

Streams = async queues. Use multiple streams to overlap transfers + compute, pipeline batches, and keep both CPU and GPU busy. Streams are a different level of parallelism from the thread/warp/block parallelism.

LEVEL 1: Thread Parallelism (INSIDE a kernel)
  └─ Thousands of threads run simultaneously
  └─ This is what makes GPU fast for matrix math

LEVEL 2: Stream Parallelism (BETWEEN operations)
  └─ Multiple kernels/transfers run simultaneously
  └─ This keeps GPU busy (no idle time between ops)

LEVEL 3: Multi-GPU Parallelism (ACROSS devices)
  └─ Multiple GPUs work simultaneously
  └─ Scales beyond single GPU

WITHOUT streams (serial operations):
───────────────────────────────────────────────────────────────────►
CPU:  [Launch]           [Launch]           [Launch]
GPU:         [==Kernel A==]     [==Kernel B==]     [==Kernel C==]
                         ↑                  ↑
                      IDLE              IDLE (wasted!)


WITH streams (parallel operations):
───────────────────────────────────────────────────────────────────►
CPU:  [Launch A,B,C quickly]  [Do other work...]
GPU Stream 1: [==Kernel A==][==Kernel D==]
GPU Stream 2:    [==Kernel B==][==Kernel E==]
GPU Stream 3:       [==Kernel C==][==Kernel F==]
              └── OVERLAPPED! No idle time ──┘  

- SetAccessedBy = create mapping so another GPU can access this memory remotely without triggering migration. Avoids page fault stalls at kernel launch
- PROBLEM:
  Stream 1 has a page fault → waits on migration
  BUT this can cause OTHER streams to block too (hidden sync)!

SOLUTION:
  cudaStreamAttachMemAsync(stream1, ptr, 0, cudaMemAttachSingle)
  
  Now: Only Stream 1 can trigger faults on ptr
       Other streams won't touch ptr → won't block
       
  = Isolation / separation of concerns

cudaMemcpy (manual):
  ✅ FAST (explicit, predictable, optimized)
  ❌ Annoying (you manage two pointers, copy calls everywhere)

Unified Memory (naive):
  ✅ EASY (one pointer, no memcpy calls)
  ❌ SLOW (surprise page faults, stalls)

Unified Memory (with prefetch/advise/stream attachment):
  ✅ EASY-ish (one pointer, few hints)
  ✅ FAST (close to cudaMemcpy performance)

Maintaining High Occupancy and GPU Utilization: 
- When one warp stalls waiting for data, another warp can run. 
- Occupancy = active warps / max warps (per SM) 

A warp stalled on memory is:
  ✅ Resident (occupies registers, scheduler slot)
  ✅ Active (still "in flight", eligible when data arrives)
  ❌ NOT currently executing (waiting for data)
  
"Active" doesn't mean "executing right now"
"Active" means "loaded on SM, participating in scheduling"

State           Resident?   Active?   Executing?   Description
------------------------------------------------------------------------
Not assigned    No          No        No           Waiting in grid queue
Resident        Yes         Yes       Maybe        On SM, ready or stalled
  └─ Executing  Yes         Yes       Yes          Currently running
  └─ Stalled    Yes         Yes       No           Waiting (memory, sync)
Finished        No          No        No           Done, exited SM

- Fundamental rule of CUDA optimization: Launch enough parallel work to fully occupy the GPU. 

- High register usage reduces occupancy. Spilling is the compiler's tradeoff to maintain occupancy at the cost of performance.

- Use vectorized operations like C = A + B instead of for i in ... C[i] = A[i] + B[i] in pytorch -- will use parallel capabilities in the optimal sense. 

1/7/26:
- The fundamental feature of a GPU is to interleave memory loads and computations from different warps in parallel --> hides memory latency across the warps. 
- 100% occupancy can still be slow if memory-bound 
- GPU FLOPS growing faster than memory bandwidth -> optimizing memory movement is absolutely critical. 

Tuning Occupancy with Launch Bounds: 
- __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) --> gives compiler hints to optimize register usage/inlining functions 
- Max threads per block: 1024
- Max resident warps per SM: 64 warps = 2048 threads 
- __launch_bounds__ -- reduces/restricts register usage to avoid spilling, maximize warp throuphput, basically sacrific per-thread performance for overall occupancy and throughput. 
- __launch_bounds__ is a compile time optimization 
- cudaOccupancyMaxPotentialBlockSize -- runtime API to find optimal block size for occupancy considering register and shared memory usage. 

┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPILE TIME vs RUNTIME                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  __launch_bounds__(maxThreads, minBlocks)                                   │
│  ─────────────────────────────────────────                                  │
│  WHEN:    Compile time (baked into binary)                                  │
│  WHAT:    Hints to NVCC compiler                                            │
│  EFFECT:  Compiler adjusts register allocation, inlining, unrolling         │
│  YOU:     Must guess good values beforehand                                 │
│                                                                             │
│  cudaOccupancyMaxPotentialBlockSize()                                       │
│  ────────────────────────────────────                                       │
│  WHEN:    Runtime (queries actual GPU)                                      │
│  WHAT:    API asks "what block size maximizes occupancy for THIS kernel     │
│           on THIS GPU?"                                                     │
│  EFFECT:  Returns optimal block size; you use it in <<<grid, block>>>       │
│  YOU:     Kernel already compiled; this just picks best launch config       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

- __launch_bounds__ → Changes how the kernel is compiled (register usage)
- cudaOccupancyMaxPotentialBlockSize → Changes how you launch an already-compiled kernel

- int gridSize = std::max(minGridSize, (N + blockSize - 1) / blockSize); -- minimum grid size to saturate GPU SM's. 
- Validate Occupancy API suggestions by timing kernels at ±1–2
candidate block sizes. Register pressure and L2 behavior on
modern GPUs can actually make a slightly sub-maximal occupancy
configuration faster in practice.
- Nsight Compute’s “Registers Per Thread”
and “Occupancy” metrics can guide optimize for registers per thread vs. occupancy tradeoff. 
- basically higher no. of registers per thread -- low no. of resident warps because there are fixed no. of registers on the SM
- low no. of registers -- usually good occupancy
- too low no. of registers assigned than necessary by the thread -- register spilling into slower global HBM3e memory 
  -- occupancy is high but speed is slow

Debugging Functional Correctness with NVIDIA Compute Sanitizer: 
- Catch subtle memory bugs and race conditions. 

NVIDIA COMPUTE SANITIZER - QUICK REFERENCE
==========================================

WHAT: Runtime bug-finder for CUDA (like Valgrind for GPU)
CLI:  compute-sanitizer [--tool toolname] [options] <app>

4 TOOLS - "MRIS" (Memory, Race, Init, Sync)
───────────────────────────────────────────

1. MEMCHECK  → Out-of-bounds & misaligned memory access
              → GPU hardware exceptions
              → Device memory leaks
              
2. RACECHECK → Shared memory data races (WAW, WAR, RAW)
              → Thread-to-thread hazards within blocks
              
3. INITCHECK → Uninitialized device memory reads
              → Missing H2D copies, skipped writes
              
4. SYNCCHECK → Bad synchronization primitives
              → Deadlocks, barrier mismatches

KEY FLAGS:
──────────
--error-exitcode    → Fail CI on errors
--kernel-name       → Filter specific kernels
--nvtx yes          → Use NVTX annotations for scope

BEST PRACTICE:
──────────────
"Integrate into CI with --error-exitcode + NVTX annotations"

MNEMONIC:
─────────
"MRIS catches Memory, Races, Inits, Syncs"

Roofline Model: Compute-Bound or Memory-Bound Workloads: 
- Tells if a model is compute/memory bound 
- Arithmetic intensity is measures as the number of FLOPS performed per byte transferred between off-chip global memory and the GPU. 
- For Blackwell, peak compute performance = 80 TFLOPs/sec, peak memory bandwidth = 8 TB/s

BLACKWELL ROOFLINE RULE OF THUMB
================================

Ridge Point = Peak Compute / Peak Bandwidth
           = 80 TFLOPs / 8 TB/s
           = 10 FLOPs/byte

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   AI < 10  →  MEMORY BOUND  (bandwidth is bottleneck)       │
│                                                             │
│   AI > 10  →  COMPUTE BOUND (ALUs are bottleneck)           │
│                                                             │
│   AI = 10  →  BALANCED (ridge point, ideal efficiency)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

- How do you reduce memory boundness?
1. Reduce precision - FP32-FP16-FP8(Modern GPUs) - FP4(Some Blackwell loads)
2. Models stored compressed in HBM, even beyond FP4 compression, and the hardware can decompress the weights on the fly. 
- Weights are stored in a compressed 4-bit/2-bit scheme and decompressed by hardware at load time and cast to FP16/FP32 for higher-precision aggregations and computations. 

- Main goal is to push from memory bound to compute bound and then use GPU's full floating-point horsepower. 
- LLM's - both compute and memory bound -- attnetion layers(prefill phase) are compute bound while mat-muls(decode phase) are memory bound.

NSIGHT TOOLS - QUICK REFERENCE
==============================

TWO TOOLS, TWO QUESTIONS:
─────────────────────────
Nsight Compute  →  "WHY is it slow?"  (per-kernel deep dive)
Nsight Systems  →  "WHEN is it slow?" (timeline, big picture)

MEMORY-BOUND SYMPTOMS (in Nsight Compute):
──────────────────────────────────────────
• High DRAM bandwidth utilization
• Low ALU utilization  
• Warps stalled on memory

NSIGHT COMPUTE SHOWS:
─────────────────────
• Per-kernel counters
• Latencies & cache hit rates
• Warp issue stalls
• Register pressure
• Launch config effects

NSIGHT SYSTEMS SHOWS:
─────────────────────
• Timeline view (GPU idle gaps)
• CPU-GPU overlap
• PCIe/NVLink transfers
• Stream concurrency

NVTX = YOUR LABELS:
───────────────────
• Mark regions: "memory copy", "kernel execution"
• Shows in timeline → verify overlap
• Essential for debugging async operations

WORKFLOW:
─────────
"Profile → NVTX label suspects → Zoom timeline → Optimize → Repeat"

MNEMONIC:
─────────
"Compute = WHY, Systems = WHEN, NVTX = WHERE"

NVTX: Annotation API for Profiling and Optimization: Just like tracy zones in tracy profiler 

Without using pinned(page-locked) memory, the cudaMemcpyAsync transfer cannot overlap with kernel execution. 

PAGEABLE MEMORY (regular malloc):
─────────────────────────────────

CPU calls cudaMemcpyAsync(dst, src, size, stream)
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  CUDA Driver: "src is pageable... I can't trust it to stay put" │
│                                                                 │
│  Step 1: Allocate temporary PINNED staging buffer               │
│  Step 2: Copy src → staging buffer (CPU memcpy, SYNCHRONOUS!)   │
│  Step 3: DMA from staging buffer → GPU                          │
│                                                                 │
│  The CPU memcpy in Step 2 BLOCKS until complete!                │
│  → "Async" in name only — actually synchronous                  │
└─────────────────────────────────────────────────────────────────┘

Timeline:
─────────────────────────────────────────────────────────►
CPU:    [cudaMemcpyAsync BLOCKS here...] [launch kernel]
GPU:                                     [===kernel===]
                    ↑
            NO OVERLAP! ❌


PINNED MEMORY (cudaHostAlloc / cudaMallocHost):
───────────────────────────────────────────────

CPU calls cudaMemcpyAsync(dst, pinned_src, size, stream)
                │
                ▼
┌─────────────────────────────────────────────────────────────────┐
│  CUDA Driver: "pinned_src is page-locked, address is stable!"   │
│                                                                 │
│  Step 1: Queue DMA transfer (returns immediately!)              │
│  Step 2: GPU DMA engine handles transfer in background          │
│                                                                 │
│  CPU is FREE to do other work immediately                       │
└─────────────────────────────────────────────────────────────────┘

Timeline:
─────────────────────────────────────────────────────────►
CPU:    [cudaMemcpyAsync returns immediately][launch kernel][other work]
GPU:    [====DMA Transfer====][===kernel===]
        └── OVERLAPPED! ✅ ──┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY-BOUND DEBUGGING DECISION TREE                     │
└─────────────────────────────────────────────────────────────────────────────┘

                        START: Kernel seems slow
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Run Nsight Systems first    │
                    │  (big picture timeline)      │
                    └──────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
           Idle gaps between              No idle gaps
           kernel launches?               (back-to-back kernels)
                    │                             │
                    ▼                             ▼
        ┌─────────────────────┐        ┌─────────────────────┐
        │ GPU waiting for     │        │ Kernel itself is    │
        │ data transfers      │        │ slow → drill deeper │
        │                     │        │                     │
        │ FIX:                │        │ Run Nsight Compute  │
        │ • Async transfers   │        │ (per-kernel metrics)│
        │ • Pinned memory     │        └──────────┬──────────┘
        │ • Overlap H2D/D2H   │                   │
        │   with compute      │                   ▼
        └─────────────────────┘        ┌─────────────────────┐
                                       │ Check: Global Load  │
                                       │ Efficiency          │
                                       └──────────┬──────────┘
                                                  │
                            ┌─────────────────────┴─────────────────────┐
                            ▼                                           ▼
                    LOW (<50%)                                    HIGH (>80%)
                            │                                           │
                            ▼                                           ▼
               ┌─────────────────────────┐                 ┌─────────────────────┐
               │ Memory access pattern   │                 │ Check: Memory Pipe  │
               │ is inefficient          │                 │ Utilization         │
               │                         │                 └──────────┬──────────┘
               │ FIX:                    │                            │
               │ • Coalesce accesses     │              ┌─────────────┴─────────────┐
               │ • Align to 128 bytes    │              ▼                           ▼
               │ • Avoid strided access  │         LOW (<50%)                  HIGH (>80%)
               │ • Use shared memory     │              │                           │
               └─────────────────────────┘              ▼                           ▼
                                            ┌──────────────────┐       ┌──────────────────┐
                                            │ Not enough       │       │ Saturating       │
                                            │ memory traffic   │       │ bandwidth! ✅    │
                                            │                  │       │                  │
                                            │ Check occupancy  │       │ You're doing     │
                                            │ & warp stalls    │       │ well on memory   │
                                            └──────────────────┘       │                  │
                                                                       │ If still slow:   │
                                                                       │ → COMPUTE bound  │
                                                                       │ → Check ALU util │
                                                                       └──────────────────┘        

│  • Global Load Efficiency (%)                                               │
│    ─────────────────────────                                                │
│    = (Requested bytes) / (Actually transferred bytes) × 100                 │
│                                                                             │
│    LOW (<50%):  Wasting bandwidth! Uncoalesced or misaligned access         │
│    HIGH (>80%): Good! Most transferred bytes are actually used              │
│                                                                             │
│    Example:                                                                 │
│      You need 4 bytes, but GPU fetches 128-byte cache line                  │
│      Efficiency = 4/128 = 3.1% → Very bad!                                  │


│  • Memory Pipe Utilization (%)                                              │
│    ───────────────────────────                                              │
│    = How busy is the memory subsystem?                                      │
│                                                                             │
│    Interpret WITH ALU Utilization (2×2 grid):                               │
│                                                                             │
│    HIGH Mem + LOW ALU:  Memory-bound by design (low AI) — not a bug ⚠️      │
│    HIGH Mem + HIGH ALU: Optimal! GPU fully utilized 🎯                      │
│    LOW Mem + HIGH ALU:  Compute-bound by design (high AI) — good ✅         │
│    LOW Mem + LOW ALU:   PROBLEM! Wasting GPU — check occupancy/access 🔴       │

- For Blackwell: 
    - Max registers per thread = 255
    - Max per-SM shared memory = 227 KB
    - Max resident warps = 64
    - Max resident thread blocks = 32

Chapter 6 done!    
