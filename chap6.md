GPU Architecture, CUDA Programming, and Maximizing Occupancy:
- Main goal: Review Single Instruction Multiple-Threads SIMT execution model and how warps, thread blocks, and grids map your GPU-based algorithms onto streaming multiprocessors(SMs).
- CUDA programming patterns, on-chip memory hierarchy, and demonstrate the GPUs asynchronous data transfer capabilities, including the Tensor Memory Accelerator(TMA) and the Tensor Mmeory(TMEM) that serves as teh accumulatoor for the Tensor Core operations. 
- CPUs: Optimize for single-threaded low-latency performance, 
- GPUs: Throughput-optimized processors built to run thousands of threads in parallel. 

- Streaming MultiProcessors: Analogous to CPU cores but streamlined for parallelism. 
- Each SM can track up to 64 warps(32 thread grouops) on Blackwell. 

┌─────────────────────────────────────────────────────────┐
│                 ONE SM (Blackwell)                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   64 warps × 32 threads = 2048 threads                 │
│   ──────────────────────                                │
│                                                         │
│   64K registers (256 KB)                                │
│   ────────────────────────                              │
│                                                         │
│   256 KB shared SRAM (L1 + shared memory)              │
│   ───────────────────────────────────────               │
│        └── 227 KB usable as shared memory              │
│                                                         │
└─────────────────────────────────────────────────────────┘

256 KB SRAM per SM
┌─────────────────────────────────────────┐
│                                         │
│  ┌─────────────────────────────────┐   │
│  │     Shared Memory(All threads in a thread block access it -> 20x faster than HBM memory)              │   │
│  │     (up to 227 KB usable)       │   │  ← You control this
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │     L1 Cache                    │   │  ← Hardware manages
│  │     (remaining portion)         │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────┐                               │
│  │ 1KB │ Reserved by CUDA              │
│  └─────┘                               │
│                                         │
└─────────────────────────────────────────┘

Per cycle, SM can issue:

4 schedulers × 1 warp each × 2 instructions = 8 instructions max
     │              │              │
     │              │              └── NOT "memory ops"
     │              │                  Could be: 1 math + 1 memory
     │              │                           1 math + 1 math (diff type)
     │              │
     │              └── Each scheduler picks 1 warp (32 threads)
     │
     └── 4 independent schedulers

So: 4 warps active × 32 threads = 128 threads issuing per cycle
    (out of 2048 resident threads)

Each SM has 4 warp schedulers --> That means in each cycle, 4 warps can be executed. But the warp execution can contain 2 different instructions like 1 math + 1 memory(dual-issue - must be from the same warp), etc.
- If all the warps correspond to a single kernel, why do we have different instructions at all among threads from different warps? 
- Coz of latency, the warps fall out of sync. 

- Blackwell SMs contain four independent warp schedulers, each capable of issuing one warp instruction per cycle with dual-issue of one math and one memory per scheduler. 
- Issue: Dispatch an instruction to the hardware that executes it 

"Issue" = Dispatch an instruction to the hardware that executes it

┌─────────────────┐         ┌─────────────────┐
│  Warp Scheduler │  ISSUE  │ Execution Unit  │
│                 │ ──────► │ (does the work) │
│  "Run this ADD" │         │                 │
└─────────────────┘         └─────────────────┘
                               ALU, Tensor Core,
                               Load/Store Unit, etc.

INSTRUCTION LIFECYCLE:
══════════════════════

1. FETCH     Get instruction from memory
                │
                ▼
2. DECODE    Figure out what it means (ADD? LOAD? MUL?)
                │
                ▼
3. ISSUE  ►  Send to the right execution unit    ◄── THIS IS "ISSUE"
                │
                ▼
4. EXECUTE   Hardware does the actual math/memory op
                │
                ▼
5. WRITEBACK Store the result

Scheduler's job:
════════════════

┌────────────────────────────────────────────────────────────────┐
│                     WARP SCHEDULER                             │
│                                                                │
│  1. Look at ready warps: "Who's not stalled?"                  │
│                                                                │
│  2. Pick one warp                                              │
│                                                                │
│  3. Look at its next instruction(s)                            │
│                                                                │
│  4. ISSUE = Send instruction(s) to execution pipelines         │
│                                                                │
└───────────────────────────┬────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
       ┌────────┐      ┌────────┐      ┌────────┐
       │  ALU   │      │ Tensor │      │ Load/  │
       │        │      │  Core  │      │ Store  │
       └────────┘      └────────┘      └────────┘
       
       Execution units that DO the work
- Dual Issue: 1 Arithmetic + 1 Memory instruction --> Not two different INT32/FP32 Arithmetic instructions. 
- Apparently, in Volta+, instead of 1 arthmetic+1 memory, we can do 1 INT32 Arithmetic + 1 FP32 Arithmetic and it will work 

- Per SM: 
N Schedulers -> 4
Max. warps issued -> 4(1 per scheduler)
Max. math ops -> 4(1 per scheduler's arithmetic issue)
Max. memory ops -> 4(1 per scheduler's load/store issue)

- So, best case you do 4 math and 4 memory ops per cycle on the SM 

- SFU (Special Function Unit): Handles slow transcendental ops (sin, cos, sqrt, reciprocal) on a separate pipeline. While SFU executes over multiple cycles, the scheduler hides latency by switching to other warps for dual-issue math+memory — it's parallel through latency hiding, not triple-issue.

- There is some trickery to memory load/store-> say a scheduler issued a memory load/store --> that means 1 warp should load different addresses --> 32 memory load/store ops. 
- Here comes the tricky part, even though requirement is 32 mem ld/st's, the hardware per scheduler only has 4 pipelines(hard wiring that's moving data from RAM, etc.) --> 1 cycle - only 4 mem ld/st's happen --> for 32 mem ld/st's, 8 cycles are needed. 
- Dual issue --> Even though arithmetic + mem ops are dual-issued, arithmetic happens fast as it takes ~1 cycle while the mem ops take 8 cycles 
- If its 8 cycles --> how to reduce? 1. Coalesced memory -> if the access is kinda sequential --> get cache lines --> will take lower no. of cycles

Memory LD/ST Mechanics:
═══════════════════════
- 1 warp issues LOAD → 32 threads each need their own address (normal — parallelism!)
- Hardware: only 4 LD/ST pipelines per scheduler
- Naive: 32 addresses ÷ 4 pipelines = 8 cycles minimum

Why Arithmetic is Faster:
- Blackwell: 32 FP32/INT32 units → all 32 threads complete in 1 cycle
- Dual-issue: math finishes fast, memory takes 8+ cycles (overlap via warp switching)

Coalesced Memory (the optimization):
- If 32 threads access CONSECUTIVE addresses → all fit in 1-2 cache lines(Address Coalescing Unit senses it and issues 1 mem request that get's loaded into L1/L2 cache)
- Hardware fetches cache line once → distributes to all 32 threads
- Result: 1-2 transactions instead of 32 → approaches 1 cycle

HOW COALESCED ACCESS ACTUALLY WORKS:
════════════════════════════════════

STEP 1: Scheduler collects all 32 addresses
─────────────────────────────────────────────
Thread 0:  0x1000
Thread 1:  0x1004
Thread 2:  0x1008
...
Thread 31: 0x107C

Hardware: "These all map to cache line starting at 0x1000!"


STEP 2: ONE request to memory system (not 32)
─────────────────────────────────────────────

┌────────────────────────────────────────────────────────────┐
│   Address Coalescing Unit (before pipelines)               │
│                                                            │
│   32 addresses → "Hey, these are all in 1 cache line"     │
│                → Generate 1 memory request                 │
│                                                            │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼  (1 request)
                    ┌─────────┐
                    │ L1/L2   │
                    │  Cache  │
                    └────┬────┘
                         │
                         ▼  (128-byte cache line returned)
                    ┌─────────┐
                    │ Crossbar│  ← Routes bytes to correct threads
                    └────┬────┘
                         │
         ┌───────┬───────┼───────┬───────┐
         ▼       ▼       ▼       ▼       ▼
       T0:4B   T1:4B   T2:4B   ...    T31:4B
       
       Each thread gets its 4 bytes from the cache line
└────────────────────────────────────────────────────────────┘

BEFORE the pipelines even get involved:
═══════════════════════════════════════

┌──────────────────┐     ┌─────────────────────┐     ┌──────────────┐
│  32 addresses    │ ──► │  Coalescing Logic   │ ──► │ 1-2 requests │
│  from 32 threads │     │  (HW combines them) │     │ to cache     │
└──────────────────┘     └─────────────────────┘     └──────────────┘

This happens BEFORE pipelines!
Pipelines handle the reduced # of transactions.

The pipelines go through a memory hierarchy:

LD/ST PIPELINE → MEMORY HIERARCHY
═════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                        SM (Streaming Multiprocessor)            │
│                                                                 │
│   ┌─────────────┐                                               │
│   │ LD/ST       │                                               │
│   │ Pipelines   │                                               │
│   └──────┬──────┘                                               │
│          │                                                      │
│          ▼                                                      │
│   ┌─────────────────────────────────────┐                      │
│   │  L1 Cache / Shared Memory (256 KB)  │  ← ~20-30 cycles     │
│   │  (on-chip, per SM)                  │                      │
│   └──────────────┬──────────────────────┘                      │
│                  │                                              │
└──────────────────┼──────────────────────────────────────────────┘
                   │ Miss?
                   ▼
          ┌────────────────────┐
          │  L2 Cache (shared) │  ← ~200-300 cycles
          │  (on-chip, all SMs)│
          └─────────┬──────────┘
                    │ Miss?
                    ▼
          ┌────────────────────┐
          │  Global Memory     │  ← ~400-800 cycles
          │  (HBM, off-chip)   │
          └────────────────────┘

GPU Engineer's Motto:
═════════════════════

"Memory is slow. 
 Hide it with parallelism.
 Repeat for 20 years."

- In short, GPUs excel at data-parallel workloads
- OpenAI's Triton -> Python-based GPU language. 

About CUDA Threads: 

Threads, Warps, Blocks, and Grids: 
CUDA structures parallel work into a 3-level hierarchy - threads, thread blocks, and grids - balance progammability with massive throughput. 
- Thread executes kernel code 
- Upto 1024 threads(your choice) -> 1 thread block -> SM executes it
- Bunch of thread blocks -> Kernel grid -> Complete GPU unit executes it

- Threadblock clusters -> Groups of thread blocks that can communicate with one another across SMs. 

BEFORE (Ampere and earlier):
════════════════════════════
Block A's shared mem → Only Block A threads

AFTER (Hopper+, Clusters):
══════════════════════════
Blocks in same cluster → Can access EACH OTHER's shared memory! --> do all sorts of shared memory operations. 

┌─────────── CLUSTER ───────────┐
│  Block A        Block B       │
│  ┌──────┐      ┌──────┐       │
│  │Shared│ ←──► │Shared│       │  DSMEM: Distributed
│  │Mem A │      │Mem B │       │  Shared Memory
│  └──────┘      └──────┘       │
│      ↑              ↑         │
│  Threads can access BOTH!     │
└───────────────────────────────┘

![alt text](image.png)

THREAD BLOCK CLUSTER: DSMEM (Distributed Shared Memory)
═══════════════════════════════════════════════════════

┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│      SM      │  │      SM      │  │      SM      │  │      SM      │
│  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │
│  │████████│  │  │  │████████│  │  │  │████████│  │  │  │████████│  │
│  │████████│  │  │  │████████│  │  │  │████████│  │  │  │████████│  │
│  └────────┘  │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │
│  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │
│  │ Shared │  │  │  │ Shared │  │  │  │ Shared │  │  │  │ Shared │  │
│  │ Memory │  │  │  │ Memory │  │  │  │ Memory │  │  │  │ Memory │  │
│  └───┬────┘  │  │  └───┬────┘  │  │  └───┬────┘  │  │  └───┬────┘  │
└──────┼───────┘  └──────┼───────┘  └──────┼───────┘  └──────┼───────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
┌──────┴─────────────────┴─────────────────┴─────────────────┴───────┐
│                        SM-to-SM DSMEM                              │
│  ◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ►  │
│         (Threads in cluster can access ANY SM's shared mem)        │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│                      L2 Cache / HBM Memory                         │
│                   (Global, all SMs can access)                     │
└────────────────────────────────────────────────────────────────────┘

BEFORE Hopper:  SM's shared memory → only that SM's blocks
AFTER (DSMEM):  SM's shared memory → accessible by entire cluster!

WHAT'S INSIDE AN SM:
════════════════════

┌─────────────────────────────────────────┐
│                  SM                      │
│                                          │
│  1. REGISTERS (256 KB on Blackwell)     │
│     └── Per-thread private storage       │
│     └── Fastest (0 cycle access)         │
│                                          │
│  2. L1 CACHE + SHARED MEMORY (256 KB)   │
│     ├── L1 Cache: Hardware-managed       │
│     └── Shared Memory: Programmer-managed│
│         (Same physical SRAM, partitioned)│
│                                          │
└─────────────────────────────────────────┘

- From now on, intra-thread-block shared memory funda: 

- Within each thread block, threads use low-latency on-chip shared memory and synchronize with __syncthreads(). 
- Because each barrier incurs overhead,
you should minimize synchronization points

__syncthreads() = TWO things in one
═══════════════════════════════════

1. BARRIER:     Wait for all threads to arrive here
2. MEMORY FENCE: Make all writes visible to all threads in block

- Kind of a consistency mechanism so that all threads have the same view of the data. 

- Upto 1024 threads form a thread block 
- Thread blocks are subdivided into warps of 32 threads that execute in lockstep under the SIMT model using a warp scheduler. 

- Keeping more warps in flight is known as high occupancy on the SM. 
- When CUDA code allows high occupancy, it means that when one warp stalls, another is ready to run --> this keeps the GPU's compute units busy. 


CRITICAL DISTINCTION:
═════════════════════

RESIDENT warps = Warps LOADED on SM, ready to run (up to 64)
EXECUTING warps = Warps running THIS CYCLE (4, one per scheduler)

Occupancy = RESIDENT ÷ MAX RESIDENT
          = NOT about per-cycle execution!

          MEMORY OPS ARE ASYNC / NON-BLOCKING:
════════════════════════════════════

Cycle 1:   Scheduler issues LOAD for Warp 0
           │
           ├──► LOAD request sent to memory system
           │    (memory system works INDEPENDENTLY)
           │
           └──► Warp 0 marked as "stalled, waiting for data"
                Scheduler FORGETS about Warp 0, moves on

Cycle 2:   Scheduler picks Warp 1 (doesn't care about Warp 0)

Cycle 3:   Scheduler picks Warp 2

           Meanwhile, IN PARALLEL:
           ┌─────────────────────────────────────┐
           │  Memory system doing its thing:     │
           │    → Request travels to L1          │
           │    → Miss, goes to L2               │
           │    → Miss, goes to HBM              │
           │    → Data fetched                   │
           │    → Travels back through caches    │
           │    → Arrives at register file       │
           └─────────────────────────────────────┘

Cycle 401: Memory system signals "Warp 0's data ready!"
           Warp 0 marked as "ready to run"
           Next time scheduler looks, Warp 0 is eligible again

┌─────────────────────────────────────────────────────────────────┐
│                          SM                                     │
│                                                                 │
│  ┌─────────────┐      ┌──────────────────────────────────────┐ │
│  │ Schedulers  │      │  Warp Status Table                   │ │
│  │             │      │  ┌──────┬────────────────────────┐   │ │
│  │ "Who's not  │◄────►│  │Warp 0│ STALLED (waiting mem)  │   │ │
│  │  stalled?"  │      │  │Warp 1│ READY ← pick this one! │   │ │
│  │             │      │  │Warp 2│ READY                   │   │ │
│  └─────────────┘      │  │Warp 3│ STALLED (waiting mem)  │   │ │
│                       │  │...   │                         │   │ │
│                       │  └──────┴────────────────────────┘   │ │
└───────────────────────┼──────────────────────────────────────┼─┘
                        │                                      │
                        │ ASYNC!                               │
                        ▼                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEM                                │
│                                                                 │
│  Request Queue: [Warp0:addr1] [Warp3:addr2] [...]              │
│                                                                 │
│  Working independently... fetching data... no scheduler needed │
│                                                                 │
│  When done → Signal "Warp 0 data ready" → Update status table  │
└─────────────────────────────────────────────────────────────────┘

- The whole point of resident warps is that they are available when others get stalled, so that the process keeps contributing and hence contribute to the SM occupancy. 

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



