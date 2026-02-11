Profiling and Tuning GPU Memory Access Patterns: 

Coalesced vs. Uncoalesced Global Memory Access: 
- Leverage cache lines 
- Modern GPUs - 128 byte lines - 4 32-byte sectors
- Algined contiguous memory access will better leverage memory mechanics. 
- On Blackwell GPU, per-device HBM3e bandwidth is 8 TB/s, on Grace Blackwell GB200/GB300 - increases to 16 TB/s. 

┌─────────────────────────────────────────────────────────────────────┐
│  CONCEPT          │  PURPOSE                │  SIZE                 │
├─────────────────────────────────────────────────────────────────────┤
│  Cache Line       │  Tracking/tagging       │  128B (metadata unit) │
│  Sector           │  Actual data transfer   │  32B (fetch unit)     │
│  Memory Bus Width │  Parallel transfer      │  128B (4 sectors)     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CACHE LINE = Fixed 128B region defined by ADDRESS                  │
│               (for tag lookup efficiency)                           │
│                                                                     │
│  SECTOR = Fetch granularity within a line-- the smallest unit that  │
|   actually gets transferred in a batch fashion.                     │
│           (for bandwidth efficiency)                                │
│                                                                     │
│  You CAN'T mix sectors from different lines into one "virtual line" │
│  because the TAG system needs fixed boundaries to work!             │
└─────────────────────────────────────────────────────────────────────┘

- Pulling 1 cache line is mostly 4 parallel transfers of memory by 4 sectors actually. 

Key insight: All 4 sectors of a cache line CAN be fetched in one cycle
             IF they're all needed AND in the same line! -- but if not needed, just 1 sector per cache line can be fetched too 

- Each cache line = 4 sectors (128B = 4 × 32B)

```
┌───────────────────────────────────────────────────────────────────┐
│                    128-BYTE CACHE LINE                            │
├───────────────┬───────────────┬───────────────┬───────────────────┤
│   Sector 0    │   Sector 1    │   Sector 2    │   Sector 3        │
│   (0-31)      │   (32-63)     │   (64-95)     │   (96-127)        │
└───────────────┴───────────────┴───────────────┴───────────────────┘
```

**Misalignment penalties:**

```
CASE 1: PERFECT (offset=0) → 1 transaction, 4 sectors ✅
Line A:  [████][████][████][████]    Line B:  [    ][    ][    ][    ]
          S0    S1    S2    S3                 S0    S1    S2    S3

CASE 2: SECTOR-ALIGNED CROSSING (offset=96) → 2 transactions, 4 sectors
Line A:  [    ][    ][    ][████]    Line B:  [████][████][████][    ]
                            S3                 S0    S1    S2
         └─ 1 sector ─┘              └─── 3 sectors ───┘
         Latency cost only (2 transactions, but no byte waste)

CASE 3: NON-SECTOR-ALIGNED (offset=100) → 2 transactions, 5 sectors ❌
Line A:  [    ][    ][    ][░░██]    Line B:  [████][████][████][██░░]
                           ↑waste                              ↑waste
         └─ 1 sector ─┘              └───── 4 sectors ─────┘
         Latency cost + Bandwidth cost (32B garbage on shared bus)
```

- Remember: *"Cross a line, pay in time. Cross a sector, pay in bytes."*

Perfect Coalescing = Contiguous + Aligned 

45 pages to go buddy -- just keep swimming -- just keep sailing! 
- Strided/irregular indexing - uncoalesced memory access. 
- How to diagonise this?
- NVIDIA Nsight Compute will show:
    - Lower Global Memory Load Efficiency
    - Higher DRAM sector read counts 
    - Average sectors per request > 4.0
- To fix memory bound problem: use SOA instead of AOS: 

Imagine you have 1000 particles, each with: x, y, z position + mass

┌─────────────────────────────────────────────────────────────────────┐
│  ARRAY OF STRUCTURES (AoS)                                          │
│  "Keep each particle's data together"                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  struct Particle { float x, y, z, mass; };                          │
│  Particle particles[1000];                                          │
│                                                                     │
│  Memory layout:                                                     │
│  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬───  │
│  │ x0 │ y0 │ z0 │ m0 │ x1 │ y1 │ z1 │ m1 │ x2 │ y2 │ z2 │ m2 │... │
│  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴───  │
│    └── Particle 0 ──┘   └── Particle 1 ──┘   └── Particle 2 ──┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  STRUCTURE OF ARRAYS (SoA)                                          │
│  "Keep each property together"                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  struct Particles {                                                 │
│      float x[1000];                                                 │
│      float y[1000];                                                 │
│      float z[1000];                                                 │
│      float mass[1000];                                              │
│  };                                                                 │
│                                                                     │
│  Memory layout:                                                     │
│  ┌────┬────┬────┬────┬─────┬────┬────┬────┬────┬─────┬────┬────┬─── │
│  │ x0 │ x1 │ x2 │ x3 │ ... │ y0 │ y1 │ y2 │ y3 │ ... │ z0 │ z1 │... │
│  └────┴────┴────┴────┴─────┴────┴────┴────┴────┴─────┴────┴────┴─── │
│    └──── all x's ────┘       └──── all y's ────┘                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    torch.compile + TorchInductor BENEFITS           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. REDUCES REDUNDANT COPIES                                        │
│     → Eliminates unnecessary memory transfers                       │
│                                                                     │
│  2. FUSES ADJACENT OPERATIONS                                       │
│     → Combines multiple ops into one kernel                         │
│     → Fewer kernel launches, better memory reuse                    │
│                                                                     │
│  3. AUTOTUNING FOR COALESCING                                       │
│     → mode="max-autotune" finds optimal memory patterns             │
│     → Picks vectorized schedules automatically                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

- Coalesced access is good - but non-alignment with 128 bytes will still ask for more cache lines 
- Good news is cudaMalloc() guarantees AT LEAST 256-byte alignment(often 512B on modern GPUs) - so array's BASE address is always 128B-aligned! 

- Global Memory load Efficiency: How much of fetched bytes are useful?

- SM Active % = % of cycles where the SM is doing work(not stalled or idle)

- Coalesced memory access:

STEP 1: Each thread issues a SEPARATE load instruction
────────────────────────────────────────────────────────────────────

__global__ void kernel(float* data) {
    float x = data[threadIdx.x];  // Each thread: "I want 4 bytes"
}

Thread 0:  LDG.32 addr=0    (load 4 bytes from address 0)
Thread 1:  LDG.32 addr=4    (load 4 bytes from address 4)
Thread 2:  LDG.32 addr=8    (load 4 bytes from address 8)
...
Thread 31: LDG.32 addr=124  (load 4 bytes from address 124)

= 32 separate load instructions issued!


STEP 2: Coalescing unit COLLECTS and ANALYZES requests
────────────────────────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────┐
│                    COALESCING UNIT (hardware)                   │
├─────────────────────────────────────────────────────────────────┤
│  Incoming: 32 requests                                          │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐       │
│  │ 0-3 │ 4-7 │8-11 │12-15│16-19│20-23│24-27│28-31│ ... │       │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘       │
│                                                                 │
│  Analysis: "These are all contiguous and within 0-127!"         │
│  Decision: "Merge into ONE 128-byte transaction"                │
└─────────────────────────────────────────────────────────────────┘


STEP 3: ONE transaction goes to memory
────────────────────────────────────────────────────────────────────

                    Single 128B Request
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MEMORY SYSTEM                              │
│                                                                 │
│   Returns: 128 bytes in one response                            │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼

STEP 4: Hardware DISTRIBUTES data back to each thread
────────────────────────────────────────────────────────────────────

128 bytes arrive → Coalescing unit splits it:

Thread 0  gets bytes 0-3    ←─┐
Thread 1  gets bytes 4-7    ←─┤
Thread 2  gets bytes 8-11   ←─┼── Hardware routes each 
...                           │   4-byte chunk to correct thread
Thread 31 gets bytes 124-127←─┘

┌─────────────────────────────────────────────────────────────────────┐
│  OVERHEAD OF STITCHING:                                             │
│                                                                     │
│  • 32 load instructions decoded                                     │
│  • Address comparison logic (are they contiguous?)                  │
│  • Merge logic (combine into transaction)                           │
│  • Distribution logic (split result back to threads)                │
│                                                                     │
│  This costs cycles and energy, even though result is 1 transaction! │
└─────────────────────────────────────────────────────────────────────┘

2/8/26:
Vectorized Memory Access: 
- Vectorized loads do not remove coalescing or stitching logic; they amortize it by running the same hardware logic fewer times over more data.

- If each thread needs only one aligned float, scalar loads are already optimal; vector loads help only when collapsing multiple contiguous per-thread loads into a single instruction -- data better be aligned. 

- The real win of float4 is reduced instruction and coalescer invocation count—not fewer bytes fetched or “better” cache-line utilization in the ideal scalar case.

- What: Load float4 (16B) or float8 (32B) instead of individual floats → fewer instructions, same data

- Alignment rule: Pointer address must be divisible by vector size (16B for float4, 32B for float8)

- cudaMalloc is safe: Always returns 256-byte aligned → fine for any vector width

- Offset trap: ptr + 1 breaks alignment; ptr + 4 keeps float4 aligned

- Hopper: 16 bytes/thread max → float4 = 1 load, 8 floats = 2 loads

- Blackwell: 32 bytes/thread max → 8 floats = 1 load

- PyTorch/compilers: Auto-vectorize if data is aligned and contiguous — help them by aligning from the start

- Alignment = starting address divisible by data size

Blackwell: 32 bytes/thread in one instruction (with CUDA 13)

Hopper: 16 bytes/thread max

Alignment requirement: Compiler must prove 32-byte alignment for single 32B load

If not proven: Compiler splits into two 16-byte loads → 2× instructions → slower

Takeaway: Use alignas(32) on custom structs to help compiler emit single 32B loads on Blackwell

Tiling and Data Reuse Using Shared Memory: 

- Common pitfall: Repeated reading the same data from global memory. TIling is a technique to avoid this by loading chunks of data into faster on-chip shared memory - and reusing those chunks across many threads. 
- 32x32 tile is a good tile size -- Aligns with a warp size of 32 (1 warp per row), fits well in shared memroy. 

32×32 tile:
- Each row = 32 floats = 1 warp can process one row
- 32 rows = 32 warps = 1024 threads = 1 block (max block size!)

- Perfect mapping: 1 thread per element, 1 warp per row
- 32 floats × 4 bytes = 128 bytes = 1 cache line ✅
- The main benefit of tiling is to increase arithmetic intensity by leveraging the on-chip shared memory. 

2/9/26: 
- The tiled kernel also accesses sA and sB in a way that avoids shared-memory bank conflicts, which is why shared memory throughput approaches 100%.
- sA is broadcast (same value to all), sB is strided by 1 (hits all 32 banks) — neither causes conflict
- Tile size is an optimizing parameter: too small → not enough reuse; too large → shared memory overflow and more bank conflicts. 32×32 is a sweet spot for many matrix operations on modern GPUs-- CUTLASS has profilers that can auto-optimize tile size for you.
- 32×32 tile is a good default — experiment with size to balance reuse vs occupancy
- Libraries do this for you — CUTLASS, cuBLAS, cuDNN, TorchInductor all auto-tile under the hood
- cuTile (2025) — NVIDIA's Python API for expressing tile shapes without writing CUDA

Avoiding Shared Memory Bank Conflicts:
- Shared memory has 32 banks, each 4 bytes wide.
- Bank conflict occurs when multiple threads in a warp access different addresses that map to the same bank
- If all threads access the same address (broadcast)(even in the same bank) → no conflict, 1 access
- If threads access addresses that map to different banks → no conflict, 1 access
- If threads access addresses that map to the same bank → conflict, serialized accesses (up to 32 accesses if all threads hit the same bank)
- In matrix transpose, naive access causes conflicts because threads access columns (strided by row size), which map to the same bank.
- Every 32nd element(after 128 bytes) maps to the same bank, so strided access by 32 causes conflicts.
- Always choose your stride and data layout so that threads in the same warp hit different banks and avoid that serializing bottleneck. 
- Fix: Adjust data layouts in shared memory to avoid conflicts. 
- Padding: 
    - Before stride - 32 -- all accesses map to the same bank → 32-way conflict
    - After stride - 33(make tile 32x33) -- accesses map to different banks → no conflict
- PyTorch has no high-level API for padding, implement in CUDA, then load with torch.utils.cpp_extension || but cuDNN and cuBLAS implement the techniques under the hood to avoid bank conflicts and maximize throughput. 
- Another technique used by libraries: Swizzling:

- What: XOR row with col (col ^ row) to scatter accesses across banks --XOR turns tile[varying_row][constant_col](causing bank conflicts) into tile[varying_row][varying_col] — different columns = different banks (both write and consecutive read should do swizzling)
- Why it works: XOR is symmetric (A^B == B^A) and preserves uniqueness — 32 unique inputs → 32 unique outputs → 32 different banks
- Benefit over padding: No wasted memory, same conflict-free result

Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization: 
- What if we skip shared memory altogether?
- NVIDIA GPUs support warp-synchronous primitives that allow threads in the same warp to exchange data through registers instead of shared memory. 
- __shfl_sync: broadcast a value from one thread to all other threads in a warp. 

2/10/26:
- What: __shfl_sync lets threads read each other's registers directly — no shared memory needed
- Why it's fast: Register-to-register transfer = zero bank conflicts, zero sync overhead
- Butterfly reduction: Halve distance each step (16→8→4→2→1), lane 0 gets final sum in 5 steps
- Limitation: Works within 1 warp (32 threads) only — cross-warp still needs shared/global memory
- Modern API: Cooperative Groups wraps shuffles with cleaner syntax (thread_group.shuffle())
- Gold: Master intra-warp and inter-warp communication patterns as memory is the main bottleneck for GPU performance.

Read-Only Data Caches: 
- For some NLP tasks, we have repetitive, read-only data access patterns -- use the larger read-only data cache in L1 instead of small 64KB constant memory cache. 
- const __restrict__ qualified pointers define function arguments as non-coherent and read-mostly(vs. read-only). 
- For read-only data, the compiler may route loads through this read-only path when it can prove immutability, nonaliasing, and safety. 
- Non-aliasing = pointers don't overlap; __restrict__ promises this, letting compiler optimize aggressively

void kernel(float* A, float* B) {
    float x = A[0];  // Load A[0]
    B[0] = 5.0;      // Store to B[0]
    float y = A[0];  // Load A[0] again
    
    // Can compiler reuse x for y?
    // If A and B DON'T alias: yes, y = x (skip second load) ✅
    // If A and B DO alias: no, must reload (B[0] changed A[0]!) ❌
    
    // Compiler must assume they MIGHT alias → conservative, slower
}

void kernel(float* __restrict__ A, float* __restrict__ B) {
    // You PROMISE: A and B point to completely separate memory
    // Compiler can now optimize aggressively!
    
    float x = A[0];
    B[0] = 5.0;
    float y = A[0];  // Compiler knows A wasn't touched → reuse x!
}

- Previously, __ldg() loads were used for read-only data, but now the compiler can route through the larger read-only cache when it can prove safety, so __restrict__ is the key to unlocking that optimization.
- Broadcast: Same address across warp → 1 load serves 32 threads (free!)
- Caching: Different addresses → cached for future reuse across warps
- Lookup tables benefit from #2 — popular entries stay hot in cache, not broadcast -- next warp has faster access
- Use texture/surface memory whenever your access pattern has 2D/3D locality that a regular cache might not handle optimally. 

Asynchronous Memory Prefetching and Tensor Memory Accelerator: 
- 800 cycles of DRAM latency -- overlap data transfer with compute 
- CUDA Pipeline API + Tensor Memory Accelerator(TMA): Instead of having each warp use the SM's load and store units to fetch data from global memory, you can invoke the TMA engine to asynchronously fech an entire tile from global memory into shared memory. 
- To start TMA transfer: use cuda::memcpy_async + cuda::pipeline
- Double buffering/ping-ponging: While one tile is being processed in shared memory, the next tile is being fetched by TMA into another buffer -- overlap compute and memory transfer for maximum throughput.
- TMA: specific hardware for bulk data transfer from global dram memory into on-chip l1 memory, that happens without the use of load/store instruction units asynchronously so that the warp can focus on computation. am i getting it right?
- Use TMA for bulk, strided, and 2D/3D transfers. 

Chap 7 done!


