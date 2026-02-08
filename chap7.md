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

Vectorized Memory Access: 
- 