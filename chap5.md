1/30/26: 
GPU-Based Storage I/O Optimizations: 
1/31/26:
Fast Storage and Data Locality: 
Solution 1:  Colocating NVMe SSDs within racks - or using NVMe over Fabrics(NVME-OF) with rack-local switch topologies - minimizes network hops and improves performance consistency. 

NVMe-oF = Access remote NVMe SSDs over a network as if they were local.

Traditional NVMe:
┌─────────┐  PCIe   ┌─────────┐
│   CPU   │◄───────►│  NVMe   │  Only local SSDs
└─────────┘         │  SSD    │
                    └─────────┘

NVMe-oF (NVMe over Fabrics):
┌─────────┐         ┌─────────┐   Network    ┌─────────┐
│   CPU   │◄───────►│   NIC   │◄────────────►│ Remote  │
└─────────┘         └─────────┘  (RDMA/TCP)  │  NVMe   │
                                             │  SSDs   │
   Looks like local NVMe,                    └─────────┘
   but storage is remote!

Even with NVMe-oF, keep the storage within the same rack to minimize hops:

RACK-LOCAL NVMe-oF (Good):
┌─────────────────────────────────────────┐
│                 RACK 1                   │
│  ┌─────┐        ┌──────────┐            │
│  │ GPU │◄──────►│  Switch  │            │
│  └─────┘        └────┬─────┘            │
│                      │ 1 hop            │
│               ┌──────▼──────┐           │
│               │  NVMe-oF    │           │
│               │  Storage    │           │
│               └─────────────┘           │
└─────────────────────────────────────────┘

CROSS-RACK NVMe-oF (Slower):
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   RACK 1    │       │   RACK 2    │       │   RACK 3    │
│  ┌─────┐    │       │  ┌──────┐   │       │  ┌───────┐  │
│  │ GPU │────┼──────►│  │Switch│───┼──────►│  │ NVMe  │  │
│  └─────┘    │       │  └──────┘   │       │  │Storage│  │
└─────────────┘       └─────────────┘       └─────────────┘
                  2+ hops = more latency, less consistent

Solution 2:  Use parallel filesystem like Lustre or General Parallel FIle SYstem(GPFS), etc. to cache the data on local SSD's. 
- Goal is to saturate the GPU's with data 
- Usually, we shard(split) the data acorss different nodes when all data doesn't fit on RAM -> PyTorch's DistributedSampler. 

Sequential versus Random Read Patterns: 
- Large sequntial reads > small random reads for throughput -> organize the data as such. 
- For images, instead of storing each image as a separate file, store them in large binary files (like TFRecords, Arrow, Parquet, etc.) to enable large sequential reads.
- Tune the read size -> reading in 1MB chunks better than 4KB chunks due to lower per-read overhead. 

- Every read has fixed overhead(syscall, context switch, etc.)
- So, 1000x4 KB reads x 1 us overhead = 1 ms overhead
- Whereas, 4x1 MB reads x 1 us overhead = 4 us overhead --> 250x lower overhead!
- Read-ahead --> You ask to get 4KB by OS, but OS actually reads 128KB anticipating future reads -> no utility for random reads. 
- Term	What It Is
Buffer size	    How much data to read at once into memory
Prefetch	    Read data BEFORE it's needed (hide latency)
pread()	        Read at specific offset without moving file pointer (thread-safe)
io_uring	    Modern Linux async I/O interface (very fast)
IOPS	        I/O Operations Per Second (key metric for random access)
Syscall	        Call from user program to kernel (has overhead)

SEQUENTIAL ACCESS (Best case):
┌─────────────────────────────────────────────────────┐
│  Read large chunks + prefetch + buffered I/O       │
│                                                     │
│  File: [████████████████████████████████████]      │
│         ──────────────────────────────────────►    │
│         Continuous read, full bandwidth            │
│                                                     │
│  Strategy:                                          │
│  • Large buffer size (1 MB+)                       │
│  • Enable prefetch                                  │
│  • Let OS read-ahead help                          │
└─────────────────────────────────────────────────────┘

RANDOM ACCESS (Challenging):
┌─────────────────────────────────────────────────────┐
│  Parallel reads + io_uring + batch requests        │
│                                                     │
│  File: [██    ██      ██  ██        ██    ██]     │
│          ▲     ▲       ▲   ▲         ▲     ▲       │
│          └─────┴───────┴───┴─────────┴─────┘       │
│          Random positions, can't predict           │
│                                                     │
│  Strategy:                                          │
│  • Parallel threads with pread()                   │
│  • io_uring for async batched I/O                  │
│  • Minimize per-read overhead                      │
└─────────────────────────────────────────────────────┘

io_uring = Linux's modern async I/O (since kernel 5.1). Submit many I/O requests at once, kernel processes them in parallel.

TRADITIONAL SYNC I/O:
┌────────┐        ┌────────┐
│  App   │        │ Kernel │
└────────┘        └────────┘
    │ read() ─────────►│
    │◄─────── wait ────│  Block!
    │ read() ─────────►│
    │◄─────── wait ────│  Block!
    │ read() ─────────►│
    │◄─────── wait ────│  Block!
    
    3 reads = 3 round trips, serialize waiting


io_uring ASYNC I/O:
┌────────┐        ┌────────┐
│  App   │        │ Kernel │
└────────┘        └────────┘
    │                  │
    │ Submit batch ───►│  ← Submit 100 reads at once!
    │ (100 requests)   │
    │                  │──► Disk
    │  Do other work   │──► Disk  (parallel!)
    │                  │──► Disk
    │◄── Completions ──│  ← Get results in batch
    
    100 reads = 1 round trip, parallel execution

DECISION TREE:

Is your access pattern sequential?
│
├─► YES: Use large buffers + prefetch
│        • TFRecordDataset, PyTorch DataLoader
│        • Buffer size: 1 MB+
│        • Prefetch: 2-4 batches ahead
│        • Let OS read-ahead help
│
└─► NO (Random): Parallelize + reduce overhead
         • Multiple threads + pread()
         • OR io_uring for async batched I/O
         • Goal: high IOPS, hide latency
         • Better yet: RESTRUCTURE to sequential!

DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,        # Parallel data loading threads
    prefetch_factor=2,    # Prefetch 2 batches per worker
    pin_memory=True,      # Faster GPU transfer
)

# Under the hood:
# Worker 0: Loading batch N+2
# Worker 1: Loading batch N+3
# Worker 2: Loading batch N+4   ← Prefetching ahead!
# ...
# GPU: Processing batch N
# Main: Preparing batch N+1

XFS = A high-performance filesystem designed for large files and parallel I/O.

- --- IGNORE ---

FILESYSTEM COMPARISON:

ext4 (Default Linux):
┌─────────────────────────────────────────┐
│ • Good for general use                  │
│ • Single-threaded allocation            │
│ • Max file size: 16 TB                  │
│ • Struggles with many parallel writes   │
└─────────────────────────────────────────┘

XFS (Optimized for large I/O):
┌─────────────────────────────────────────┐
│ • Designed for large files (video, AI)  │
│ • Parallel allocation groups            │
│ • Max file size: 8 EB (exabytes!)      │
│ • Excellent concurrent I/O              │
│ • Used by Netflix, NASA, AI clusters    │
└─────────────────────────────────────────┘

Linux NVMe Servers
NVMe = Non-Volatile Memory Express, the fast SSD protocol.
Linux NVMe server = A Linux server using NVMe SSDs for storage.

NVMe = Non-Volatile Memory Express, the fast SSD protocol.
Linux NVMe server = A Linux server using NVMe SSDs for storage.

STORAGE SPEED EVOLUTION:

HDD (Spinning disk):     ~100-200 MB/s, ~10ms latency
         │
         ▼
SATA SSD:                ~500 MB/s, ~100μs latency
         │
         ▼
NVMe SSD (PCIe direct):  ~3-7 GB/s, ~10-20μs latency  ← This!
         │
         ▼
NVMe over PCIe 5.0:      ~14 GB/s

"Linux NVMe server" = Server with fast NVMe SSDs running Linux

+-------------+--------------------------------------+-----------------------------------------------+
| Term        | What It Is                           | Why It Matters for AI                         |
+-------------+--------------------------------------+-----------------------------------------------+
| XFS         | High-performance filesystem          | Handles large files + parallel I/O well       |
| NVMe        | Fast SSD protocol (PCIe)             | 10-100x faster than SATA SSD                  |
| noatime     | Skip access-time writes              | Eliminates write-per-read overhead            |
| Bursting(Amazon EFS)    | Variable throughput (credit-based)   | Bad for sustained training (credits deplete)  |
| Provisioned(Amazon EFS)  | Fixed guaranteed throughput          | Predictable, no surprises                     |
| Max I/O(Amazon EFS)      | High aggregate IOPS mode             | Better for parallel AI workloads              |
+-------------+--------------------------------------+-----------------------------------------------+

An I/O scheduler decides the order in which disk read/write requests are processed. It sits between your application and the storage device.

Why Schedulers Exist (Historical Context)
For HDDs (Spinning Disks) - Schedulers Were Critical

HDD: Physical disk head must MOVE to read data

Disk:  [A]----[C]--------[B]----[D]
        │      │          │      │
        ▼      ▼          ▼      ▼
     Track 1  Track 50   Track 30 Track 80

Random order: A → B → C → D
Head moves:   1 → 30 → 50 → 80  (lots of seeking!)

Scheduler reorders: A → C → B → D  
Head moves:         1 → 50 → 30 → 80  
                    ↓
              Optimized: 1 → 30 → 50 → 80 (sequential sweep!)
              
HUGE performance gain from smart scheduling!

For SSDs/NVMe - Less Critical

NVMe SSD: No moving parts, random access ≈ sequential access

Request A ──►  ┌─────────┐
Request B ──►  │  Flash  │ ──► All ~same speed!
Request C ──►  │  Chips  │     No seeking penalty
Request D ──►  └─────────┘

Scheduler overhead > benefit
→ Just pass requests through ("none" scheduler)

The Schedulers Explained
1. none (No Scheduler)

┌─────────────────────────────────────────────────────────────┐
│  "none" SCHEDULER                                           │
│                                                             │
│  App requests: [A][B][C][D]                                │
│       │                                                     │
│       ▼  (pass through, no reordering)                     │
│  To device:   [A][B][C][D]                                 │
│                                                             │
│  • Zero overhead                                            │
│  • Best for NVMe (device has its own scheduler)            │
│  • Lowest latency                                           │
└─────────────────────────────────────────────────────────────┘

2. mq-deadline (Multiqueue Deadline)
┌─────────────────────────────────────────────────────────────┐
│  "mq-deadline" SCHEDULER                                    │
│                                                             │
│  App requests: [A][B][C][D]                                │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────┐                       │
│  │ • Batches requests              │                       │
│  │ • Guarantees deadline (no starvation)│                  │
│  │ • Slight reordering for efficiency   │                  │
│  └─────────────────────────────────┘                       │
│       │                                                     │
│  To device: [A][C][B][D] (maybe reordered)                 │
│                                                             │
│  • Good balance of throughput + fairness                   │
│  • Prevents any request from waiting forever               │
│  • Small overhead, still good for NVMe                     │
└─────────────────────────────────────────────────────────────┘

3. bfq (Budget Fair Queueing)
┌─────────────────────────────────────────────────────────────┐
│  "bfq" SCHEDULER                                            │
│                                                             │
│  Process 1 requests: [A][B][C]                             │
│  Process 2 requests: [X][Y][Z]                             │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────────────────────────┐                       │
│  │ • Fair sharing between processes│                       │
│  │ • Each gets "budget" of I/O     │                       │
│  │ • Good for desktop/interactive  │                       │
│  └─────────────────────────────────┘                       │
│       │                                                     │
│  To device: [A][X][B][Y][C][Z] (interleaved fairly)        │
│                                                             │
│  • Best for shared systems (multiple users)                │
│  • Higher overhead than none/mq-deadline                   │
│  • NOT recommended for dedicated AI servers                │
└─────────────────────────────────────────────────────────────┘

4. cfq (Completely Fair Queueing) - OBSOLETE
┌─────────────────────────────────────────────────────────────┐
│  "cfq" SCHEDULER (OLD - Don't use!)                        │
│                                                             │
│  • Designed for spinning HDDs                               │
│  • Single queue (can't use multiple CPU cores)             │
│  • Removed from modern kernels                              │
│  • Replaced by bfq for fairness needs                      │
└─────────────────────────────────────────────────────────────┘

blk-mq: Multiqueue Block Layer
blk-mq = The modern Linux block I/O layer that supports multiple queues.

OLD (Single Queue):
┌───────────┐      ┌─────────────┐      ┌────────┐
│  All CPUs │─────►│ ONE Queue   │─────►│  NVMe  │
│           │      │ (bottleneck)│      │        │
└───────────┘      └─────────────┘      └────────┘
                         ↑
                    Lock contention!
                    Can't parallelize


NEW (blk-mq - Multiple Queues):
┌───────────┐      ┌─────────────┐
│   CPU 0   │─────►│  Queue 0    │──┐
├───────────┤      ├─────────────┤  │   ┌────────┐
│   CPU 1   │─────►│  Queue 1    │──┼──►│  NVMe  │
├───────────┤      ├─────────────┤  │   │(32-128 │
│   CPU 2   │─────►│  Queue 2    │──┤   │ queues)│
├───────────┤      ├─────────────┤  │   └────────┘
│   CPU 3   │─────►│  Queue 3    │──┘
└───────────┘      └─────────────┘
                         ↑
               No lock contention!
               Full parallelism!

NVMe SSDs have hardware queues (often 32-128), and blk-mq lets Linux use them all.

blk-mq = multiple lanes (no CPU contention); queue depth = cars per lane (keep device busy). You need both for maximum NVMe throughput.

- For high-performance NVMe, it’s recommended to still use the
none or mq-deadline multiqueue scheduler to maximize through‐
put. You can verify and set the scheduler using /sys/block/nvme*/
queue/scheduler. It’s almost always configured properly out of the
box, but it’s worth verifying with a quick check.

- NVMe = "Remove the middleman" — direct PCIe connection + modern protocol = 10× lower latency, 10× more throughput  than SATA SSDs.

- 2. Tune read ahead -- increase size from 128KB(default) to MB's using blockdev --setra
- Motherboard has many PCIe slots -> PCIe 5.0, 4.0, 3.0, etc. -> ensure NVMe SSDs are in the fastest slots.
- PCIe lanes: PCIe lanes are like highway lanes between devices and CPU. 
- Usually, CPUs have a limited number of total PCIe lanes(128)
- RAID0 striping: Single SSD has limited throughput -> combine multiple SSDs in RAID0 to increase throughput.

SINGLE SSD BOTTLENECK:

┌─────────┐      7 GB/s       ┌─────────┐
│  NVMe   │──────────────────►│   GPU   │  GPU wants 20 GB/s
│  SSD    │                   │         │  but SSD only gives 7 GB/s
│ 7 GB/s  │                   │         │  
└─────────┘                   └─────────┘
                                   ↑
                          GPU starving for data!

The Solution: RAID 0 (Striping)
Striping = Split each file across multiple SSDs, read from all simultaneously.                  

RAID 0 (3 SSDs Striped):

File "dataset.bin" is split into chunks:
[Chunk A][Chunk B][Chunk C][Chunk D][Chunk E][Chunk F]...

Stored across 3 SSDs:
┌─────────┐  ┌─────────┐  ┌─────────┐
│  SSD 0  │  │  SSD 1  │  │  SSD 2  │
│ Chunk A │  │ Chunk B │  │ Chunk C │
│ Chunk D │  │ Chunk E │  │ Chunk F │
│   ...   │  │   ...   │  │   ...   │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     │   7 GB/s   │   7 GB/s   │   7 GB/s
     │            │            │
     └────────────┼────────────┘
                  │
                  ▼  (Combined: 21 GB/s!)
            ┌─────────┐
            │   GPU   │  Now GPU gets 21 GB/s!
            │         │  
            └─────────┘

WITHOUT RAID 0 (Sequential read from one SSD):
Time ──────────────────────────────────────────────►

SSD 0: [████A████][████B████][████C████][████D████]
                                                    
Total time: ████████████████████████████████████████


WITH RAID 0 (Parallel read from 3 SSDs):
Time ──────────────────────────────────────────────►

SSD 0: [████A████][████D████]
SSD 1: [████B████][████E████]    ← All read in PARALLEL!
SSD 2: [████C████][████F████]
                    
Total time: ██████████████  (3× faster!)            

Slot = physical connector size (where you plug in); Lane = actual data wires inside (determines bandwidth). A x16 slot might only have x8 lanes wired—always check specs(Halves bandwidth!).

- If you can, put all your data into CPU+GPU cache, if you can't, stream the data with optimized I/O.
- Use multiple workers in data loading (tune by empirical testing).
- NVIDIA Grace CPU has 72 cores
- pin_memory=True = Workers load data into pinned RAM → faster DMA(mechanism for data transfer) to GPU, skips one copy.
- # In training loop:
data = data.to(device, non_blocking=True)
target = target.to(device, non_blocking=True)
# CPU continues immediately, transfer happens in background

- non_blocking=True = CPU doesn't wait for GPU transfer to finish, enables overlap

non_blocking=False (Default):
Timeline:
─────────────────────────────────────────────────────────────►

CPU:  [Prepare batch]  [WAIT...]  [Prepare next]  [WAIT...]
                          │                          │
GPU:               [Transfer][Compute]      [Transfer][Compute]


non_blocking=True:
Timeline:
─────────────────────────────────────────────────────────────►

CPU:  [Prepare batch][Prepare next][Prepare next]...
                │          │
GPU:      [Transfer][Compute][Transfer][Compute]...
              └──────┬──────┘
           Overlapped! CPU doesn't wait

- persistent_workers=True = Workers stay alive across epochs, no respawn overhead.
- prefetch_factor=N = Each worker keeps N batches ready; higher = less GPU starvation, more RAM.
- Too few workers → GPU starves waiting for data, too many → thread contention --> tune by optimizing for 100% utilization of disk throughput and some headroom on CPU. 

Using NVIDIA GDS: 
- GPU Direct Storage (GDS) = Bypass CPU for data transfer from NVMe SSD to GPU memory.
- Traditional path: NVMe SSD → CPU RAM → GPU memory (2 copies, CPU overhead)
- GDS path: NVMe SSD → GPU memory (1 copy, no CPU overhead)
- GDS complements GPUDirect RDMA since GDS accelerates storage-to-GPU DMA, while GPUDirect RDMA accelerates network-to-GPU DMA. Neither eliminates CPU orchestration(Still involved). Both remove the host memory bounce buffer.
- GPUDirect RDMA/GDS = CPU still orchestrates (setup, initiate, completion), but data bypasses CPU memory—the expensive copy is gone, only cheap control plane remains.
- IBGDS: Eliminates CPU orchestration too 

┌─────────────────────────────────────────────────────────────────────┐
│                    GDS REQUIREMENTS                                 │
│                                                                     │
│  HARDWARE:                                                          │
│  ☐ Modern NVIDIA GPU (Volta or newer)                              │
│  ☐ NVMe SSD (local or NVMe-oF)                                     │
│  ☐ RDMA-capable NIC (for networked storage)                        │
│                                                                     │
│  SOFTWARE:                                                          │
│  ☐ NVIDIA drivers (with nvidia-fs module)                          │
│  ☐ CUDA toolkit (with cuFile library)                              │
│  ☐ Supported filesystem (XFS or EXT4)                              │
│  ☐ O_DIRECT flag (bypass page cache)                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

         SUPPORTED                          NOT SUPPORTED
         
┌─────────────────────────┐        ┌─────────────────────────┐
│ • Local NVMe + XFS      │        │ • Regular HDD           │
│ • Local NVMe + EXT4     │        │ • SATA SSD              │
│ • NVMe-oF + RDMA        │        │ • NFS over TCP (no RDMA)│
│ • NFS over RDMA         │        │ • Standard page cache   │
│ • BeeGFS, WekaFS, VAST  │        │ • Without O_DIRECT      │
│ • IBM Storage Scale     │        │                         │
└─────────────────────────┘        └─────────────────────────┘
         ↓                                    ↓
   GPU ◄── Direct DMA               GPU ◄── CPU bounce buffer
      (fast!)                            (slow)

- GDS needs: Modern GPU + NVMe storage + XFS/EXT4 with O_DIRECT + NVIDIA drivers with nvidia-fs. Supported: local NVMe, NVMe-oF, NFS/RDMA, enterprise parallel filesystems (BeeGFS, WekaFS, VAST).
- cuFile is the GDS enabling library that takes care of read from GDS. 
- GDS gave 20% speedup in some sequntial workloads on A100 and 30% on H100(higher NIC bandwidth+greater CPU burden).
- O_DIRECT is a flag you pass when opening a file that tells the OS: "Bypass the page cache—read/write directly to/from my buffer."
- OS Page Cache: Kernel-managed RAM buffer

NORMAL I/O (Without O_DIRECT):

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  App: "Read file.dat"                                              │
│          │                                                          │
│          ▼                                                          │
│  ┌────────────────────────────────────────┐                        │
│  │           OS PAGE CACHE                │  ← OS copies data here │
│  │  (Kernel-managed RAM buffer)           │     first              │
│  │  [file.dat cached copy]                │                        │
│  └──────────────────┬─────────────────────┘                        │
│                     │ Copy                                          │
│                     ▼                                               │
│  ┌────────────────────────────────────────┐                        │
│  │         YOUR APP'S BUFFER              │  ← Then copies to you  │
│  └────────────────────────────────────────┘                        │
│                                                                     │
│  SSD ──► Page Cache ──► App Buffer                                 │
│              ↑                                                      │
│         Extra copy!                                                │
│         Extra RAM usage!                                           │
└─────────────────────────────────────────────────────────────────────┘

Good for: Repeated reads of same file (cache hit = fast!)
Bad for:  Large sequential reads (cache fills up, evicts useful data)

┌─────────────────────────────────────────────────────────────────────┐
│              1. NORMAL I/O (No O_DIRECT, No GDS)                   │
│                                                                     │
│  ┌─────┐      ┌────────────┐      ┌─────────┐      ┌─────┐        │
│  │ SSD │─────►│ Page Cache │─────►│ CPU RAM │─────►│ GPU │        │
│  └─────┘      └────────────┘      └─────────┘      └─────┘        │
│                     ↑                   ↑               ↑          │
│                 Copy #1             Copy #2         Copy #3        │
│                                                                     │
│  3 copies, CPU busy, page cache fills up                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              2. O_DIRECT ONLY (No GDS)                             │
│                                                                     │
│  ┌─────┐                          ┌─────────┐      ┌─────┐        │
│  │ SSD │─────────────────────────►│ CPU RAM │─────►│ GPU │        │
│  └─────┘         DMA              └─────────┘      └─────┘        │
│                   ↑                    ↑               ↑           │
│           Direct to app buffer     Copy #1         Copy #2         │
│           (bypass page cache)   (still in CPU)   (to GPU)         │
│                                                                     │
│  2 copies, no page cache pollution, CPU still involved             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│              3. O_DIRECT + GDS (Full optimization)                 │
│                                                                     │
│  ┌─────┐                                           ┌─────┐        │
│  │ SSD │──────────────── DMA ─────────────────────►│ GPU │        │
│  └─────┘                                           └─────┘        │
│                          ↑                                         │
│                    Direct path!                                    │
│                    Zero CPU copies!                                │
│                                                                     │
│  0 copies through CPU, CPU free, fastest!                          │
└─────────────────────────────────────────────────────────────────────┘

- Gold: Use O_DIRECT + GDS. 
- Engineer: Validate the speed-ups on your workload and fabric, as uplifts vary by IO size, queue depth, NIC generation, filesystem implementation, etc. 

- GDS gains are dependent on CPU's capabilities and amount of saturation. 
- We usually benefit from GDS during reads which is major modality during training but we also want to do checkpointing 
- WekaFS is a well known storage provider for ultrascale AI training workloads. They offer a parallel filesystem that ships with GDS-aware plugins for both read and write workloads over RDMA. 

Checkpoiniting GPU State with cuda-checkpoint: 
- Checkpointing = Saving a complete snapshot of a running program's state so you can resume from that exact point later.
- Snapshots the entire process without app cooperation:
┌─────────────────────────────────────────────────────────────────────┐
│              SYSTEM-LEVEL CHECKPOINT (cuda-checkpoint + CRIU)       │
│                                                                     │
│  STEP 1: cuda-checkpoint suspends CUDA                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • Wait for GPU kernels to finish                           │   │
│  │  • Copy GPU memory → CPU memory                             │   │
│  │  • Release GPU resources                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  STEP 2: CRIU snapshots CPU process                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • Dump all CPU memory to files                             │   │
│  │  • Save file descriptors, sockets, etc.                     │   │
│  │  • Save thread states                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  RESULT: Complete process image saved to disk                      │
│          Can restore on same or different machine!                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

- cuda-checkpoint = NVIDIA tool that quiesces CUDA state (waits for kernels, copies GPU memory to CPU, releases GPU resources) so CRIU can snapshot the entire process -> you can retrieve and use it later on. 
- CUDA checkpoints are useful for fault tolerance, preemption, and migration of long-running training and inference jobs. 
- CRIU = Checkpoint/Restore In Userspace, Linux tool to snapshot and restore running processes. 
- Together, cuda-checkpoint + CRIU enable full GPU process checkpointing -> No direct DMA though(unlike GDS)

Measuring GDS with gdsio:
- Measure throughput between NVMe SSD and GPU memory

- DeepSeek's 3FS-> Fire-Flyer File System--> AI workloads are random accesses -> remove cache management and optimize for direct storage I/O

- RDMA: Remote Direct Memory Access - It allows one computer to read/write directly into another computer's memory over the network, without involving the remote CPU. 

┌─────────────────────────────────────────────────────────────────────┐
│              TRADITIONAL NETWORK (TCP/IP)                          │
│                                                                     │
│  Machine A                              Machine B                   │
│  ┌─────────┐                            ┌─────────┐                │
│  │   App   │                            │   App   │                │
│  └────┬────┘                            └────▲────┘                │
│       │ syscall                              │ syscall             │
│  ┌────▼────┐                            ┌────┴────┐                │
│  │   CPU   │ ◄── Process packets ──►    │   CPU   │                │
│  └────┬────┘                            └────▲────┘                │
│       │ copy                                 │ copy                │
│  ┌────▼────┐                            ┌────┴────┐                │
│  │ Kernel  │                            │ Kernel  │                │
│  │ Buffer  │                            │ Buffer  │                │
│  └────┬────┘                            └────▲────┘                │
│       │ copy                                 │ copy                │
│  ┌────▼────┐     Network packet         ┌────┴────┐                │
│  │   NIC   │ ─────────────────────────► │   NIC   │                │
│  └─────────┘                            └─────────┘                │
│                                                                     │
│  PROBLEMS:                                                          │
│  • CPU on BOTH sides processes every packet                        │
│  • Multiple memory copies (app → kernel → NIC)                     │
│  • Context switches (user ↔ kernel)                                │
│  • High latency (~50-100+ μs)                                      │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                    RDMA (InfiniBand / RoCE)                        │
│                                                                     │
│  Machine A                              Machine B                   │
│  ┌─────────┐                            ┌─────────┐                │
│  │   App   │                            │   App   │                │
│  │ Buffer  │                            │ Buffer  │                │
│  └────┬────┘                            └────▲────┘                │
│       │                                      │                     │
│       │ (registered memory)                  │ (registered memory) │
│       │                                      │                     │
│  ┌────▼────┐     Direct memory access   ┌────┴────┐                │
│  │  RDMA   │ ─────────────────────────► │  RDMA   │                │
│  │   NIC   │    (NIC writes directly    │   NIC   │                │
│  └─────────┘     to app's buffer!)      └─────────┘                │
│                                                                     │
│  BENEFITS:                                                          │
│  • Remote CPU NOT involved (zero-copy on receiver!)                │
│  • No kernel involvement after setup                               │
│  • Single memory copy (or zero on receive side)                    │
│  • Very low latency (~1-2 μs)                                      │
└─────────────────────────────────────────────────────────────────────┘
- A FUSE based filesystem -> No GDS -> GDS requires kernel-level filesystem integration with O_DIRECT semantics 
- To feed data directly into GPU pipelines, DeepSeek integrates RDMA-based transfers
in 3FS. If you require a true GDS path, use a GDS-enabled kernel filesystem client
such as NVMe, NVMe-oF, BeeGFS, WekaFS, IBM Storage Scale, or VAST. This allows
asynchronous, zero-copy data movement directly into GPU device memory with
minimal overheads.

┌─────────────────────────────────────────────────────────────────────┐
│         SCENARIO 1: LOCAL STORAGE → GPU (Same Node)                │
│                                                                     │
│  ┌─────────┐      PCIe       ┌─────────┐                          │
│  │  NVMe   │────────────────►│   GPU   │                          │
│  │  SSD    │    (GDS/DMA)    │         │                          │
│  └─────────┘                 └─────────┘                          │
│                                                                     │
│  Interconnect: PCIe (not RDMA)                                     │
│  Technology: GPUDirect Storage (GDS)                               │
│  Scope: Within one server/node                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│         SCENARIO 2: REMOTE STORAGE → GPU (Across Network)          │
│                                                                     │
│  Storage Node                           Compute Node               │
│  ┌─────────┐     InfiniBand/RoCE      ┌─────────┐    ┌─────────┐  │
│  │  NVMe   │────────────────────────►│   NIC   │───►│   GPU   │  │
│  │  SSDs   │       (RDMA)             │  (RDMA) │    │         │  │
│  └─────────┘                          └─────────┘    └─────────┘  │
│                                                                     │
│  Interconnect: InfiniBand or RoCE (RDMA network)                   │
│  Technology: RDMA + GPUDirect RDMA                                 │
│  Scope: Across network (3FS, distributed storage)                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

2/1/26:
- So, DeepSeek's built their own filesystem that optimizes concurrent I/0 alongside computation -> 6.6TB/s read throughput with a 10x16TB NVMe SSDs AI-HPC cluster. 
- Not all can do their own file system -> explore existing ones 
- Generally, people use shared fs like NFS, or parallel fs like GPFS, Ceph, etc. -> all nodes access the same dataset. 
- The fs can be a bottleneck if not configured prpoerly
- NFS: use multiple NIC's on a server, use multiple servers. 
- For modern AI training clusters -> prefer parallale filesystems and cloud storage caches like Amazon FSx for Lustre, etc. 
- Some tuning configs for NFS: 
+------------------+------------+--------------------------------------------------+
| Option           | Value      | What It Does                                     |
+------------------+------------+--------------------------------------------------+
| rsize            | 1048576    | Max read request size - fewer requests, better   |
|                  | (1 MB)     | throughput                                       |
+------------------+------------+--------------------------------------------------+
| wsize            | 1048576    | Max write request size - fewer requests, better  |
|                  | (1 MB)     | throughput                                       |
+------------------+------------+--------------------------------------------------+
| noatime          | -          | Skip access-time updates - eliminates            |
|                  |            | write-per-read overhead                          |
+------------------+------------+--------------------------------------------------+
| async            | -          | Async writes - don't wait for server             |
|                  |            | confirmation                                     |
+------------------+------------+--------------------------------------------------+
| actimeo          | 60         | Cache file attributes for 60s - reduces          |
|                  |            | metadata round-trips                             |
+------------------+------------+--------------------------------------------------+
| lookupcache=pos  | -          | Cache directory entries - speeds up file lookups |
+------------------+------------+--------------------------------------------------+

- Backend Requirements
    Use NVMe SSD storage on NFS server
    Consider RAID 0 for higher throughput(striping data for parallel r/w)
    Fast network (10 GbE+, ideally RDMA-capable)

Object storage = Store data as "objects" (files + metadata) in a flat namespace, accessed via HTTP APIs (not filesystem paths).

TRADITIONAL FILESYSTEM:                    OBJECT STORAGE (S3):

Hierarchical (folders/paths):              Flat namespace (buckets + keys):

/home/                                     Bucket: my-training-data
  /user/                                   +---------------------------+
    /data/                                 | Key (object name)  | Data |
      image1.jpg                           +---------------------------+
      image2.jpg                           | image1.jpg         | ...  |
/mnt/                                      | image2.jpg         | ...  |
  /datasets/                               | model-v1.pt        | ...  |
    model-v1.pt                            +---------------------------+

Access: /home/user/data/image1.jpg         Access: GET my-bucket/image1.jpg
        (filesystem path)                          (HTTP API call)

+------------------+------------------------+---------------------------+
| Aspect           | Filesystem (NFS/local) | Object Storage (S3)       |
+------------------+------------------------+---------------------------+
| Access method    | POSIX (open/read/seek) | HTTP API (GET/PUT)        |
| Structure        | Hierarchical (folders) | Flat (bucket + key)       |
| Latency          | Low (us-ms)            | Higher (10s-100s ms)      |
| Throughput       | Fast (local/LAN)       | Slower (internet/WAN)     |
| Scalability      | Limited                | Virtually unlimited       |
| Cost             | Higher (SSD/NVMe)      | Very cheap ($/GB)         |
| Durability       | Depends on setup       | 99.999999999% (11 nines)  |
| Best for         | Active training data   | Long-term storage, backup |
+------------------+------------------------+---------------------------+

OBJECT STORAGE (S3) FOR AI - KEY POINTS:

1. WHAT: Flat storage via HTTP API, not filesystem
   - Cheap, scalable, durable
   - High latency per request (50-200ms)

2. PROBLEM: Slow for training if accessed directly
   - Each file = HTTP round-trip
   - No local caching by default

3. SOLUTIONS:
   a) STAGE LOCALLY: Download to NVMe before training
      - Tools: s5cmd, aws s3 cp (use parallel options!)
   
   b) CACHING LAYER: FSx for Lustre on top of S3
      - Transparent filesystem interface
      - Auto-caches hot data
   
   c) PARALLEL TOOLS: s5cmd, AWS S3 C++ SDK
      - Multi-threaded, multi-connection
      - 10-100x faster than naive sequential

4. RULE: Never read directly from S3 in training loop
   - Stage or cache first, then train

- Parallel fs: GPFS, Lustre -> designed for high concurrency and throughput 
- Lustre has multiple Object Storage Targets(OSTs)--> stripe your files across OSTs and use multiple OST's to multiply throughput. 

- Tunin, Replicating, and Compressing Data: 
- lfs setstrip to set strinping for a large dataset 
- lmt for Lustre monitoring tool -> check OST utilization, network throughput, etc.
- Generally, people just get the data into local storage eliminating remote data access. 
- Store compressed files on fs -> decompress them on the fly during training
- GPU specialized Decompression Engine: supports decoding of formats such as LZ4, Snappy, and Deflate
- nvJPEG decodes images on GPU
- Decompression time < I/0 time -> no bottleneck
- Due to the high CPU-GPU NVLInk-C2CC interconnect(upto 900 GB/s) bidirectional bandwidth -> shift data to GPU and decompress there simultaneously with kernel computations. 

Monitoring Storage I/O: 

- Vendor specific tools to monitor queues, latencies, read-ahead effects, and cache hit ratios. 
- NAS = A dedicated storage device on your network that serves files to multiple clients.
- Linux Tools for Storage Monitoring:
+-------------+----------------------------------------------+
| Tool        | What to Check                                |
+-------------+----------------------------------------------+
| iostat      | Disk throughput, IOPS, utilization, latency  |
| iotop       | Which process is doing I/O                   |
| nvme-cli    | NVMe health, queue depth, SMART data         |
| perf        | Trace I/O syscalls                           |
| eBPF        | Custom latency histograms, deep tracing      |
+-------------+----------------------------------------------+
- NVIDIA GPU-Specific Tools
+------------------+----------------------------------------------+
| Tool             | What to Check                                |
+------------------+----------------------------------------------+
| Nsight Systems   | I/O wait times, overlap with GPU kernels     |
|   --trace=gds    | cuFile API activity on timeline              |
+------------------+----------------------------------------------+
| DCGM             | GPU I/O statistics, GPU starvation detection |
+------------------+----------------------------------------------+
| /etc/cufile.json | Enable GDS tracepoints for Nsight            |
+------------------+----------------------------------------------+

QUESTION: Is GPU waiting for data? Where's the bottleneck?

Step 1: Measure total GPU idle time
        -> time how long next(data_iterator) takes

Step 2: Isolate DataLoader/Python cost
        -> set num_workers=0 (disables prefetch)-- disables parallel optimizations based latency hiding 
        -> time the iterator pull
        -> shows pure Python + transforms cost

Step 3: Isolate Host-to-Device copy cost
        -> wrap .to("cuda") with torch.cuda.Event timers
        -> OR check "Copy" lanes in Nsight Systems
        -> shows H2D transfer overhead

Step 4: Compare timings
        -> If Python cost high: add workers, simplify transforms
        -> If H2D copy high: use pin_memory, GDS, faster interconnect

"Keep the data pipeline full"

Monitor every stage:
  Disk -> [check] -> Network -> [check] -> CPU -> [check] -> GPU

Small inefficiencies add up:
  - 5ms latency here
  - 10MB too-small buffer there
  = Big impact at scale

DATA SOURCES                    DESTINATION
============                    ===========

1. LOCAL STORAGE (NVMe SSD)  ──────────────────────────►  GPU
   (same server/rack)

2. NETWORK STORAGE (NAS/3FS) ──────────────────────────►  GPU
   (different server/rack)

3. OTHER GPUs                ──────────────────────────►  GPU
   (gradients, activations)

+================================================================================+
|                        PATH 1: LOCAL STORAGE -> GPU                            |
|                        (Training data on local NVMe SSD)                       |
+================================================================================+

TRADITIONAL (CPU in hot path):
┌───────────┐      ┌───────────┐      ┌───────────┐
│  NVMe SSD │ ──── │    CPU    │ ──── │    GPU    │
└───────────┘ PCIe └───────────┘ PCIe └───────────┘
                   copies data!

WITH GDS (CPU orchestrates only):
┌───────────┐                         ┌───────────┐
│  NVMe SSD │ ─────── PCIe DMA ────── │    GPU    │
└───────────┘                         └───────────┘
       CPU says "go!" but doesn't touch data

+================================================================================+
|                        PATH 2: NETWORK STORAGE -> GPU                          |
|                        (Training data on remote storage - NAS, S3, 3FS)        |
+================================================================================+

TRADITIONAL (CPU in hot path):
┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐
│  Remote   │ ──── │    NIC    │ ──── │    CPU    │ ──── │    GPU    │
│  Storage  │      └───────────┘      └───────────┘      └───────────┘
└───────────┘                         copies data!

WITH GPUDirect RDMA (CPU orchestrates only):
┌───────────┐      ┌───────────┐                         ┌───────────┐
│  Remote   │ ──── │    NIC    │ ─────── PCIe DMA ────── │    GPU    │
│  Storage  │ RDMA └───────────┘                         └───────────┘
└───────────┘            CPU says "go!" but doesn't touch data

+================================================================================+
|                        PATH 3: GPU -> GPU                                      |
|                        (Gradients, activations during training)                |
+================================================================================+

WITHIN NVL72 RACK (NVLink):
┌───────────┐                         ┌───────────┐
│   GPU 0   │ ─────── NVLink ──────── │   GPU 1   │
└───────────┘       1.8 TB/s          └───────────┘
              No CPU involved at all!

ACROSS RACKS (InfiniBand + GPUDirect RDMA):
┌───────────┐      ┌───────────┐    InfiniBand    ┌───────────┐      ┌───────────┐
│   GPU 0   │ ──── │    NIC    │ ──────────────── │    NIC    │ ──── │   GPU 1   │
└───────────┘ PCIe └───────────┘                  └───────────┘ PCIe └───────────┘
                          CPU orchestrates but doesn't touch data    

Tuning the Data Pipeline: 

- Typical data pipeline: 
Read data from storage -? decode/deseialize data like parsing JSON/decoding JPGs --> apply transformations like tokenizing text/cropping images --> collage data into batches || CPU intensive

- Techniques to opimize these: 
1. Use multiple PyTorch DataLoader workers (num_workers>0)
2. Processing like tokenizing should be vectorized or use C/C++/Rust bindings-never pure python code -- eg. Hugging Face Tokenizers library, TorchText
3. CPU-GPU overlap: By the time GPU is done with batch N, CPU should be ready with batch N+1. 
4. Perform transform operations on the whole batch instead of sequential per sample basis leveraging vectorization
5. Set ulimit(buffer size) high and then pin_memory=True in DataLoader to speed up host-to-device transfers

Two Different Uses of DMA
USE 1: DMA for H2D Copy (Traditional DataLoader)
================================================

CPU RAM (pinned) ──── DMA ────► GPU HBM
                      |
              GPU's DMA engine
              moves the data

Data still moves from CPU RAM to GPU.
But GPU hardware does it, not CPU.


USE 2: DMA for Storage->GPU (GDS)
=================================

NVMe SSD ──── DMA ────► GPU HBM
               |
        SSD's DMA engine
        writes directly to GPU

Data SKIPS CPU RAM entirely.
This eliminates the H2D copy.

WITHOUT pin_memory=True:
========================

┌──────────┐    CPU copy    ┌──────────┐    DMA      ┌──────────┐
│ Pageable │ ─────────────► │ Staging  │ ──────────► │   GPU    │
│   RAM    │    (slow!)     │ (pinned) │   (fast)    │   HBM    │
└──────────┘                └──────────┘             └──────────┘
     |                           |                        |
  batch in               temp buffer               final location
  regular RAM          (CUDA creates this)


WITH pin_memory=True:
=====================

┌──────────┐               DMA                ┌──────────┐
│ Pinned   │ ──────────────────────────────► │   GPU    │
│   RAM    │          (fast!)                 │   HBM    │
└──────────┘                                  └──────────┘
     |                                              |
  batch already                              final location
  in pinned RAM
  
ONE FEWER COPY!

- prefetch_factor = number of batches each worker loads ahead. Total queue size = workers × prefetch_factor. Bigger queue = GPU less likely to starve during slow I/O.
- Another common pitfall: hidden bottleneck in your pipeline(eg. debug logging/expensive CPU transforms)
- To catch these, profiler the DataLoader in isolation by timininng how long it takes to produce 100 batches with all downstream GPU work disabled

DataLoader's job: Disk → CPU RAM (that's it!)

H2D Transfer: CPU RAM → GPU Memory (separate step)

Isolated = "DataLoader sprinting alone" -- no GPU work, no H2D transfers -- usually very fast as CPU is free from kernel launch overhead too
Target = "GPU's ideal pace" -- forward compute 
Real Idle = "GPU's actual waiting time" -- difference between two forward computes includes H2D transfer time + DataLoader time

Isolated > Target?  → DataLoader too slow (fix CPU/disk)
Isolated < Target BUT Real Idle high?  → H2D transfer too slow (fix pin_memory)
Real Idle ≈ 0?  → You're GPU-bound (all good!)

Scaling Out Workers as You Scale Out Number of GPU's:
- When you scale up compute using multiple GPU's, you need to scale up data loading too.
- Always measure CPU Utilization, as the data input pipeline will become the bottleneck as GPU training accelerates. 

Multimodal Data Processing with NVIDIA DALI: 

Multimodal Data processing with NVIDIA DALI: 
- Data Loading Library(DALI) -> accelerates pre/processing of of complex/heavy data by moving it to the GPU/using optimized C++ code. 
- For image/vid-> decoding/augmentation --> leverages NVIDIA's media acceleration hardware. 
- For eg., CPU utilization can be cut down from 800%(8x100% 1xCPU utliilzation) -> (2000%-> 2x CPU utilization for just data I/o. while the gPU does the preprocessing steps)
- You can also use GPU-friendly pre-processing operations and fuse them directly into GPU-based preprocessing computation graph --> Like the preprocessing steps go into the PyTorch/Tensorflow models directly. 
- Benchmark to compare CPU-only pipeline/DALI-enabled/fully fused GPU-graph. 

Creating High Quality LLM Datasets with NVIDIA Nemo Curator: 

┌─────────────────────────────────────────────────────────────────┐
│                     OFFLINE (Curator)                           │
│  Raw Data → Clean → Dedupe → Tokenize → Pack → .bin/.idx        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ONLINE (Training)                           │
│           Just read + light shuffle → GPU go brrrr              │
└─────────────────────────────────────────────────────────────────┘

Clean OFFLINE, train ONLINE — GPUs should compute, not clean

#   Core Concept        One-Liner
-   ---------------     ------------------------------------------
1   Purpose             Curator = Janitor for terabyte datasets
2   Scaling             Multi-GPU preprocessing -> don't wait days
3   Packing             Million small files -> few big binaries
4   Synthetic data      Human data scarce? Generate more
5   Quality filters     Dedupe + remove junk = cleaner model
6   Offline vs Online   Do the hard work BEFORE training starts
7   Goal                Training = GPU math, NOT string wrangling
8   File format         .bin + .idx = memory-mapped, fast reads

- Say for N epochs, you want N shuffles of the data --> in the preprocessing, it is not a bad idea to just store like N different copies --> not a crazy idea apparently lol 
- Always preprocess your data well so that GPU is focussed only on compute 
- NeMo's data loading still runs on CPUs. To bypass CPU I/O, you need to integrate it with tools like GDS. 
- Optimiizing data pipelines can yield better performance gains than algorithmic optimizations. 

Continuous Profiling and Tuning Workflow: 
- Metric: samples/sec 
- Steps for profiling: 
1. Establish a baseline: 
- Measure throughput(samples/sec), inference latency(ms)
    - Start with 1 GPU, multiple GPU's in a node, multi-nodes 
    - Check how the metrics are scaling 
2. Profile the multi-GPU run for bottlenecks: 
- NSight Systems --> Is GPU stalling frequently? Check for all kinds of bottlenecks:
- Is the main process lagging behind the other worker processes?
- Synchronization points where every thread is waiting?
- If GPU is idle during all-reduce, communication is the bottleneck, if idle at start, data loading is bottleneck
3. Zoom into specific kernels if needed: 
- NSight Compute to identify if individual kernels are network-bandwidth bound/memory-bandwidth bound/compute bound. 
4. Identify the cause: 
- Take hypothesis -> validate/invalidate them using profiling tools. 
- If memory bound, Fuse kernels = skip the HBM round-trip; smaller batch = less luggage to carry
5. Apply fixes or optimizations: 
- Based on all the information gathered so far. 

- Use topoplogy aware comm algorithms 
- If GPUs are spread across PCIe switches, bind the job to a single NUMA node. 
6. Remeasure after every change
7. Keep software updated, but always verify
8. Leverage modern hardware features 
9. Automate monitoring in production--> Kubernetes and other job schedulers integrate well with monitoring tools 
10. Document and educate--> share configuration learnings to team 

- Combine Nsight systems for a high-level system view, Nsight Compute for low-level GPU kernel profiles, and logging for NCCL and PyTorch. 

Diagnosing Communication-Versus Compute-Bound Workloads: 
- Change the ratio of communication to computation -- check how this affects the achieved network throughput measured in GB/s on the NIC. 
- Fix comms, reduce compute - if throughput is still the same - network is bottleneck as more work being produced is not resulting in higher throughput. 
- One way to do is to increase/decrease batch size
- Very good advice - RTB
- Watch 2 things: 
    - Absolute GB/s on the NIC
    - Relative time spent in communication relative to computation
- Use Nsight Systems to get an end-to-end timeline. If you see GPUs idle, waiting on data in form of long gaps between compute kernels corresponding to NCCL wait, then you're most likely communication bound. If the GPUs are busy but not reaching expected FLOPS, you are likely memory bound or compute bound. NSight Compute and PyTorch profiler can help determine the kernel's memory and compute efficiency. 





 


