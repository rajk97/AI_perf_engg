Occupancy TUning, Warp Efficiency, and Instruction Level Parallelism

2/11/26: 
- NSight Systems: Timeline View: Helps pinpoint concurrency issues, transfer overhead, and idle periods. 
- Use NVTX markers to annotate code sections for better visibility in the timeline --> like Tracy zones
- CPU is preparing data while GPU is idle: Dataloading pipeline needs imporvement -- tune the no. of data loader threads, overlap CPU preprocessing with GPU compute using double buffering, or move more preprocessing onto the GPU. 
- NSight Compute: Collects metrics for individual kernels -- achieved occupancy, issued warp instructions per cycle, memory throughput(GB/s), utilization of execution units, and many others.
- Goal: touch the roofline at YOUR algorithm's natural AI — higher AI ≠ better, just different workload

- PyTorch Profiler: collects trace data, Kineto lib under the hood. 
- PyTorch Profiler: torch.profiler.profile(with_flops=True, profile_memory=True) — estimates FLOPs and memory at operator level (not hardware counters)
- Nsight Systems: Now supports Python backtrace sampling + PyTorch domain — correlates Python code with GPU timeline
- Nsight Compute: Correlates to CUDA/PTX/SASS source — compile with -lineinfo for source line mapping
- Bridge Python → GPU: Use torch.cuda.nvtx ranges to link model python code to kernel-level profiling in NSight compute/systems.
- Key shift: Nsight tools now support Python/PyTorch natively — no longer just for CUDA C++ devs

Analyzing Warp Stall Reasons with Nsight Compute: 
- Warp stall reasons: memory-related, execution dependency, execution contention, and others like texture-cache-related stalls. 

Memory-related stalls:
- Long Scoreboard Stalls: Warps waiting on high latency global DRAM loads(400-800 cycles).Registers waiting for data to arrive through the long DRAM - to - register path. 
- Short Scoreboard Stalls: Warps waiting on shared memory loads(20-30 cycles): Registers waiting for data to arrive through the short shared memory - to - register path.

- Memory Transaction Slots: Fixed number of in-flight memory requests the SM can track simultaneously
- Memory Throttle: All slots full → warp can't issue new load → stalls
- Not Selected: Warp is ready but scheduler chose another warp — completely independent reason

- Execution dependency stalls: Due to instruction dependencies, e.g., waiting for a previous instruction -- maybe poor use of instruction level parallelism (ILP) or register reuse.
- Execution Unit Contention:
    - Execution units are saturated basically
    - Stall: Math pipe throttle: Multiple warps ready to execute but competing for the same execution unit (e.g., ALU, SFU) -- "ALU pipe busy"
- If there are no stalls and near 100% pipeline active, it's likely compute bound. 

Inspecting Acheived Occupancy and GPU Utilization: 
- Achieved Occupancy: Ratio of active warps to max warps per SM(~64).
- NSight Compute also reports occupancy limiters: Eg. Occupancy is limited by max registers per thread/ limited by shared memory per block/ limited by block size.
- Low occupancy is bad as it is bad at latency hiding, but high occupancy doesn't guarantee good performance if there are other bottlenecks (e.g., memory bound, execution bound).
Nsight
- Compute reports metrics such as achieved memory bandwidth in GB/s, achieved FLOPS in TFLOPS, instructions per cycle (IPC), issue-slot utilization, and other resource utilization statistics. These numbers will show how close your kernel is to
the GPU’s physical hardware limits.
- Roofline model: Plots performance (e.g., GFLOPS) against arithmetic intensity (FLOPs per byte of memory access). 

- High memory throughput + Low ALU: Memory-bound — optimize data access
- High ALU + Low memory throughput: Compute-bound — use lower precision (FP16/FP8/FP4) + Tensor Cores
- Low ALU + Low memory throughput: Latency-bound — not enough parallelism, threads, or ILP
- Trend: Each GPU gen: compute grows fast, memory grows slow → kernels become MORE memory-bound over time

┌─────────────────────────────────────────────────────────────────┐
│  KEY LESSON FROM Q4:                                            │
│                                                                 │
│  Don't compare to PEAK COMPUTE.                                 │
│  Compare to YOUR roofline at YOUR AI.                           │
│                                                                 │
│  AI=2, Performance=16T → ON roofline → OPTIMAL ✅                │
│  AI=2, Performance=8T  → BELOW roofline → FIXABLE ❌             │
│  AI=2, Performance=80T → IMPOSSIBLE ⛔                           │
│                                                                 │
│  If you want 80T, change the ALGORITHM to have higher AI.       │
└─────────────────────────────────────────────────────────────────┘

2/13/26:
Kernel Memory Throughput vs. Peak HBM Memory Bandwidth:
- Bandwidth util >80%: Memory-bound — more compute won't help, reduce memory traffic instead
- Increase AI by: Lower precision (FP16/FP8), Tensor Cores, kernel fusion (fewer intermediates)
- Blackwell's advantage: 126 MB L2 cache + 10 TB/s inter-die NV-HBI — previously memory-bound kernels may now be compute-bound
- Check L2 hit rate: High L2 hits = kernel may actually be compute-limited (cache is absorbing memory pressure)
- L2 persistence (Blackwell): Pin critical data in L2 to keep it resident

Kernel Compute Throughput vs. Peak GPU FLOPS: 
- Low achieved FLOPS can be caused by low occupancy(not enough warps to do latency hiding) or instruction-level stalls. 
- You can use NSight Compute's Occupancy section and Source Counters to pinpoint these issues. 
- Acheived occupancy is measured per-SM -- warps in same SM only can share resources
- Exec dependency stalls: Due to instruction dependencies
- Power management can also throttle compute performance
- Other tools for power usage: 
    - CUDA C++ nvmlDeviceGetPowerUsage(), nvmlDeviceGetEnforcedPowerLimit() API's
    - NVML Python API's

Iteratively Profiling and Determining the Kernel Bottleneck: 
- 4 fundamental bottlenecks of GPU's: 
    1. underutilization 
    2. latency-bound
    3. memory-bound
    4. compute-bound

Underutilized → Latency-Bound → Memory-Bound → Compute-Bound
     │                │               │               │
     ▼                ▼               ▼               ▼
  Not enough      Warps stall     Bandwidth        ALUs are
  threads/work    waiting for     saturated,       the limit
                  data            ALUs idle

Bottleneck	    Symptom	                                                Fix
Underutilized	Low FLOPS + Low bandwidth + Idle gaps	                Launch more threads/work
Latency-Bound	Low bandwidth (not saturated), warps stalling on loads	Increase occupancy, ILP, prefetching, pipelining
Memory-Bound	Bandwidth saturated (~80%+), ALUs idle	                Tiling, fusion, caching, lower precision
Compute-Bound	ALUs maxed out, bandwidth has headroom	                ILP, unrolling, better instruction mix, Tensor Cores

INT32 + FP32 can't run same cycle on unified cores — instruction mix matters for compute-bound kernels

2/14/26:
- Memory-bound = DRAM bandwidth saturated (shared by all SMs) — individual SM pipes are fast enough, the shared source (HBM) is the bottleneck

GPU KERNEL BOTTLENECK REFERENCE TABLE
=====================================

┌──────────────────┬───────────────────────────────────┬──────────────────────────────────┬──────────────────────────────────────┐
│ Limiting Factor  │ Description                       │ Profiler Indicators              │ Remedies                             │
├──────────────────┼───────────────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┤
│                  │                                   │                                  │                                      │
│ Memory Bound     │ Moving as much data as you can    │ High memory-bandwidth            │ Increase arithmetic intensity         │
│                  │ — close to peak DRAM bandwidth    │ utilization is near peak,        │ (tiling, fusion) and improve          │
│                  │ — but not enough work per byte    │ low FLOPS.                       │ coalescing and caching.(Reduce loading waste memory too)               │
│                  │ to fully utilize ALUs.            │                                  │                                      │
│                  │                                   │                                  │                                      │
├──────────────────┼───────────────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┤
│                  │                                   │                                  │                                      │
│ Compute Bound    │ Hidden memory latency, no longer  │ High FLOPS approaching GPU       │ Exploit more ILP (dual-issue,         │
│                  │ saturating memory bandwidth.      │ peak, low memory utilization.     │ loop unroll), use specialized         │
│                  │ ALUs (CUDA cores + Tensor Cores)  │                                  │ units (FP16/FP8/FP4/Tensor Cores),   │
│                  │ are the bottleneck.               │                                  │ reduce dependencies, fuse work,      │
│                  │                                   │                                  │ leverage lower precision/sparsity.   │
│                  │                                   │                                  │                                      │
├──────────────────┼───────────────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┤
│                  │                                   │                                  │                                      │
│ Latency Bound    │ Not sustaining enough concurrent  │ Low achieved bandwidth well      │ Raise occupancy, add ILP (unroll,     │
│                  │ work to hide individual load/     │ below peak, high "stall-on-      │ multiple accumulators), intra-kernel  │
│                  │ store latencies, so warps stall   │ scoreboard" or "not selected"    │ pipelining, and software prefetch.    │
│                  │ waiting on data.                  │ percentages.                     │                                      │
│                  │                                   │                                  │                                      │
├──────────────────┼───────────────────────────────────┼──────────────────────────────────┼──────────────────────────────────────┤
│                  │                                   │                                  │                                      │
│ Underutilizing   │ Not fully occupying SMs or        │ Low occupancy and low achieved   │ Increase problem size or batch work,  │
│ the GPU          │ launching enough work — both      │ bandwidth, low FLOPS, timeline   │ launch more threads/blocks, fuse      │
│                  │ memory and compute resources      │ shows gaps or sparse kernel      │ tasks, use persistent kernels or      │
│                  │ remain idle.                      │ activity.                        │ streams.                             │
│                  │                                   │                                  │                                      │
└──────────────────┴───────────────────────────────────┴──────────────────────────────────┴──────────────────────────────────────┘

OPTIMIZATION ORDER:
═══════════════════
Underutilized → Latency Bound → Memory Bound → Compute Bound

"Fix one bottleneck, reveal the next"

QUICK DIAGNOSIS (2x2 Grid):
═══════════════════════════

                    LOW Memory BW Util       HIGH Memory BW Util
                 ┌──────────────────────┬──────────────────────┐
  LOW FLOPS      │  Underutilized OR    │  Memory Bound        │
                 │  Latency Bound       │                      │
                 ├──────────────────────┼──────────────────────┤
  HIGH FLOPS     │  Compute Bound       │  Optimal! (both      │
                 │                      │  near peak)          │
                 └──────────────────────┴──────────────────────┘

Tuning Occupancy: 
- While maximizing occupancy ensures that many warps are availabe to run, those warps might still be idle if waiting on memory or if executing divergent code. Therefor, after achieving a reasonable occupancy (e.g., 50-70%), focus on warp efficiency and latency-hiding rather than obsessing over 100% occupancy. 
- Blackwell has 255 registers per thread limit. 
- How to tune occupancy:
    - Increase no. of threads
    - Be mindful of the resource usage
    - __launch_bounds__ = you tell the compiler your TUNED launch config so it budgets registers to make that occupancy actually fit. 
    - cudaOccupancyMaxPotentialBlockSize() = asks runtime 'what block size gives max occupancy for this kernel's register/shmem usage?'
    - cudaOccupancyMaxActiveBlocksPerMultiprocessor() = asks runtime 'how many blocks of this config can fit on one SM?'
    - Occupancy API answers at runtime (knows actual resource usage), __launch_bounds__ instructs at compile time (shapes resource usage).

- CUDA Graphs = record a kernel sequence once, replay with near-zero launch overhead → GPU stays busy between kernels instead of waiting for CPU.

__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor): Compiler optimizes register usage, 

- PyTorch handles occupancy tuning internally — built-in kernels already pick optimal block sizes and launch configs.
- Small tensors → few threads → GPU underutilized. Fix: batch operations or fuse with torch.compile.
- torch.compile merges many small ops into one big fused kernel → more work per launch → better occupancy.
- torch.compile(mode='max-autotune') for stable shapes, mode='reduce-overhead' for small-batch loops — both enable CUDA Graphs when profitable.
- Use fused optimizers (AdamW(fused=True)) + AMP to reduce per-op overhead.
- Only write custom CUDA kernels if built-in ops aren't enough — if you do, same occupancy rules apply.

Improving Warp Execution Efficiecy: 
- Warp inefficiencies: 
    1. Warp divergence: When threads within a warp do not all do the same work
    2. Threads are waiting on data loads

- To improve warp efficiency: 
    - Improve memory coalescing
    - MInimize thread divergence
    - Use warp-level intrinsics for optimal intrawarp communication 

- Speeds are only 5-30% but important 

Intra-warp divergence cause: 
- If/else: In SIMT, the warp must exeute both paths serially: when the if threads are executing, other threads are idle, and vice versa. 

Techniques to avoid warp divergence: 

1. Restructure Conditions: 
    Three strategies:
        - Sort/partition data → homogeneous warps (no divergence)
        - Lift condition out of inner loop → branch once, not every iteration
        - Separate kernels → kernel A handles negatives, kernel B handles positives
2. Separate into multiple kernels: 

    - Separate kernels = keep data in place, use prefix sum to find which elements belong to which category, then launch a dedicated kernel per category
    - It's an alternative to sorting — instead of reorganizing data, you reorganize work across kernels
    - Cost: extra prefix sum computation + multiple kernel launches
    - Benefit: no divergence within any kernel, original data order preserved
    - Worth it only when divergent code paths are large (many instructions each)   
3. Rewrite conditions to be warp-unanimous: 
    - More like all OR none -- all threads in the warp execute an instruction or not 
    - Split work by warp ID, not thread data — warps agree, threads don't diverge.
4. Utilize warp-vote parallel algorithms: 
    - LANE: Thread within a warp
    - Lane = thread's position (0-31) within a warp — just another name for thread-in-warp
    - __ballot_sync = every lane votes, returns 32-bit mask of who said yes
    - __any_sync = "did anyone vote yes?" — quick check to skip work entirely
    - __all_sync = "did everyone vote yes?" — quick check for unanimous path
    - Voting lets warp COLLECTIVELY decide, then compact work into fewer lanes
    - Avoids divergence by converting per-lane branching into lane-count-based execution
    - Tradeoff: eliminates divergence but may cause load imbalance (few lanes do all work) -- not clear how it eliminates divergence -- will need to wait for when it comes up again 
5. Predicate short lines: 
    - Predication = compiler runs BOTH paths for ALL threads, SELECTs correct result per thread — avoids branch management overhead (~3-5 cycles) at the cost of computing both paths
    - IMP: Worth it only for SHORT branches where overhead > compute; for long branches the savings are negligible
    - Encourage with ternary (?:) or arithmetic masking (cond * f(x) + (1-cond) * g(x))

Profiling and Detecting Warp Divergence: 

- "Warp execution efficiency" = profiler metric showing % of active threads per warp (e.g., 30% = only ~10 of 32 lanes doing useful work)
- "Predicated-off instructions" = profiler counts instructions that were issued but masked — look for inflated dynamic instruction count vs actual data processed
- "Goodput" = effective throughput after subtracting wasted masked-lane work

Using Predication to Minimize Divergence: 

- Predication eliminates DIVERGENCE (no serialized paths, no branch overhead) but NOT compute waste
- Every lane still executes both sides — one result is thrown away per lane
- The only true zero-waste solution: hardware native ops (max, min, abs) that compute the answer in 1 instruction
- Predication vs divergence = same wasted compute, different packaging overhead
- Overhead is supposedly lower in predicattion compared to warp divergence. 

- GPU ALUs have native hardware for max, min, abs, clamp, saturate — no branching needed at all
- torch.maximum(X, 0) compiles to a SINGLE fmax instruction — not predication, not branching, just one ALU op
- PyTorch's "vectorized operations" use these native instructions — that's why they're divergence-free
- The "special hardware" = dedicated circuits in the ALU for common math ops that would otherwise branch
- Mnemonic: "max is a circuit, not a choice — hardware solved it before software had to."

- Warp divergence = lanes execute instructions they don't need → ALU cycles consumed with no useful output = lost goodput
- Goodput = useful work done / total work done — divergence inflates the denominator

- Warp execution efficiency = average % of active (non-masked) lanes per instruction across all warps
- Formula: (total active lane-instructions) / (total issued lane-slots) × 100

HARDWARE REALITY:
  All 32 lanes physically run the instruction through the ALU
  ← You're correct about this!

PROFILER'S DEFINITION:
  "Active" = lane whose result is COMMITTED (written to register)
  "Inactive" = lane whose result is DISCARDED (masked off)
  
  The profiler doesn't care if the ALU computed something.
  It only counts lanes that PRODUCED useful output.

Exposing Instruction-Level Parallelism: 

- ILP = overlap independent instructions to fill idle cycles (higher IPC, not more FLOPs)
- GPU is in-order: developer unrolls, compiler schedules across parallel units (FP, INT, LD/ST, SFU, Tensor)
- CPU has hardware OOO: finds ILP automatically — GPU traded that for more cores + less power
- ILP hides latency WITHIN a warp; occupancy hides latency ACROSS warps — complementary
- Only helps if idle slots exist — useless when already compute-bound
- Tools: #pragma unroll, -maxrregcount (more registers = more ILP, less occupancy)

Warp Scheduling and Dual Issue Instructions: 





