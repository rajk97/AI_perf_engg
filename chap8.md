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
