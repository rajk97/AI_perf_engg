Profiling, Debugging, and Tuning Inference at Scale

4/15/26

Chapter overview

- Profiling and debugging: Nsight Systems, Prometheus/Grafana, key metrics
- Operational tuning: GPU utilization, latency reduction, throughput scaling
- Quantization: GPTQ, AWQ, weight-only vs weight+activation, 8-bit/4-bit
- Application-level: prompt compression, prefix caching, dedup, query routing, streaming

Profiling, debugging, and tuning inference performance

- Disaggregated prefill/decode has many moving parts → tuning is iterative

- Observe-hypothesize-tune loop:
  1. Observe metrics → identify bottleneck (low GPU util, high latency)
  2. Hypothesize fix (e.g. increase batch size, overlap comms)
  3. Implement + test in staging with representative workload + profiling
  4. Deploy to production + monitor Grafana/logs to validate improvement
  5. Repeat as new bottlenecks appear

- Use canary rollouts for production optimizations
  - Deploy to small subset of traffic first → catch side effects early → reduce blast radius

- Example bottleneck: CPU at 100% from tokenization/preprocessing
  - Fix: move preprocessing to GPU (GPU-accelerated tokenizer or custom CUDA/Triton kernel)
  - Validate: CPU util drops + throughput increases = no longer CPU-bound

Cache monitoring

- Track hit/miss rates for ALL caches: prefix cache, prompt-embedding cache, KV cache
- High miss rate → tune cache size, eviction policy (LRU, LFU), or caching strategy
- vLLM LMCache: adjustable GPU vs CPU cache ratio, paged cache offload to CPU if GPU memory limited

- Prefix cache specifically:
  - Reuses KV for identical input-sequence prefixes across batched requests
  - vLLM metrics: vllm:gpu_prefix_cache_queries, vllm:gpu_prefix_cache_hits
  - Hit rate = hits / queries
  - Prefix merge rate correlates with actual cache benefit → adjust batching to maximize shared prefixes
  - If prefix matching fails → check for tokenizer differences first (most common cause)

- If using NVMe-based KV cache extension → monitor device I/O latency (high I/O latency kills cache perf)

Capacity planning and scaling decisions

- Track utilization + latency vs load → project when system hits limits (e.g. p95 latency rising exponentially)
- When batch-size increases stop helping (per-GPU concurrency saturated):
  - Scale out: more GPUs, more model replicas, more experts
  - Or compress first: FP8/FP4 for more effective throughput per GPU before adding hardware
- Once SMs at 100% + memory bandwidth at peak → adding GPUs or TP/PP is the only path
- Increasing expert count raises throughput ceiling only if routing/scheduling also improved
  - Otherwise bottleneck just shifts to network
- Always weigh cost of new hardware vs efficiency gains
  - Sometimes newer GPUs (more memory, more FLOPS) are cheaper than scaling out old ones

Monitoring system metrics and counters

- LLM requests are nonuniform (variable length, variable compute) unlike traditional microservices
  - Can't predict latency from request alone → need continuous monitoring

- Stack: Prometheus (scrape + collect) → Grafana (visualize + alert)

- GPU metrics via DCGM (Data Center GPU Manager):
  - DCGM_FI_DEV_GPU_UTIL: SM utilization %
  - DCGM_FI_DEV_MEM_COPY_UTIL: memory copy engine utilization
  - DCGM_FI_DEV_FB_USED: framebuffer memory used
  - Also: GPU temperature, power (throttling detection), Xid error counters
  - cudaMemPool metrics for memory fragmentation monitoring

- Low-level counters (L1/L2 activity, occupancy, instruction throughput):
  - Collected via Nsight Compute or CUPTI, not DCGM

- Interconnect metrics:
  - NVLink/NVSwitch bandwidth, NIC throughput
  - DCGM exposes NVLink error counters but per-link bandwidth may require direct DCGM query
  - Also use nvidia-smi nvlink and Nsight tools for sustained link utilization
  - Alert on saturation of cross-GPU and cross-node communication

- Application-level metrics to track:
  - Requests/sec, avg latency, p95/p99 latency
  - Tokens/sec throughput
  - Active contexts count
  - KV cache utilization and size (overall + per node)
  - Batch size changes (use counters, not log searching)

- Counters > log searching:
  - Increment Prometheus counter for app-level events (batch size changes, errors, etc.)
  - Instantly viewable in Grafana alongside GPU metrics in real time
  - Log-based analysis requires slow offline aggregation (Spark) + manual correlation

- Structured logging + distributed tracing (OpenTelemetry / APM tools):
  - Correlate logs/traces with metrics → consistent timeline across entire system
  - Speeds up debugging significantly

- Dynamic batching: inference servers expose "maximum latency" setting
  - Increasing it → larger batches → higher throughput
  - Too much → p99 latency exceeds SLO
  - Continuously tune against latency targets

You only ever have 3 questions when something is slow:

  1. IS THE GPU BUSY?
     └→ SM util %  (DCGM)

  2. IS THE GPU FED?
     └→ Memory BW, cache hit rate, NVLink/NIC throughput

  3. IS THE USER HAPPY?
     └→ p99 latency, tokens/sec


  That's it. Everything else is just WHERE to look:

  ┌─────────────────────┬────────────────────────┐
  │ Question            │ If answer is NO        │
  ├─────────────────────┼────────────────────────┤
  │ GPU busy?           │ Increase batch size    │
  │                     │ or overlap more        │
  ├─────────────────────┼────────────────────────┤
  │ GPU fed?            │ Fix cache misses,      │
  │                     │ fix interconnect,      │
  │                     │ fix memory pressure    │
  ├─────────────────────┼────────────────────────┤
  │ User happy?         │ Something above broke  │
  │                     │ OR batch too big (p99) │
  └─────────────────────┴────────────────────────┘


  The tooling is just plumbing:

  DCGM         = "give me GPU numbers"
  Prometheus   = "store them"
  Grafana      = "show me pretty graphs"
  Nsight       = "deep dive when graphs look bad"
  OpenTelemetry= "trace a single request end to end"  

Debugging KV cache swapping

- When GPU memory is near max → inference engine swaps KV cache to CPU or NVMe
- Symptom: high copy-engine util + low SM util at the same time = swapping
  - Also: abnormal NVLink util that aligns with GPU idle periods

- Fixes:
  - Tune paging parameters to reduce thrashing
  - Apply FP8/FP4 quantization (smaller KV cache)
  - Increase GPU memory allocation for cache
  - Change swapping strategy
  - Goal: copy util down, compute util up

Correlating latency spikes

- p99 spike? Overlay RPS (requests/sec) on latency graph in Grafana
  - If spike correlates with traffic surge → dynamic batch size grew too large
  - Fix: decrease max request-batch queue delay or cap max batch size
- Combine log timeline (DEBUG level, enable/disable as needed) with Prometheus metrics
  - Logs: step-by-step events (batch formed, comm start/end)
  - Metrics: aggregate view (how often all-to-all >5ms?)
  - Together: spot outliers like network degradation on one link

- MoE-specific: if all-to-all keeps spiking on one path
  - Raise capacity factor to 1.2-1.5 → excess tokens spill to secondary expert replica
  - Prefer replica on GPU with more stable network path
  - Better to spill than to queue behind a stalled expert

Profiling with Nsight Systems

- Nsight Systems: timeline view of CPU threads + GPU kernels + CUDA events + NCCL comms
  - Microsecond resolution, shows idle gaps and unexpected synchronizations

- NVTX annotations: label regions on the timeline for clarity
  - Mark "Prefill", "Decode", "All-to-all" etc. with colored ranges
  - Use C API (nvtxRangePushEx / nvtxRangePop) for C++ runtimes
  - Scope tightly: wrap exactly the work, nothing else

- Per-stream NVTX ranges: when using multiple CUDA streams
  - Name streams: nvtxNameCudaStreamA(stream, "transfer_stream")
  - Wrap each stream's host code with distinct NVTX ranges
  - Timeline shows overlap between transfer and compute streams

- Key things to look for in Nsight Systems timeline:
  - GPU idle gaps between kernels → scheduling or sync issue
  - Prefill/decode overlap with communication → good (overlap working)
  - Long all-to-all blocks with idle SMs → communication bottleneck

- Nsight Compute: two modes
  - Full instrumentation (default): replays kernel with HW counters, exact counts, slow (5-20x replay)
  - PC Sampling: periodically snapshots program counter, statistical hotspots, fast + low overhead
  - Strategy: PC Sampling first on live server → find hotspot → full instrumentation on that one kernel

Inference troubleshooting recipes

- Production: no heavy profilers running continuously
- Rely on lightweight metric-based monitoring → detect anomaly → hypothesis → fix → verify

  ┌──────────────────────────┬──────────────────────────┬──────────────────────────────────┐
  │ Symptom                  │ Probable cause           │ Fix                              │
  ├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
  │ SM util < 50%            │ Small batches or         │ Increase batch size, enable      │
  │                          │ unfused kernels          │ FlashAttention / fused SDPA,     │
  │                          │                          │ custom Triton kernels            │
  ├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
  │ KV cache preemption      │ Insufficient KV cache    │ Raise GPU mem util threshold,    │
  │ warnings (vLLM)          │ space                    │ reduce max batched tokens,       │
  │                          │                          │ use PagedAttention               │
  ├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
  │ p95 > 200ms              │ Decode hotspot or        │ Check router logs, tune prefetch │
  │                          │ head-of-line blocking    │ threshold, enable spec decoding  │
  ├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
  │ Cache hit rate < 60%     │ Unbalanced shards or     │ Validate prefix cache config     │
  │                          │ missing prefix cache     │ (LMCache NIXL), increase TTL     │
  ├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
  │ Unexpected OOM           │ Overcommitted GPU mem    │ Lower per-instance mem util,     │
  │ (multitenant GPU)        │                          │ enable CPU/NVMe offload,         │
  │                          │                          │ pin to CPU socket                │
  ├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
  │ Irregular perf outliers  │ Mismatched clocks or     │ Sync clocks, monitor thermal/    │
  │                          │ thermal throttling       │ power throttling                 │
  └──────────────────────────┴──────────────────────────┴──────────────────────────────────┘

- Log lines to watch for:
  - vLLM: "preempted by PreemptionMode.RECOMPUTE because not enough KV cache space"
    → KV recompute triggered, wastes GPU compute, increases latency
  - NVIDIA Dynamo router: "prefix-cache hit (90%)" = good, local prefill
    → "cache miss; dispatching remote prefill to GPU-node-03" = miss, remote dispatch

Full-stack inference optimizations

- Every layer of the stack contributes, not just one:

  ┌───────────────────┬──────────────────────────────────────────────┐
  │ Layer             │ Techniques                                   │
  ├───────────────────┼──────────────────────────────────────────────┤
  │ Model             │ Pruning, distillation, sparsity, MoE,       │
  │                   │ FlashAttention, quantization-aware training  │
  ├───────────────────┼──────────────────────────────────────────────┤
  │ Kernel            │ Fused ops, FlashInfer, Tensor Cores,        │
  │                   │ block tiling, async memory transfers        │
  ├───────────────────┼──────────────────────────────────────────────┤
  │ Runtime           │ Dynamic batching, paged KV cache,           │
  │                   │ CUDA Graphs, compute-comm overlap           │
  ├───────────────────┼──────────────────────────────────────────────┤
  │ Orchestration     │ Prefill/decode disaggregation, routing,     │
  │                   │ multitenancy isolation, autoscaling          │
  ├───────────────────┼──────────────────────────────────────────────┤
  │ Deployment        │ Geo-distributed edge serving, smart API     │
  │                   │ gateway batching, CI/CD for model variants,  │
  │                   │ NVLink/NVSwitch + IB, NUMA affinity         │
  ├───────────────────┼──────────────────────────────────────────────┤
  │ QoS / scaling     │ SLA-aware dynamic batching, MIG/stream      │
  │                   │ priorities, real-time profiling dashboards,  │
  │                   │ dynamic parallelism switching (TP/PP/DP)    │
  └───────────────────┴──────────────────────────────────────────────┘

- Cross-layer synergy example:
  quantization (model) → smaller footprint → larger batch (runtime) → more requests merged (orchestration)

- Profiling-driven focus: after fusion + quantization, if CPU becomes bottleneck
  → faster tokenizers or offload preprocessing to GPU

- Complexity tradeoff: speculative decoding + Medusa = complex, reserve for extreme cases
  - Lighter methods (sparsity, batching, disaggregation) deliver bulk of production gains

- Keep stack up to date (CUDA, cuDNN, NCCL) — newer versions include latest optimizations

Debugging correctness issues

- Memory leak detection: if GPU memory climbs over time
  - Use Compute Sanitizer: compute-sanitizer --tool memcheck your_binary
  - Catches device memory errors, race conditions, out-of-bounds accesses

- NCCL failures:
  - Symptom: one GPU near 0% util while others at 90% → GPU dropped out of NCCL group
  - Check logs for NCCL_COMM_FAILURE warnings
  - Enable debug: NCCL_DEBUG=WARN (verbose!)
  - Use NCCL test suite for all-reduce / all-to-all correctness
  - ncclCommGetAsyncError + ncclCommAbort for async error handling
  - Fix: reinitialize NCCL communicators or full node reboot (rejoin group after)

- Prometheus alert rules to set up:

  ┌──────────────────────────┬────────────────────┬──────────┐
  │ Metric                   │ Condition          │ Severity │
  ├──────────────────────────┼────────────────────┼──────────┤
  │ GPU util                 │ < 10% for >60s     │ Idle     │
  │ GPU util                 │ > 90%              │ Bottlenk │
  │ Memory usage             │ > 80%              │ Warning  │
  │ Memory usage             │ > 95%              │ Critical │
  │ Temperature              │ > 85C              │ Warning  │
  │ Temperature              │ > 95C              │ Critical │
  │ NVLink replay/recovery   │ >= 1               │ Critical │
  │ NVLink CRC errors        │ > 100/sec          │ Critical │
  │ PCIe replay errors       │ >= 1               │ Critical │
  │ Uncorrectable ECC errors │ >= 1               │ Critical │
  └──────────────────────────┴────────────────────┴──────────┘

- Hardware dropouts: GPU or NVLink silently disconnects after fatal error
  - DCGM per-link NVLink metrics: DCGM_FI_DEV_NVLINK_TX/RX_BANDWIDTH_L*
  - Fallback: nvidia-smi nvlink, Nsight, NVSwitch counters
  - Long all-to-all blocks with idle SMs → communication bottleneck 

4/21/26:
- Catch NCCL errors -- scraped by FLuentd/AWS CloudWatch -- put them on top of the GPU utilization graphs in Grafana → correlate GPU idle periods with NCCL failures

Application-level optimizations:

Dynamic Batching/Scheduling/Routing

Dynamic batching: 

- Groups requests for a few ms to form batches, boosting throughput and GPU utilization
- Batching adds tiny delay at low load, but reduces overall and tail latency at high load
- Batch size and delay are tuned to meet p99 latency SLOs (e.g., 1–2 ms delay) and according to Requests per second (RPS) patterns
- Adaptive batching keeps latency flat as throughput rises, up to an inflection point

Continuous batching: 
- Continuous batching: at every token step, evict finished requests and admit waiting ones — GPU slots always full
- Result: 2-3x throughput on large models, minimal latency cost, ideal for chat/low-latency workloads

CONTINUOUS BATCHING vs CONTINUOUS SCHEDULING
════════════════════════════════════════════

  Continuous BATCHING:  "same kernel, many requests in one tensor"
    → works when sequences can share a matmul
    → evicts finished, admits new, at each token step

  Continuous SCHEDULING: "many kernels, many streams, GPU interleaves warps"
    → works when sequences CAN'T share a matmul (different shapes, padding waste)
    → launches 10 small decode kernels on 10 CUDA streams
    → GPU's warp scheduler interleaves them like an OS interleaves processes


WHY NEEDED — THE DECODE PROBLEM
────────────────────────────────
  Decode = generate 1 token at a time per request
  Small compute + heavy memory movement → GPU underutilized

  10 users decoding different-length sequences:
    Can we batch into one big matmul? NO — too much padding
    Can we run them sequentially?      NO — GPU sits idle between

  Solution: launch 10 small kernels on 10 streams → warp scheduler
            interleaves them → GPU stays busy


VISUAL
──────
  Sequential (bad):
    Stream 0: [k1]        [k2]        [k3]     ← GPU idle between

  Continuous scheduling (good):
    Stream 0: [k1─────────────]
    Stream 1:    [k1─────────────]
    Stream 2:       [k1─────────────]
    Stream 3:          [k1─────────────]
    → warp scheduler interleaves on SMs, no idle time


SUMMARY (one-liners)
────────────────────
- Continuous scheduling treats the GPU like an OS scheduler does a CPU — many small kernels on separate streams, warp scheduler interleaves them
- Used when decode workloads are too small/varied to batch into one matmul
- One kernel stalls on I/O → another proceeds → no idle cycles
- Production engines (vLLM, SGLang) combine continuous batching + continuous scheduling to kill GPU "bubbles"
- PagedAttention (vLLM): KV cache split into pages, group page compute across sequences
- RadixAttention (SGLang): tree-based KV cache + lazy eviction for unused pages

Chunked prefill (stall-free scheduling) — summary

- Split a very long prompt into smaller prefill chunks so decode can run in between instead of waiting behind one giant prefill step
- Chunking does not reduce total attention work; it only reshapes it into smaller scheduling units
- Benefit: lower per-iteration stall, smoother tail latency, better overlap between prefill and decode
- Cost intuition: same total attention compute, more predictable latency

Mnemonic: chunking helps latency scheduling, not compute complexity.

Latency-aware scheduling and dynamic routing — summary

- FIFO batches by arrival order; latency-aware scheduling batches by expected cost so one GPU does not get stuck with all the long prompts
- In the example, rebalancing prompt lengths cuts the critical path from 38M to 22M attention ops, so TTFT improves a lot
- Core idea: move shorter prompts earlier and spread long prompts across GPUs to balance prefill work
- Works best when attention kernels operate on true sequence lengths; padding-heavy kernels reduce the benefit

Mnemonic: do not batch by arrival order, batch by compute weight.

Systems-level optimizations:

Overlapping Communication and Computation

- Goal: keep GPUs doing useful work nearly 100% of the time by hiding communication behind computation
- Main technique: use separate CUDA streams, async copies, pinned host memory, and sync only when data is actually needed
- In distributed inference, overlap collectives too: chunk transfers so one chunk computes while the next chunk is being sent
- Extras: enable GPUDirect RDMA, NCCL group calls, and sometimes SHARP to reduce communication overhead further

Mnemonic: move data in the background, keep compute in the foreground.

Overlap patterns — summary

- Overlap at every boundary: GPU compute with NCCL collectives, CPU preprocessing with GPU work, pipeline-stage sends with next-stage compute, and token generation with network streaming
- Practical recipe: split work into chunks, launch comms asynchronously on separate streams/threads, and use CUDA events only at true dependency points
- GPUDirect RDMA helps cross-node transfers by letting NICs access GPU memory directly without host-memory staging, reducing CPU overhead and latency
- Core benefit: fewer GPU idle bubbles, smoother latency, and higher throughput with no change in model outputs

Mnemonic: if two things do not depend on each other, pipeline them.

Maximizing GPU utilization — throughput vs latency trade-off

- Target useful GPU utilization (goodput), not raw 100% — running at 100% can trigger thermal/power throttling
- Plot a throughput-vs-latency curve across batch sizes: there is always a "knee" where more throughput starts costing too much latency
- Rule of thumb: cap batch size at ~90% of peak throughput (headroom buffer) for predictable latency without meaningful throughput loss
- Monitor p95/p99 tail latency, not just p50 — in large clusters even 0.1% outliers are frequent
- Reducing tail latency has direct cost benefits: less overprovisioning needed to meet SLOs

Tuning recipe:
  1. Turn on full concurrency and overlap first
  2. Gradually increase batch size until resource utilization peaks
  3. Measure single-query latency at that point
  4. Scale batch size back just enough to meet p99 SLO

Mnemonic: aim for 90% utilization with predictable latency, not 100% with throttle-induced spikes.




