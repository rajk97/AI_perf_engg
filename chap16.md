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
  └───────────────────┴──────────────────────────────────────────────┘
  - Long all-to-all blocks with idle SMs → communication bottleneck