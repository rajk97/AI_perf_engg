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