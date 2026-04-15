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