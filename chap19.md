Chapter 19 — Dynamic and adaptive inference engine optimizations

Why this chapter exists

- Static serving configs break under changing traffic, model size, and latency targets
- Modern inference engines adapt at runtime:
	- parallelism
	- precision
	- kernel scheduling
	- memory/cache usage
- Goal: maximize throughput while still meeting latency SLOs

Adaptive parallelism strategies

Core idea

- No single parallelism strategy wins for every workload
- Pick based on request length, concurrency, model size, and architecture

Traffic pattern → best parallelism

	┌────────────────────────────────┬──────────────────────────┬──────────────────────────────────────────┐
	│ Traffic pattern                │ Best strategy            │ Why                                      │
	├────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────┤
	│ many short requests            │ data parallel / replicas │ no inter-GPU sync, max independent RPS   │
	│ (<256 tokens, high RPS)        │                          │ if model fits on one GPU                 │
	├────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────┤
	│ few long requests              │ pipeline parallelism     │ split layers across GPUs, reduce per-    │
	│ (>=8k tokens, low concurrency) │ with microbatches        │ request latency for deep models          │
	├────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────┤
	│ mixed load                     │ hybrid dynamic switch    │ run short chats local, pipeline long     │
	│ (short + some long)            │                          │ ones to meet SLA                         │
	├────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────┤
	│ model too large for 1 GPU      │ tensor + pipeline hybrid │ must fit model while balancing compute   │
	│ (>1 GPU memory)                │                          │ and memory                               │
	├────────────────────────────────┼──────────────────────────┼──────────────────────────────────────────┤
	│ MoE inference                  │ expert parallelism       │ place experts across GPUs; route only to │
	│ (sparse expert selection)      │                          │ top-k experts per token                  │
	└────────────────────────────────┴──────────────────────────┴──────────────────────────────────────────┘

What each one means

- Data parallel / replicas:
	- full model copied to each GPU
	- requests spread across replicas
	- zero per-request inter-GPU sync
	- best for many small / medium requests when model fits

- Tensor parallelism (TP):
	- split matrices across GPUs
	- speeds big matmuls
	- cost = all-reduce / sync overhead

- Pipeline parallelism (PP):
	- split layers across GPUs
	- solves memory limits for deep models
	- cost = sequential stage delay, aka pipeline bubbles

- Expert parallelism (MoE):
	- experts live on different GPUs
	- gating network sends each token to top-k experts only
	- saves per-device memory/compute on sparse models

Visual

	short-chat flood      → replicas
	long-doc requests     → PP
	huge dense model      → TP + PP
	MoE sparse routing    → expert parallelism

Mnemonic: small requests want replicas, deep long requests want pipelines, giant dense models want TP+PP, and sparse MoE models want expert routing.

Adaptive TP/PP switching at runtime

Expert parallelism reminder

- MoE activates only a few experts per token
- Example: DeepSeek-R1 has 256 experts, but router picks top 9 (including 1 shared)
- Result: lower per-device memory, less compute, faster sparse inference

Static sharding is not enough

- Traditional serving picks TP / PP / hybrid once at model load time
- Problem: workload changes
	- long prompts need more memory headroom
	- short latency-sensitive prompts hate PP bubbles
- Modern engines route each request to a DIFFERENT pre-sharded worker pool

Core idea

- Prelaunch multiple model instances, each with a different sharding strategy
- Router chooses the best pool per request based on:
	- sequence length
	- GPU memory utilization
	- concurrency / load
	- latency / throughput SLO
- Do NOT reshard live on the fly
	- destroys cache locality
	- stresses memory + network too much

Two-pool example across 8 GPUs

	┌──────────────────────────────┬────────────────────────────────────┐
	│ Pool                         │ Best for                           │
	├──────────────────────────────┼────────────────────────────────────┤
	│ TP-only                      │ short, latency-sensitive requests  │
	│                              │ avoids pipeline bubbles            │
	├──────────────────────────────┼────────────────────────────────────┤
	│ TP + PP hybrid               │ long prompts / high memory         │
	│                              │ pressure / OOM avoidance           │
	└──────────────────────────────┴────────────────────────────────────┘

DeepSeek-R1 style example on 8× B200

- 8 GPUs × 180 GB = 1440 GB total HBM
- Normal case:
	- serve with TP=4 across GPUs 0-3
	- keep GPUs 4-7 for another pool / spare capacity
- Extreme long-context case (>1M tokens):
	- create 2 pipeline stages
	- stage 1 = GPUs 0-3
	- stage 2 = GPUs 4-7
	- each stage now has ~720 GB HBM budget
	- avoids OOM on ultralong input
- Many short prompts arrive:
	- route to TP-only pool
	- avoid PP bubbles
	- lowest latency across all GPUs

Visual

	Short prompt flood:
		request → TP-only pool → low latency

	Huge context:
		request → TP+PP pool → more memory headroom

Decision function idea

- Example policy:
	- if seq_len > 4096 OR gpu_mem_util > 0.8 → use `tp_pp_hybrid`
	- elif concurrent_reqs > 4 → use `tensor_parallel`
	- else → use `tensor_parallel`
- Real systems use richer telemetry, but this is the shape of it

What to monitor

- GPU memory utilization
- GPU compute utilization
- NVLink / NVSwitch traffic
- queue depth / concurrent requests
- TTFT / TPOT behavior

Adaptive tuning logic

- If long PP bubbles leave GPUs idle and memory headroom exists
	- collapse to fewer pipeline stages
- If stages are memory-bound or compute-bound
	- expand to more PP stages or raise TP degree
- Goal: keep GPUs busy while staying inside memory + latency constraints

Why this beats one-size-fits-all

- Long requests get memory-safe routing
- Short requests avoid unnecessary PP overhead
- Cluster stays closer to full utilization under mixed traffic
- Static configs cannot do this well across changing workloads

Mnemonic: don't reshuffle a huge model live; prebuild multiple sharded pools, then send long memory-hungry requests to TP+PP and short latency-sensitive ones to TP-only.
