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
