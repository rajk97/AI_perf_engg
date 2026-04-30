Scaling disaggregated prefill and decode for inference

4/29/26

Chapter overview

- Core idea: split inference into prefill and decode, then scale them separately
- Why: prefill and decode want different hardware, scheduling, and latency goals
- Goal: reduce interference, improve TTFT and TPOT, and scale to huge deployments

Prefill vs decode

	┌──────────────┬───────────────────────────────────────┬──────────────────────────┐
	│ Phase        │ What it does                          │ Bottleneck               │
	├──────────────┼───────────────────────────────────────┼──────────────────────────┤
	│ Prefill      │ Process full input prompt, build KV   │ Compute / FLOPS          │
	│              │ cache over many tokens in parallel    │                          │
	├──────────────┼───────────────────────────────────────┼──────────────────────────┤
	│ Decode       │ Generate output token by token using  │ Memory I/O / bandwidth   │
	│              │ KV cache                              │                          │
	└──────────────┴───────────────────────────────────────┴──────────────────────────┘

- Prefill = parallel, throughput-heavy, matrix-multiply dominated
- Decode = sequential, latency-sensitive, KV-read/write dominated

Why monolithic serving struggles

	One GPU
		↓
	prefill + decode share device
		↓
	batching helps throughput
		↓
	but long prefills block short requests
		↓
	TTFT gets worse
		↓
	decode backlog hurts TPOT too

- One long prompt can monopolize the GPU during prefill
- After that, decode underuses compute because token generation is small and sequential
- A single colocated engine cannot optimize TTFT and TPOT at the same time very well

Disaggregation

- Send prefill and decode to different worker pools
- Let each pool specialize for its own phase
- Prefill and decode do not need the same hardware or same scheduling policy

	request
		↓
	prefill worker pool
		↓
	KV cache ready
		↓
	decode worker pool
		↓
	streamed tokens

- This idea was popularized by DistServe
- Reported upside: much higher request throughput under strict TTFT/TPOT SLOs
- At industry scale, disaggregated prefill/decode is now standard practice

Why prefill-decode disaggregation matters

- Modern interactive services care about both:
	- TTFT = time to first token
	- TPOT = time per output token
- Without separation, long prefills create interference for everyone else
- This causes head-of-line blocking:
	- long prompt at front of queue
	- short prompt behind it waits anyway
	- tail latency explodes

Visual

	FIFO colocated system:
	[long prefill........][decode][short prefill][decode]
								 ↑
					everyone behind waits

	Disaggregated system:
	prefill pool: [long prefill........]   [short prefill]
	decode pool : [decode][decode][decode][decode]

- Flexible routing can even bypass dedicated prefill workers for light prompts when useful
- Short requests no longer need to suffer behind heavy prompts

Advantages of disaggregation

1. Reduced interference
	 - prefill work no longer blocks decode work on the same device
	 - decode-heavy traffic no longer delays new prompt processing
	 - latency becomes more predictable

2. Phase-specific optimization
	 - prefill pool can optimize for compute throughput
	 - decode pool can optimize for memory bandwidth and low per-token latency

Scale takeaway

- At ultrascale, splitting phases lets operators optimize each bottleneck independently
- This improves utilization, latency, QoS, and cost efficiency across very large clusters

Mnemonic: prefill wants FLOPS, decode wants bandwidth, so split them before they fight.

Goodput, phase-specific tuning, and cluster pools — summary

Goodput under latency SLOs

- Important rule: throughput only counts if both latency SLOs are met
- Example SLOs:
	- P90 TTFT <= 0.4 s
	- P90 TPOT <= 0.04 s
- Colocated system example:
	- ~3 RPS within TTFT bound
	- ~1.6 RPS within TPOT bound
	- so real goodput = 1.6 RPS because both bounds must hold

2P1D example

- Configuration:
	- 2 prefill GPUs
	- 1 decode GPU
- Measured capacity:
	- each prefill side together: 5.6 x 2 = 11.2 RPS total prefill capacity
	- decode side: 10 RPS total decode capacity
- End-to-end system goodput is limited by the slower side:

	total goodput = min(prefill capacity, decode capacity)
								= min(11.2, 10)
								= 10 RPS total

	per-GPU goodput = 10 RPS / 3 GPUs = 3.3 RPS per GPU

- Main lesson:
	- decode-side improvement helps TPOT
	- prefill isolation helps TTFT
	- both SLOs must pass for requests to count as useful throughput

- Disaggregation also tightens tail latency because phases stop interfering with each other
- Cost question still matters:
	- more GPUs can improve goodput and latency
	- but you must decide if the cost/perf trade-off is worth it for your workload

Phase-specific optimizations

- Prefill tuning:
	- compute-bound
	- prefers more tensor parallelism when needed to drive FLOPS
	- benefits strongly from low precision like FP8 / FP4
- Decode tuning:
	- memory-bandwidth-bound
	- often prefers little or no tensor parallelism because sync overhead hurts
	- benefits from fused kernels and high-memory-throughput GPUs

Visual

	Prefill:
	more compute, more parallel math, lower precision helps

	Decode:
	more KV traffic, less cross-GPU sync, bandwidth matters most

- Monolithic serving forces one GPU choice and one parallelism strategy for both phases
- Disaggregation lets each phase choose what fits best

Heterogeneous clusters

- You can assign different GPU types to different roles
- Example:
	- compute-optimized GPUs for prefill
	- memory-optimized GPUs for decode
- This can improve throughput per dollar versus using one GPU type everywhere
- Newest GPUs often work well for both, but heterogeneity remains a useful cost lever

Profiling reminder

- Use Nsight Systems and related tools to inspect:
	- prefill kernels
	- decode kernels
	- RDMA / KV transfers
	- overlap between communication and compute

Disaggregated cluster pools

- Basic setup:
	- one pool of prefill workers
	- one pool of decode workers
	- a scheduler/router coordinates KV handoff between them
- Practical design keeps them in the same data center to preserve latency SLOs

Visual

	client request
			 ↓
	 decode worker ingress
			 ↓
	decide: local prefill or offload?
			 ↓
	prefill worker computes KV
			 ↓
	KV transferred over fast interconnect
			 ↓
	decode worker generates tokens

Interconnect / transfer path

- KV handoff should use fast GPU-to-GPU networking:
	- NVLink / NVSwitch
	- InfiniBand
	- GPUDirect RDMA
	- UCX / NIXL-style transport
- Goal: move KV without host copies and with minimal latency

Decode-centric design

- Common architecture: request enters on decode worker first
- Why:
	- prefill workers are already compute-heavy
	- decode nodes can handle client I/O, session state, routing, and policy logic
	- simplifies ingress, autoscaling, and control flow
- Alternative: dedicated API router in front
	- works too
	- but adds another moving part and more coordination overhead

Role example

- Prefill workers:
	- fewer, heavier compute-oriented GPUs
- Decode workers:
	- more memory-oriented GPUs with larger decode pool
- Specialized accelerators can also appear:
	- e.g. context-processing chips aimed specifically at prefill

Mnemonic: goodput = throughput that satisfies both TTFT and TPOT, and disaggregation lets each phase use the hardware it actually wants.

KV handoff, elastic scaling, and prefill-worker design — summary

Heterogeneous worker example

- Prefill workers can use compute-strong GPUs
- Decode workers can use high-HBM GPUs
- Example pattern:
	- B200-like GPUs for compute-heavy prefills
	- B300-like GPUs for memory-heavy decodes
- Goal: match GPU strengths to phase bottlenecks while lowering cost

NIXL / GPUDirect RDMA handoff

- KV cache should move directly GPU-to-GPU with minimal control overhead
- NIXL abstracts the transport layer for:
	- NVLink
	- RDMA / UCX fabrics
	- GPUDirect Storage tiers
- Decode workers register regions of GPU memory that remote prefill workers can write into directly

Fast path

	decode worker reserves KV buffer
				 ↓
	buffer registered for RDMA
				 ↓
	small descriptor / ID shared via control plane
				 ↓
	prefill worker computes KV
				 ↓
	prefill worker writes KV directly into decode GPU memory
				 ↓
	decode starts token generation

- Key idea: send small buffer IDs/descriptors, not bulky full memory metadata every time
- Discovery/control plane tools such as etcd can manage worker discovery, leases, and handle exchange
- First-contact setup is heavier; steady-state control messages stay lightweight

Layout mismatch note

- If prefill and decode use different tensor-parallel layouts, decode must do a KV layout transform after receiving the transfer
- This is cheaper than recomputing prefill and usually small relative to network transfer time

Elastic scaling

- Prefill and decode scale independently
- If long prompts dominate, add prefill workers
- If long outputs dominate, add decode workers
- Because phases are separated, scaling one role does not directly disrupt the other

Dynamic cluster behavior

- Some runtimes can add/remove workers without stopping the cluster
- New workers register and begin pulling tasks
- If prefill workers disappear, decode workers can temporarily do more local prefill to absorb the gap
- This flexibility matters at ultrascale where load changes constantly

Prefill worker design

- Prefill workers are prompt servers optimized for heavy forward-pass compute
- They should use GPUs with high FLOPS and strong large-GEMM performance
- They commonly use:
	- tensor parallelism
	- pipeline parallelism
	- and, when needed, data / expert / context parallelism

Memory behavior

- Prefill workers must hold:
	- model weights
	- working activations for forward pass
	- temporary KV cache for the prompt
- Important detail:
	- KV cache is produced, transferred out, then usually does not stay long on the prefill node
- Many systems preallocate a large memory region for prefill workspace to reduce fragmentation and allocation overhead

Latency vs throughput trade-off on prefill

Latency-first mode

- Start prompts immediately with little or no batching
- Best for minimizing TTFT
- Cost:
	- lower GPU utilization
	- fewer concurrent requests served per cluster
- Useful when strict latency SLOs matter more than utilization

Throughput-first mode

- Batch prompts together before launching prefill
- Best for maximizing total RPS and keeping GPUs full
- Cost:
	- batching delay before execution starts
	- larger batches increase wait time for each request

Visual

	Latency-first:
	prompt arrives → run now

	Throughput-first:
	prompt arrives → wait for batch → run full GPU

- Large-batch prefill raises arithmetic intensity and overall utilization
- But it can worsen TTFT for individual prompts
- Extreme throughput setups may even assign multiple GPUs per request

Rule of thumb

- Use latency-first when TTFT SLOs are strict
- Use throughput-first when total served RPS matters more than individual prompt start time

Mnemonic: prefill workers are prompt factories, so feed them FLOPS, ship KV fast, and choose between instant starts or fuller batches.
