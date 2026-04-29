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
