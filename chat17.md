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

RK: It looks like prefill is compute bound but optimized for latency, while decode is memory bound but optimized for throughput. Disaggregation lets each phase pick the hardware and scheduling that fits its needs best.

Prefill scheduling trade-offs and decode-worker design — summary

Using more GPUs per prefill request

- Data parallelism:
	- full model replicated on each GPU
	- batch split across GPUs
	- outputs aggregated at the end
- Benefit:
	- more aggregate compute and memory bandwidth for that batch
- Cost:
	- fewer independent requests can run at once
	- too many GPUs per request hurts cluster-wide concurrency

- Pipeline parallelism:
	- model layers split across GPUs by stage
	- microbatches flow through like an assembly line
- Benefit:
	- better per-batch throughput when balanced well
- Cost:
	- inter-GPU communication overhead
	- pipeline bubbles if stage split or microbatch size is poor

Visual

	Data parallel:
	same model on many GPUs
	batch split across them

	Pipeline parallel:
	GPU0 stage A -> GPU1 stage B -> GPU2 stage C
	microbatches flow through the stages

- Core trade-off:
	- more GPUs per request = more throughput for that request
	- but fewer total simultaneous requests in a fixed cluster

Latency-aware scheduling and batching

- Scheduler should choose the largest batch/parallelism plan that still satisfies TTFT SLOs
- At low load:
	- often batch size 1
	- run immediately
- At higher load:
	- allow small batching window, e.g. 2-10 ms
	- collect a microbatch only when the delay is acceptable

Adaptive policy

	low load    -> run now
	medium load -> tiny batching window
	high load   -> fuller batches, more aggressive packing

- Many systems intentionally bias prefill toward latency, not perfect utilization
- Reason: fast first token improves UX more than squeezing every last GPU cycle
- Common practice:
	- overprovision prefill capacity
	- absorb prompt bursts without TTFT spikes

Autoscaling and priority classes

- Orchestrators can scale each tier independently
- Example:
	- prefill utilization high, decode low -> add prefill workers
- Useful signals:
	- prefill queue length
	- per-tier utilization
	- TTFT SLO miss rate
- Priority / fast-lane scheduling also helps:
	- short prompts on low-batching fast lane
	- long prompts on throughput-optimized lane

Decode worker design

- Decode workers are generation servers focused on low TPOT and many concurrent sequences
- They receive ready KV cache and continue autoregressive generation efficiently
- In decode-centric designs, request often lands on decode first
	- decode decides local prefill vs remote prefill
	- if remote, request is pushed to prefill queue
	- prefill worker may reuse prefix cache, compute missing prefix, and write KV back
	- decode worker resumes generation

Decode flow

	request arrives at decode worker
					↓
	 local prefill or remote prefill?
					↓
		if remote: enqueue prefill task
					↓
		prefill reads prefix cache + computes missing KV
					↓
		KV written back to decode worker
					↓
		decode pipeline starts token generation

- Decode-worker priorities:
	- handle many active sequences at once
	- keep KV memory footprint under control
	- reduce TPOT with continuous batching and strong memory management

KV transfer into decode

- Decode scheduler allocates destination KV blocks first
- Remote prefill request carries identifiers for those blocks
- Prefill worker uses NIXL to do direct GPU memory reads/writes over the chosen transport
- Transfer should be nonblocking so compute and data movement overlap
- Best practice:
	- preregister peer memory
	- use large pinned windows
	- reduce reregistration churn

Profiling check

- Verify with Nsight Systems that:
	- KV transfer is zero-copy
	- RDMA overlaps compute
	- decode is not stalling on transfers

Mnemonic: prefill scheduling chooses between immediacy and packing, while decode lives or dies by overlapped KV movement and low TPOT.

Continuous batching, sequence grouping, and decode launch — summary

Nsight Systems profiling for end-to-end disaggregated runs

- Use `nsys profile` with traces for CUDA, OSRT, NVTX, UCX, GDS
- Add NIC + IB switch metrics for full network telemetry
- Result: correlated CUDA kernels + UCX activity + storage + network behavior
- Helpful flags:
  - `--trace=cuda-hw,osrt,nvtx,ucx,gds`
  - `--gpu-metrics-device=all`
  - `--nic-metrics=true`
  - `--ib-switch-metrics-device=<GUIDs>`
  - `--storage-metrics --gds-metrics=driver`

Layout transform reminder

- If prefill and decode use different TP layouts
- Insert a layout-transform kernel on the receiver after NIXL read
- Realigns KV blocks to the layout decode expects before use

Continuous batching (iteration-level batching)

- Decode = many small per-token vector-matrix ops → low arithmetic intensity
- Solution: batch the next-token step across many active sequences
- 32 active requests → one forward pass produces 32 next tokens

Visual

  Without batching:
  seq1 step → seq2 step → seq3 step → ... (GPU underused)

  With continuous batching:
  every iteration:
    gather all sequences ready for next token
    run one big matmul
    emit one token per sequence

- Batch size fluctuates each step:
  - finished sequences leave immediately
  - newly-ready sequences (just out of prefill) join next iteration
  - long prompts not yet ready are skipped until ready
- Always tries to maximize batch up to a limit
- High load → big batch, high throughput
- Low load → small batch, low latency (no waiting)

Why decode pools love this

- Disaggregated decode GPUs do only decode
- Never interrupted by a giant bespoke prefill
- Smoother throughput, more predictable TPOT

vLLM knobs

- `--max-num-seqs`: max concurrent decode slots per iteration
- `--max-num-batched-tokens`: total tokens per iteration cap
- `--max-seq-len-to-capture`: CUDA Graph capture coverage; longer seqs fall back to eager
- Set explicitly for predictable memory behavior

Other engines

- DeepSpeed and TensorRT-LLM also implement continuous / in-flight batching with paged KV
- Common idea: scheduler groups decode tasks across streams + paged cache keeps memory tight

Variable-length sequence grouping

- Mixing short and long prompts in one batch causes padding waste
- Padding can be up to ~50% of tokens in some workloads
- Padded "no-op" tokens still cost GPU cycles + memory + network

Fix: bucket by length

  ┌────────────────┬───────────────────────────┐
  │ Bucket         │ Range                     │
  ├────────────────┼───────────────────────────┤
  │ short          │ 0-512 tokens              │
  ├────────────────┼───────────────────────────┤
  │ medium         │ 513-1024 tokens           │
  ├────────────────┼───────────────────────────┤
  │ long           │ 1025+ tokens              │
  └────────────────┴───────────────────────────┘

- Each batch contains similar-length sequences → minimal padding

vLLM SequenceGroup pool

- Rotating pool of SequenceGroups (one per prompt)
- Each iteration advances each group by a fixed token budget
- Finished groups leave; new groups join
- Keeps pipeline full without static padding buckets

Decode-launch optimizations

- Per-token kernel launches add overhead → "bubbles" between decode iterations
- NVIDIA features that reduce this:
  - Programmatic Dependent Launch (PDL): overlap dependent kernels at end of a decode step
  - Device-initiated CUDA Graph Launch: launch graphs from the GPU itself
- Usage notes:
  - instantiate with `cudaGraphInstantiateFlagDeviceLaunch`
  - keep nodes on a single device
  - usually exposed through the framework, not hand-coded
- Effect: trims launch bubbles between decode iterations → lower TPOT

Mnemonic: continuous batching = fluid roster of sequences per iteration; bucket by length to kill padding; PDL + device graphs kill launch bubbles.

KV cache memory management for decode

Combined stack recap

- Length-bucketing + continuous batching + disaggregation + PDL + device-launched CUDA Graphs
- Used by vLLM, SGLang, NVIDIA Dynamo
- Goal: high throughput AND low latency, even with wildly varying prompt lengths

vLLM tuning notes

- `--max-seq-len-to-capture` (default 8192): max seq length covered by CUDA Graphs
- vLLM may pad to nearest captured size → align other knobs to avoid waste
- Tune together:
  - `--max-num-seqs`
  - `--max-num-batched-tokens`
  - `--max-seq-len-to-capture`
- CUDA Graphs ≠ batching control; batching is run by the SequenceGroup pool

Why KV cache is the decode bottleneck

- Decode attends to ALL prior tokens (prompt + already-generated)
- Each sequence stores K and V tensors for every layer × every past token
- KV memory grows linearly with token count
- Decode workers can hit GPU memory limit just from KV, before model weights matter

KV size formula

- bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
- The 2× = keys AND values per layer per token
- Use the model's actual values; FP8/FP4 change bytes_per_element

Worked example: Llama-class 13B, 40 layers, head_dim=128, 4096 tokens

  ┌──────────┬──────────┬──────────────┬──────────────┐
  │ Attn     │ n_kv_hd  │ FP16 KV      │ FP8 KV       │
  ├──────────┼──────────┼──────────────┼──────────────┤
  │ MHA      │ 40       │ ~3.36 GB     │ ~1.68 GB     │
  ├──────────┼──────────┼──────────────┼──────────────┤
  │ GQA (8)  │ 8        │ ~0.671 GB    │ ~0.336 GB    │
  ├──────────┼──────────┼──────────────┼──────────────┤
  │ MQA      │ 1        │ ~0.084 GB    │ —            │
  └──────────┴──────────┴──────────────┴──────────────┘

- MHA = full keys/values per head
- GQA = query heads share KV across groups
- MQA = all heads share one KV

Strategies decode workers use to manage KV

- Paged GPU memory allocator
  - vLLM PagedAttention: KV cache split into fixed-size pages
  - Inactive pages can swap to CPU
  - Also in SGLang, TensorRT-LLM, Dynamo
  - LMCache-style projects layer DRAM + NVMe tiers, schedule recompute vs I/O

- High-memory GPUs + custom allocators
  - Blackwell B200 = 180 GB HBM
  - Blackwell B300 = 288 GB HBM
  - Fixed-size pages reduce fragmentation, enable prefix reuse, scale to many seqs

- KV offloading / paging out
  - Push older KV blocks to CPU RAM or NVMe
  - Bring back on demand → small latency penalty
  - Prefetch / overlap to hide it

- Context limits + compression
  - Hard cap on output length = bounded KV growth
  - Lower-precision KV (FP16 → FP8 → INT8) shrinks footprint
  - Architectural: MQA, GQA, DeepSeek MLA all reduce KV size

Memory hierarchy advantage of disaggregation

- Decode GPU memory = model weights + KV cache only
- No transient prefill memory spikes stealing room
- Example budget on a 180 GB GPU:
  - weights: 70 GB
  - leftover for KV: ~122 GB
  - → bounds (concurrent_seqs × tokens_per_seq) on that GPU
- Disaggregation lets you pick decode HW for memory capacity + bandwidth

Visual

  ┌─────────────────────────────────────────────┐
  │ Decode GPU HBM                              │
  ├─────────────────────────────────────────────┤
  │ [ model weights ] [ KV cache pages ......] │
  │                       ↑ paged, evictable    │
  │                       ↓ overflow → CPU/NVMe │
  └─────────────────────────────────────────────┘

When to offload prefill

- Sending a request to remote prefill pool isn't free
  - queueing delay
  - network transfer (KV back to decode)
- Only offload when it actually helps latency / utilization
- Decision = a routing policy

Disaggregated routing strategies

  ┌──────────────────┬──────────────────────────────────────┐
  │ Strategy         │ How it picks a worker                │
  ├──────────────────┼──────────────────────────────────────┤
  │ Round robin      │ next node in rotation                │
  ├──────────────────┼──────────────────────────────────────┤
  │ Least requests   │ worker with fewest active requests   │
  ├──────────────────┼──────────────────────────────────────┤
  │ Prefix aware     │ uses request's prefix to pick worker │
  ├──────────────────┼──────────────────────────────────────┤
  │ KV aware         │ worker whose KV cache best matches   │
  └──────────────────┴──────────────────────────────────────┘

- Prefix/KV-aware = exploit cache locality → fewer recomputes, faster TTFT

Mnemonic: KV cache is the silent memory hog of decode — page it, compress it, share it via prefix routing, or it eats your HBM before throughput ever does.

Routing factors: when to offload prefill

Core question

- Decode worker gets a request → run prefill locally OR offload to prefill pool?
- Offload isn't free (queue + KV transfer back)
- Goal: maximize cache hits, balance load, hit TTFT SLO

Routing factor table

  ┌──────────────────┬───────────────────────────────────────────────┐
  │ Factor           │ Decision effect                               │
  ├──────────────────┼───────────────────────────────────────────────┤
  │ prompt length    │ long → offload; short → local                 │
  ├──────────────────┼───────────────────────────────────────────────┤
  │ prefix cache hit │ big hit → local (effectively short)           │
  │                  │ no hit  → offload                             │
  ├──────────────────┼───────────────────────────────────────────────┤
  │ prefill queue    │ long  → local (pool saturated)                │
  │                  │ light → offload OK                            │
  ├──────────────────┼───────────────────────────────────────────────┤
  │ decode load      │ busy  → offload to free decode GPU            │
  │                  │ idle  → can do prefill locally                │
  ├──────────────────┼───────────────────────────────────────────────┤
  │ SLO urgency      │ high prio → offload for fast TTFT             │
  │                  │ low prio  → local or delay                    │
  └──────────────────┴───────────────────────────────────────────────┘

Effective prompt length idea

- effective_len = prompt_length − prefix_cached_length
- Big prefix hit → smaller effective_len → cheap → keep local
- Also: transferring cached KV out and back would be pointless
- Full cache hit → no prefill at all, decode runs immediately

Prefill queue as load-shedding signal

- Long queue = "pool can't keep up"
- System gracefully falls back to local prefill
- Queue drains → offloading resumes
- Self-balancing equilibrium between decode and prefill tiers

KV-cache-aware router (vLLM)

- Workers emit KV cache events
- Router tracks which worker holds which prefix
- New request → routed to worker with best cache match
- Effect: fewer recomputes, less cross-worker KV traffic

QoS / priority class hooks

- High-priority request → bypass checks, offload immediately
- Low-priority → local or delayed
- Reserve decode workers for urgent decode-side tasks

Tuning thresholds empirically

- E.g. <50 tokens → faster local on a B200
- New GPUs (more FLOPS + BW) → break-even prompt length goes UP
- Re-tune `PREFILL_LENGTH_THRESHOLD` etc. on hardware upgrades

Pseudo-policy (decode worker side)

- inputs: prompt_length, prefix_cached_length, prefill_queue_size, decode_active_reqs, ttft_slo_ms
- compute: eff_len = max(0, prompt_length − prefix_cached_length)
- tunables (B200/B300 example):
  - PREFILL_LENGTH_THRESHOLD = 256
  - PREFILL_QUEUE_MAX        = 10
  - DECODE_LOAD_THRESHOLD    = 8
- decision rules:
  - long_prefill   = eff_len ≥ PREFILL_LENGTH_THRESHOLD
  - pool_available = prefill_queue_size < PREFILL_QUEUE_MAX
  - if long_prefill AND pool_available → OFFLOAD
  - elif decode_active_reqs ≥ DECODE_LOAD_THRESHOLD AND eff_len ≥ 64 → OFFLOAD
  - else → LOCAL

Decision flow

  request arrives
        │
        ▼
  effective_len = prompt_len − prefix_cached_len
        │
        ▼
  ┌──────────────────────────────────┐
  │ long prefill AND pool free?      │── yes ──► OFFLOAD
  └──────────────────────────────────┘
        │ no
        ▼
  ┌──────────────────────────────────┐
  │ decode busy AND prompt moderate? │── yes ──► OFFLOAD
  └──────────────────────────────────┘
        │ no
        ▼
       LOCAL prefill (better cache locality, lower overhead)

Worker behavior after decision

- Offload = push prompt to global prefill queue, do other work, await KV return
- Local  = decode worker runs prefill itself, then enters decode loop
- If prefill pool starts lagging → new requests naturally fall back to local

Why this conditional strategy wins

- Short / cache-hitting prompts: fast TTFT locally, no overhead
- Long / cold prompts: parallelism via remote prefill
- Prefill pool used only when it pays off
- Optimizes goodput (useful throughput under SLO), not just raw throughput
- Used in DistServe, vLLM, NVIDIA Dynamo

Mnemonic: offload only when the prompt is long, the cache is cold, and the prefill pool has room — otherwise stay local and ride the cache.

Routing config, capacity-aware planning, latency-aware routing

Production routing = config-driven, not hard-coded

- Tools like NVIDIA Dynamo read JSON/YAML at startup (or dynamically)
- Operators tune behavior without code changes

Example Dynamo Planner config (annotated)

  ┌─────────────────────────────────────────────────────────┐
  │ split_policy                                            │
  │   prompt_length_threshold: 256                          │
  │   prefix_cache_weight:  10.0   ← cache hit dominates    │
  │   queue_length_weight:   1.5                            │
  │   decode_load_weight:    0.5                            │
  │   enable_hotspot_prevention: true                       │
  ├─────────────────────────────────────────────────────────┤
  │ cache                                                   │
  │   reuse_prefix: true                                    │
  │   min_cache_hit_ratio: 0.75                             │
  ├─────────────────────────────────────────────────────────┤
  │ autoscale.prefill                                       │
  │   replicas: 4 .. 12                                     │
  │   scale_up:   queue ≥ 8  OR gpu_util ≥ 80%              │
  │   scale_down: queue ≤ 2  AND gpu_util ≤ 40%             │
  ├─────────────────────────────────────────────────────────┤
  │ autoscale.decode                                        │
  │   replicas: 8 .. 24                                     │
  │   scale_up:   queue ≥ 16 OR kv_cache_usage ≥ 75%        │
  │   scale_down: queue ≤ 4  AND kv_cache_usage ≤ 30%       │
  ├─────────────────────────────────────────────────────────┤
  │ qos                                                     │
  │   enable_early_rejection: true                          │
  │   low_priority_threshold_ms: 500                        │
  │   reject_on_slo_violation: true                         │
  └─────────────────────────────────────────────────────────┘

Capacity-aware routing (Dynamo GPU Planner)

- Continuously watches: TTFT, TPOT, KV transfer cost, GPU util
- Can do three things:
  1. modify routing (offload more / less)
  2. rescale a tier (add or remove prefill/decode replicas)
  3. switch to monolithic (no disaggregation) for a request
- Reacts to workload shape:

  workload type        → planner action
  ───────────────────────────────────────────────
  big summarization     → shift GPUs toward prefill
  reasoning / long gen  → shift GPUs toward decode
  short, balanced       → stay monolithic, same GPU

- Goal: keep both TTFT and TPOT within SLO during traffic surges

Latency-aware routing

- Beyond simple thresholds: pick worker with lowest latency cost
- Each candidate worker gets a real-time score
- Lowest score = freest / best fit → request goes there

Simple latency cost example

- latency_cost = 0.7 × occupancy_percent + 0.3 × active_req_count
- occupancy_percent = how busy the GPU is
- active_req_count  = in-flight request count
- weights tuned empirically:
  - if KV memory pressure causes most slowdowns → bump KV-usage weight
  - if queue length dominates                  → bump queue weight

Visual: per-request worker selection

  request
     │
     ▼
  for each worker:
     score = w1·occupancy + w2·active_reqs + w3·kv_use + ...
     │
     ▼
  pick worker with min(score)

More advanced cost factors (foreshadowed)

- KV cache match (prefix locality)
- KV cache pressure (HBM headroom)
- network distance / NIC saturation
- TTFT-so-far for queued requests
- worker-specific HW (B200 vs B300)

Mnemonic: simple thresholds offload, configs declare policy, planners reshape capacity, and latency-cost routers pick the freshest worker per request — four layers of "where should this token go?".
