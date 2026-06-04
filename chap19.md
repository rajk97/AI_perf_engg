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

Dynamic precision changes

Why do this at runtime

- Blackwell-class GPUs support FP8 / FP4 Tensor Core math
- Lower precision gives:
	- more throughput
	- less memory use
	- usually small quality loss if chosen carefully
- Goal: run at the LOWEST precision that still preserves output quality

What triggers precision changes

- Model confidence
	- sharp / peaky next-token distribution → low precision is usually safe
	- flat / uncertain distribution → stay in higher precision
- Memory pressure
	- KV cache near full → compress activations / KV more aggressively

Confidence-based switching

- Measure output confidence from the token distribution
- Book examples:
	- Shannon entropy of softmax
	- max softmax probability
	- top1-top2 logit margin
- Low entropy / high margin = confident
- High entropy / low margin = uncertain

Practical rule

- deterministic continuation (quotes, lists, boilerplate)
	→ FP4 / FP8 often safe
- ambiguous reasoning branch
	→ FP8 / BF16 / FP16 safer

Visual

	confidence high   → drop precision → faster
	confidence low    → raise precision → safer

Memory-pressure switching

- If KV cache reaches ~90% memory usage
	- quantize new KV entries INT8 → INT4
	- or retroactively compress older entries
- Cuts KV memory by ~50% for those entries
- Risk: quantization error can accumulate across many decode steps
- So quality should be rechecked periodically

Precision tradeoff table

	┌──────────────────────────────┬──────────────┬──────────────┬─────────────────────────┐
	│ Mode                         │ Memory       │ Throughput   │ Quality impact          │
	├──────────────────────────────┼──────────────┼──────────────┼─────────────────────────┤
	│ FP16 baseline                │ 1.0×         │ 1.0×         │ none                    │
	│ FP16 weights + FP8 acts      │ ~0.5×        │ ~1.5×        │ negligible (<0.1%)      │
	│ INT4 weights + FP8 acts      │ ~0.25×       │ ~1.8×        │ ~0.5% drop              │
	│ INT4 weights + FP4 acts      │ ~0.2×        │ ~3.5×        │ ~1% drop, tune carefully│
	└──────────────────────────────┴──────────────┴──────────────┴─────────────────────────┘

- FP8 often gives near-free savings
- FP4 gives bigger gains but needs careful per-layer validation

Recommended software stack

- PyTorch AMP (`torch.autocast`) manages BF16 / FP16 only
- AMP does NOT manage FP8 / FP4 automatically
- For FP8/FP4 on NVIDIA GPUs, use Transformer Engine (TE)
- Blackwell guidance:
	- latency-critical decode via AMP → prefer BF16 over FP16
	- FP8 path → use Transformer Engine MXFP8
	- FP4 path → use NVFP4 selectively, especially KV / light layers

Layer-wise precision control

- Not every layer needs same precision
- Example strategy:
	- early / sensitive layers → FP8 or BF16/FP16
	- lighter / tolerant layers → FP4
- Runtime hooks can raise or lower layer precision on demand

Token-wise precision control

- Even finer: decide precision per decode step
- Typical loop:
	1. run current token in default precision
	2. compute confidence signal on device
	3. every N steps, reevaluate precision choice
	4. if confidence sustained high → enter FP8
	5. if confidence drops → exit FP8 back to BF16/FP16

Why the example uses hysteresis

- Enter threshold > exit threshold
- Prevents "precision flapping" every step
- EMA smoothing + reevaluation interval reduce host-device sync overhead

Example threshold idea

- enter FP8 when confidence > 6.0
- exit FP8 when confidence < 3.0
- Must be calibrated on a validation set for your prompts + model

What the sample code is really doing

- Uses exactly one precision context per step
- If TE is present and confidence is high → FP8 path
- Otherwise → AMP BF16 / FP16 path
- Confidence metric stays on device, host checks only periodically
- Result: elastic precision without syncing CPU every token

Why this works

- Low-entropy stretches run faster in low precision
- High-entropy stretches recover quality with higher precision
- Best of both: high throughput plus bounded quality loss

Kernel autotuning lead-in

- Precision is only one runtime knob
- Next knob: kernel choice itself
- Attention / MLP speed depends on:
	- tile size
	- block size
	- loop unrolling
	- memory access pattern
- Static heuristics are not enough when batch size / sequence length vary per request
- Runtime kernel autotuning picks or JIT-compiles the best kernel for the current shape

Mnemonic: use FP4/FP8 only when the model is confident or memory is tight, fall back to BF16/FP16 when uncertainty rises, and smooth the switch with hysteresis so precision adapts without flapping.

Kernel autotuning for attention and MLP paths

Why autotuning matters

- Two big compute paths dominate transformers:
	- self-attention
	- feed-forward MLP
- Best kernel depends on runtime shape:
	- sequence length
	- batch size
	- tile size
	- shared-memory pressure
	- occupancy
- One fixed kernel leaves performance on the table

Attention autotuning

- Short sequences:
	- simpler attention kernel can win
	- avoids FlashAttention setup overhead
- Long sequences:
	- tiled / FlashAttention-style kernels win
	- shared-memory reuse avoids repeated HBM fetches

Practical rule of thumb

- if seq_len < ~128-256 → standard kernel may be faster
- else → tiled / FlashAttention kernel usually wins
- Exact breakeven depends on hardware, so benchmark it

Visual

	short L   → simple kernel
	long  L   → tiled / FlashAttention kernel

Tile-size tradeoff example

	┌───────────┬──────────────────┬───────────────┬──────────────────┐
	│ Tile      │ Shared mem       │ Occupancy     │ Throughput       │
	├───────────┼──────────────────┼───────────────┼──────────────────┤
	│ 64×64     │ 48 KB            │ 85%           │ 8.2 GOPS         │
	│ 128×64    │ 64 KB            │ 78%           │ 10.5 GOPS        │
	│ 128×128   │ 96 KB            │ 72%           │ 9.8 GOPS         │
	│ 256×128   │ 128 KB           │ 60%           │ 11.3 GOPS        │
	└───────────┴──────────────────┴───────────────┴──────────────────┘

What this shows

- Larger tiles:
	- more data reuse
	- fewer global-memory loads
	- higher sustained throughput on long sequences
- But also:
	- more registers / shared memory
	- fewer blocks resident per SM
	- lower occupancy
- So best tile is a shape-dependent tradeoff, not "largest wins"

MLP autotuning

- MLP = big GEMMs with nonlinearity in between
- cuBLAS / cuBLASLt / CUTLASS often have multiple algorithms per shape
- Best algorithm can differ for:
	- batch=1
	- batch=16
	- same hidden dim, different shape aspect ratio
- Engines can:
	- benchmark candidate GEMM kernels on first encounter
	- cache the winner for that shape
	- reuse it on subsequent calls

Runtime strategy

- New shape appears:
	1. benchmark a few candidate kernels
	2. pick fastest algorithm
	3. cache by shape
	4. reuse until workload changes

Triton / CUTLASS style tuning

- Triton can JIT multiple variants with different tile shapes
- CUTLASS / cuBLASLt can search algorithm space too
- Runtime measures actual performance on the target GPU
- Empirical tuning beats theoretical guesses under real cache/bank-conflict behavior

Profiling guidance

- Nsight Systems:
	- CUDA timelines
	- memcpy / NVLink activity
	- kernel overlap
- Nsight Compute:
	- memory workload analysis
	- L2 misses
	- cache / bank conflict clues
- Use both when comparing tile variants side by side

Occupancy-based launch tuning

- Larger tile may lower occupancy by consuming too much shared memory
- Smaller tile may improve latency on short sequences by letting more blocks run concurrently
- Some systems query CUDA Occupancy API at runtime to choose:
	- thread block size
	- shared-memory usage
	- launch parameters

Occupancy intuition

	long sequences:
		bigger tiles → better reuse → fewer HBM loads → faster

	short sequences:
		smaller tiles → more resident blocks / SM → lower latency

Autotuner's real job

- Balance:
	- throughput
	- occupancy
	- shared-memory footprint
	- launch overhead
	- actual observed latency
- Inference frameworks usually automate this internally

Mnemonic: short sequences want small simple kernels, long sequences want large tiled kernels, and the autotuner's job is to trade shared-memory reuse against occupancy until it finds the fastest shape-specific path.

6/4/26:

Autotuning workflow and occupancy-aware shared memory

Six-step autotuning loop

	1. measure workload
		 - inspect batch size, sequence length, and shape
	2. select candidates
		 - attention kernel choices, GEMM variants, launch configs
	3. estimate / benchmark
		 - quick trial runs on sample inputs
	4. choose best
		 - lowest latency or enough throughput
	5. cache result
		 - store by shape / workload signature
	6. execute
		 - run the layer with the chosen kernel

- This is like a database query optimizer picking a query plan
- Over time the engine builds a library of "best kernels for this shape"

How to keep autotuning overhead low

- Tune asynchronously in a separate stream while a default kernel runs
- Or tune during low-traffic periods
- Add warm-up on model load:
	- max sequence length
	- max batch size
	- common production shapes
- Keep monitoring per-layer latency at runtime
- If a layer becomes a bottleneck, revisit kernel selection
- Advanced systems may use multiarmed bandits to keep exploring alternatives

Bottom line

- Static kernels become adaptive kernels
- Engine keeps retuning itself as traffic patterns change

Dynamic shared-memory allocation + occupancy-aware selection

Core idea

- Shared memory per SM is limited
- Registers per SM are limited
- Occupancy = how many blocks/warps can stay active on an SM
- Better performance comes from balancing:
	- data reuse in shared memory
	- occupancy / latency hiding

Tile size tradeoff

- Let T = attention tile width in tokens
- Each block stores Q, K, V tiles in shared memory
- Shared memory per block grows like O(T^2)

Large tile (e.g. T=256)

- Pros:
	- reuse K/V more
	- fewer DRAM loads
	- better for long sequences
- Cons:
	- huge shared-memory footprint
	- fewer blocks fit per SM
	- occupancy can drop hard
	- example: 1 block/SM, ~30% occupancy

Small tile (e.g. T=64)

- Pros:
	- much lower shared-memory use
	- more blocks fit per SM
	- better latency hiding and utilization
- Cons:
	- reload K/V more often
	- more DRAM traffic

Visual

	big T  → more reuse, less DRAM, lower occupancy
	small T → less reuse, more DRAM, higher occupancy

What determines optimal T

- sequence length L
- GPU shared memory per block
- number of SMs
- register pressure / threads per block
- actual DRAM thrash vs occupancy in practice

Runtime selection pattern

- choose T from candidate set like 64 / 128 / 256
- compute `shared_mem_bytes = 3 * T * T * sizeof(float)` for Q/K/V tiles
- launch same kernel binary with different dynamic shared-memory size
- use `extern __shared__` buffer inside the kernel
- same kernel, different launch-time tile configuration

Occupancy feedback loop

- Query CUDA Occupancy API after picking T
- If only 1 block fits per SM and occupancy is poor:
	- reduce T
- If DRAM is thrashing and occupancy is already healthy:
	- increase T
- Can use CUDA Occupancy API or DCGM to drive the loop
- Result: each layer adapts tile size to current L and hardware limits

Why this matters

- Long-sequence attention wants more reuse
- Short-sequence attention wants more active blocks
- Dynamic shared-memory allocation lets one kernel binary serve both efficiently

Mnemonic: autotuning is a measure-pick-cache loop, and occupancy-aware shared memory means choosing the biggest tile that improves reuse without starving the SM of active blocks.
