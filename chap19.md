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

Register pressure, carveout tuning, and speculative KV prefetching

Register pressure matters too

- More registers per thread can speed one thread up
- But SMs have a fixed register file
- If each thread uses too many registers:
	- fewer warps fit on the SM
	- occupancy drops
- Runtime can switch to a lower-register kernel variant
	- more instructions
	- but higher occupancy

How adaptive systems detect the problem

- Nsight Systems / Nsight Compute:
	- achieved occupancy
	- execution efficiency
	- memory stalls
- CUDA Occupancy API:
	- how many blocks/warps fit per SM
- DCGM:
	- real-time SM / GPU utilization at system level

What the controller looks for

- idle warps / low active warps (< 50%)
- memory stall cycles (> 70%)
- example: seq_len 2048 attention only reaches 30% occupancy

Possible fix

- reduce per-block shared memory
- split attention work into more passes
- use fewer registers or smaller blocks
- may raise occupancy enough to improve throughput
- but if kernel is memory-bound, MORE shared memory can still win by reducing stalls

Bottom line

- occupancy is not always "higher is better"
- best point is where reuse, occupancy, and stalls are balanced

Runtime launch table idea

- Keep a small mapping from problem size → launch config
- Example:
	- seq_len 512  → 128 threads/block, 16 KB smem
	- seq_len 4096 → 256 threads/block, 64 KB smem
- Store in JSON / config for easy updates across new models and GPUs

L1 vs shared-memory carveout

- Modern NVIDIA GPUs share one on-chip pool between:
	- L1 cache
	- shared memory
- Runtime can bias the split per kernel using `cudaFuncSetAttribute`
- Example:
	- default maybe 50/50
	- large-tile attention may prefer 75% shared / 25% L1
- Important: carveout is a HINT, not a guarantee
- Must verify effect with profiling

Why this section matters

- Dynamic shared memory + occupancy-aware selection keep SMs busy for each layer shape
- Especially important when some layers or batch sizes would otherwise underutilize the GPU

Speculative KV prefetching for faster TTFT

What problem it solves

- TTFT suffers when decode must wait for the needed KV state to be ready
- If some KV lives in compressed / off-GPU tiers, loading it can delay the first token
- Idea: fetch likely-needed KV BEFORE the decode step asks for it

Core intuition

- "Speculative" here means:
	- predict which KV blocks / token path will be needed next
	- prefetch them early
- If prediction is right, decode finds the KV already warm / loaded
- If prediction is wrong, you wasted some bandwidth, but maybe still improved latency overall

SpeCache-style flow

- KV is compressed and partly off-GPU
- After the first output token is generated:
	- system also computes a speculative next token/path
	- while decode continues, it prefetches reduced-precision KV for that speculative path
- On later steps:
	- actual path + speculative path are advanced in parallel
	- top-k likely KV chunks for the speculative branch are prefetched ahead of time

Visual

	Without prefetch:
		need token t1
			↓
		fetch KV from slower tier
			↓
		decode t1
		→ TTFT / next-step latency includes fetch wait

	With speculative prefetch:
		predict likely next path
			↓
		fetch likely KV EARLY while other compute is happening
			↓
		when decode needs it, KV is already there
		→ wait largely hidden

Simple mental model

	normal:
		ask for book → librarian walks to shelf → returns → you read

	speculative prefetch:
		librarian guesses which book you'll ask for next and puts it on the desk early
		if correct, you read immediately

What is being prefetched exactly

- Not "all possible future KV"
- Usually a reduced-precision subset of the most likely KV chunks / pages
- Often top-k most relevant items for the speculative branch
- This keeps bandwidth under control

Why this can help TTFT

- First-token latency is sensitive to setup delays
- If KV movement/prep overlaps with earlier compute, the critical path shrinks
- Especially useful when KV is tiered across GPU / CPU / storage

Trade-off

- Correct speculation → lower latency
- Wrong speculation → wasted transfer / prefetch work
- So validate access patterns and storage tiers before using it

Mnemonic: speculative KV prefetching means guessing the next KV pages the decoder will need, pulling them in early while other work is happening, and hoping the guess is right so decode doesn't stop to wait on memory.

KV offload and one-layer-ahead prefetching

Why this exists

- Modern KV caches are enormous:
	- many layers
	- huge context windows
	- long reasoning traces
- So systems often tier KV across:
	- GPU HBM
	- CPU memory
	- SSD / NVMe

Naive way (bad for TTFT)

- Need next token
- Synchronously fetch missing KV from CPU / SSD
- Then run decode
- Result: first token and next-step latency stall on data movement

Better way: prefetch KV early

- Start moving needed KV pages toward GPU as soon as possible
- Overlap data movement with prefill / current-layer compute
- By the time decode reaches that layer, KV is already local or in flight

Regular KV prefetching

- Keeps only current layer's KV on GPU
- Offloads other layers' KV to CPU
- During layer i compute:
	- prefetch layer i+1 KV into GPU
	- write layer i-1 KV back to CPU
- Net effect: one-layer-ahead pipeline

Visual

	layer 1 compute      | prefetch layer 2 KV | evict layer 0 KV
	layer 2 compute      | prefetch layer 3 KV | evict layer 1 KV
	layer 3 compute      | prefetch layer 4 KV | evict layer 2 KV

	→ compute, prefetch, and eviction overlap continuously

Why it helps

- GPU rarely waits for KV to arrive
- CPU-resident KV becomes much less painful
- Throughput penalty still exists, but often only ~5-10%
- Remaining cost comes from CPU DRAM bandwidth + PCIe overhead

Hugging Face example

- `generate(cache_implementation="offloaded")`
	- dynamic / variable-length serving
- `generate(cache_implementation="offloaded_static")`
	- static shapes + `torch.compile` + CUDA Graphs
	- highest throughput when shapes are fixed

What OffloadedCache is doing underneath

- Move layer 1 KV to GPU
- While layer 1 computes, async DMA layer 2 KV to GPU
- Before layer 2 starts, its KV is already local
- Repeat for each layer
- This is classic prefetching, not speculative branching yet

TTFT angle

- By end of prefill, ideally the decode-critical caches are:
	- already in GPU memory, or
	- already queued for transfer
- That shrinks the stall between prefill finishing and first token generation starting
- Use NVTX markers to measure first-token decode idle time and catch missed prefetches

How to implement overlap yourself

- Use a dedicated nonblocking prefetch stream
- Use main compute stream for forward/decode
- Launch async copy on prefetch stream
- Record an event when KV copy is ready
- Make compute stream wait ONLY right before the KV is consumed

Visual: streams

	compute_stream : [forward current token] ----wait kv_ready---- [consume KV]
	prefetch_stream:        [cudaMemcpyAsync next KV chunk] --record event--

	→ sync happens just-in-time, not upfront

Important implementation details

- Host buffers must be pinned (`cudaMallocHost` / page-locked)
	- otherwise `cudaMemcpyAsync` may serialize
- If source is another GPU:
	- use `cudaMemcpyPeerAsync`
	- enable peer access
- If using Unified Memory:
	- `cudaMemPrefetchAsync` can stage pages ahead of time
- If pattern repeats:
	- capture the sequence in CUDA Graphs to cut launch overhead

Pipeline mindset

- Treat data movement as part of inference, not as a separate afterthought
- Always try to have the NEXT needed KV / weights in flight while CURRENT compute runs
- Same idea as compute pipelining, but for memory transfers

Where this sits relative to speculative prefetching

- Regular KV prefetching:
	- one-layer-ahead, deterministic
	- "I know layer i+1 will be needed next"
- Speculative KV prefetching:
	- predict which future path/pages will likely be needed
	- useful when branch/path is uncertain or multi-path

Mnemonic: ordinary KV prefetching is a conveyor belt, not a guess — while layer i computes, layer i+1's KV is already moving in and layer i-1's KV is moving out, so decode almost never stops to wait on memory.

Real-time KV cache compression and policy switching

Why compress KV at all

- KV grows linearly with generated/context tokens
- For long chats, documents, and reasoning traces, KV often becomes the biggest memory consumer
- Compression reduces:
	- GPU memory pressure
	- network/offload bandwidth
	- OOM risk

Simplest and most practical method: quantization

- Baseline KV is often FP16 / BF16
- Many systems can compress KV to INT8 or INT4 with little quality loss
- Hugging Face supports this via `cache_implementation="quantized"`
- Cost: a bit of quantize/dequantize compute
- Benefit: large memory savings, usually small quality impact

If quantization and CPU offload are combined

- Host buffers should be pinned (page-locked)
- Otherwise copies may serialize and copy bandwidth drops

Policy switching = change compression strategy as conditions change

Example policy ideas

- Keep the newest 128 tokens in full precision
- Compress older tokens to 4-bit
- Rationale:
	- recent context usually matters most
	- older context can often tolerate more compression due to recency bias

Adaptive policy triggers

- Prompt length gets very large
	- widen or shrink the full-precision window
- GPU memory usage crosses threshold
	- e.g. >80% → compress all KV to 8-bit
	- severe pressure → switch to 4-bit
- Value distribution / variance suggests aggressive quantization is safe

Multi-tier compression policy

	mild pressure    → INT8 KV
	high pressure    → INT4 KV
	hottest recent   → FP16/BF16 residual window

Why dynamic switching is tricky

- During generation, the engine may need to change policy mid-session
- That means the compressed representation must be ready before it is needed
- Practical tools:
	- double buffering
	- background compression threads
	- safe-point switching (end of iteration, not mid-matmul)

One practical design

- Keep FP16 active initially
- Build INT8 copy in background
- If memory threshold is crossed:
	- switch reads to INT8 version
	- free FP16 memory
- Future attention reads dequantize from INT8

Important implementation note

- Switch policies only at safe boundaries, like the end of a decode iteration
- Hide requantization latency on a background stream
- Avoid mid-calculation format swaps

What not to over-focus on

- Lossless methods (entropy coding, ZFP, clustering) exist
- But production KV paths rarely use them because throughput is worse
- In practice, quantization wins on the speed / memory / simplicity tradeoff

Production guidance

- Quantization is the default practical answer
- Expect roughly 2-4× memory reduction depending on bit-width
- Lossless compression remains mostly research/offline today

How good quantization is done

- Better than one global scale:
	- per-head
	- group-wise / per-channel scaling
	- residual window for recent tokens
- Hugging Face QuantizedCache uses per-channel group-wise quantization
- Range is calibrated per attention head
- This is basically magnitude / min-max style quantization

HQQ backend

- Half-Quadratic Quantization (HQQ)
- Calibration-free, on-the-fly quantizer
- Supports 2/3/4/8-bit
- Handles outliers / heavy-tailed error well
- Integrated with Transformers KV cache path
- Has both PyTorch and CUDA implementations

Dynamic bit-width switching in practice

- Transformers does NOT let you mutate an already-created cache object's bit-width in place
- Practical workaround:
	- generate in chunks
	- when memory pressure rises, start next chunk with a new cache config
	- similar to falling back to offloaded cache on OOM

Visual

	start of session:
		recent KV  → FP16/BF16
		old KV     → FP16/BF16

	memory pressure rises:
		recent KV  → FP16/BF16 residual window
		older KV   → INT8

	severe pressure:
		recent KV  → FP16/BF16 residual window
		older KV   → INT4

Why this works

- Most useful context is often recent
- Older tokens can often be stored more cheaply
- Compression policy becomes another runtime control knob, like precision or parallelism

Mnemonic: KV compression is the memory-pressure relief valve — keep the freshest tokens high precision, squeeze the cold history to INT8 or INT4, and switch policies only at safe iteration boundaries.

Chunked cache switching, eviction policies, and RL tuning

Dynamic quantized cache in practice

- Start with modest compression, e.g. INT8 HQQ
- Watch true device memory with `torch.cuda.mem_get_info()`
- If used ratio crosses threshold (example: 90%)
	- switch NEXT chunk to a lower-bit cache, e.g. INT4
- Generate in small chunks so policy can change safely between chunks
- Do not mutate an already-live cache object in place

Why chunking matters

- Transformers' quantized cache API configures the cache at creation time
- So safe policy switching means:
	- finish current chunk
	- create next chunk with new cache config
	- continue generation
- This avoids private/internal cache hacks

Recommended operational safeguards

- Log every policy switch or increment a counter
- Correlate switches with:
	- output anomalies
	- quality regressions
	- latency spikes
- Add hysteresis / cooldown so bit-width does not flap up and down

Policy can also depend on request class

- Premium user → lighter compression / better quality
- Free tier → heavier compression / max throughput
- Request metadata becomes another control signal

Eviction and context management

- Compression is not the only tool
- Another runtime policy is eviction / dropping old context
- Example options:
	- LRU-style discard of very old tokens
	- sliding-window attention
	- compress old context into a summary
- Useful when model has recency bias or explicit sliding-window attention

Visual

	recent tokens     → keep in window / high precision
	old tokens        → quantize more aggressively
	very old tokens   → evict or summarize

Production rule of thumb

- A high-watermark memory trigger (around 80%) is often enough to avoid OOMs
- But always validate 4-bit vs 8-bit on your domain-specific outputs

Reinforcement learning agents for runtime tuning

Why RL enters the picture

- The system now has many runtime knobs:
	- parallelism mode
	- precision mode
	- batch size / wait time
	- cache compression on/off
	- speculative decoding on/off
	- draft-model choice
	- speculative KV prefetch on/off
- Hand-written heuristics can become messy
- RL offers one unified controller for trading off throughput vs latency vs quality

How to frame it

- Environment = live inference system + metrics
- State = current observations
	- GPU utilization
	- memory utilization
	- average latency
	- queue length
	- etc.
- Actions = control knobs the agent can change
- Reward = business objective

Example RL action space

	1. choose parallelism mode (single / TP / PP / hybrid)
	2. choose precision mode (FP8 vs FP8+FP4)
	3. adjust batch size or wait time
	4. enable/disable cache compression
	5. enable/disable speculative decoding
	6. choose smaller draft model
	7. choose larger draft model
	8. enable/disable speculative KV prefetching

Example reward

- `reward = throughput - λ * max(0, latency - SLA)`
- Meaning:
	- maximize throughput
	- punish latency only when it exceeds SLA
- λ controls how painful SLA violations are

How to think about λ

- Bigger λ:
	- safer latency
	- less aggressive throughput chasing
- Smaller λ:
	- more throughput-seeking
	- tolerates some latency overshoot
- Practical calibration:
	- choose λ so a typical latency overshoot costs about as much as the throughput gain you would trade for it

Important training guidance

- Normalize state features
- Example:
	- queue length / max_queue
	- utilization as 0..1
- Helps convergence because agent doesn't need to learn feature scale separately

What the agent can learn over time

- when cache compression is worth it
- when PP beats TP for the current load
- when speculative decode helps enough to justify cost
- when to accept slight latency risk for throughput gain

Practical recommendation

- Start with heuristics as the baseline
- Only layer RL on top after the simple rules are stable and measurable
- RL is an incremental optimizer, not the first thing to build

Mnemonic: switch cache bit-width only between chunks, log and cool down policy flips, use eviction when compression is not enough, and think of RL as the meta-scheduler that learns how to tune all those knobs together.

RL policy optimization, guardrails, and observability

Why RL can beat fixed rules

- Some good configurations are nonintuitive
- Example pattern the agent might discover:
	- long prompts → PP + FP4 compression
	- short prompts → TP-only + FP8
- Hard-coded rules often miss these interactions across knobs

How the RL loop works

- Observe state
	- GPU util
	- memory util
	- latency
	- queue depth
- Pick action
	- e.g. change precision, batching, PP/TP mode
- Apply policy
- Observe new state
- Compute reward
- Update agent

Visual

	state → action → apply config → observe result → reward → policy update

Training approaches

- Offline:
	- log production traces
	- train in simulator / replay environment
	- libraries like TRL can help for RL workflows
- Online shadow mode:
	- live system explores cautiously
	- decisions do not fully control production at first
- Explore/exploit loop:
	- mostly use current best policy
	- occasionally try alternatives to gather new data

Why PPO is mentioned

- Proximal Policy Optimization (PPO)
- Good for gradual policy updates
- Less likely to thrash between extreme decisions
- Fits continuous / changing runtime environments well

How to reduce oscillation

- Damping:
	- keep an action active for a minimum time / request count
- Hysteresis-like behavior for policy changes
- Override only for critical SLO violations

Guardrails are mandatory

- RL can make unsafe decisions while learning
- So constrain the action space:
	- only reasonable precision modes
	- bounded batch sizes
	- safe parallelism choices
- Start from a solid heuristic default policy
- Let RL tune around that baseline, not from scratch

Reward shaping

- Give strong negative reward for unsafe events:
	- latency hard-limit violation
	- OOM
	- catastrophic throughput collapse
- Can also hard-code forbidden actions regardless of reward
- This gives RL + rule-based safety together

Reward design patterns

- Hard penalty:
	- `reward = tokens_per_second - 1000 * (1 if latency > SLA else 0)`
	- simple, harsh, binary
- Soft penalty:
	- `reward = tokens_per_second - λ * max(0, latency - SLA)`
	- smoother tradeoff, often more stable

Hard vs soft penalty intuition

- Binary penalty:
	- easy to reason about
	- can create abrupt policy jumps
- Continuous penalty:
	- smoother gradients
	- gentler tradeoff decisions
	- often better training stability

Multi-objective view

- This is really throughput vs latency vs sometimes quality
- Weighted sum is simplest
- More advanced framing:
	- partially observable decision process
	- Pareto-front style tradeoffs
- Useful when one scalar reward is too crude

Observability requirements

- Log every:
	- state
	- action
	- reward
	- resulting latency / throughput outcome
- Use structured logging, counters, dashboards
- Needed to debug weird behavior fast

Escape hatch / kill switch

- Always provide fallback to safe static policy
- Example:
	- if p95 latency jumps >50% after enabling RL
	- auto-disable agent actions
	- alert on-call
- This keeps experimentation reversible

Big picture

- RL tuning is not yet mainstream, but it is emerging
- As inference systems gain more runtime knobs, self-tuning becomes more attractive
- Direction of travel: inference servers that learn expert-level tuning from telemetry

Mnemonic: RL is the meta-controller above all the other knobs — let it learn the weird interactions, but cage it with safe actions, shaped rewards, damping, logs, and a kill switch.

Dynamic memory-allocation switching

Why allocators matter

- Inference servers allocate/free thousands of tensors per second
- Bad allocation strategy causes:
	- fragmentation
	- allocation latency
	- false OOMs (memory exists, but not contiguously enough)

Three allocator ideas in this section

	┌──────────────────────┬──────────────────────────────────────────────┐
	│ Allocator style      │ Main idea                                    │
	├──────────────────────┼──────────────────────────────────────────────┤
	│ BFC / caching        │ grab big chunks, subdivide, reuse, coalesce  │
	│ (PyTorch default)    │ avoids frequent cudaMalloc/cudaFree          │
	├──────────────────────┼──────────────────────────────────────────────┤
	│ Buddy + slab         │ buddy handles big power-of-two pages, slab   │
	│                      │ handles many small fixed-size objects fast   │
	├──────────────────────┼──────────────────────────────────────────────┤
	│ cudaMallocAsync      │ CUDA driver-managed stream-ordered memory    │
	│                      │ pools with recycling/coalescing              │
	└──────────────────────┴──────────────────────────────────────────────┘

What fragmentation looks like

- Reserved memory is high
- Allocated memory is much lower
- Yet large allocations fail
- Meaning: memory is free, but split into unusable holes

How to detect it

- `torch.cuda.memory_reserved()`
- `torch.cuda.memory_allocated()`
- Large growing gap = fragmentation warning
- Also use:
	- `torch.cuda.memory_summary()`
	- PyTorch profiler with memory profiling
	- Nsight Systems memory events / UM faults
	- Nsight Compute memory workload analysis

What the book means by "dynamic switching"

- It does NOT really mean flipping allocators inside one live Python process
- That's the confusing part
- In PyTorch, allocator backend is effectively fixed once:
	- `torch` imports and/or
	- first CUDA context is created
- After that, changing `PYTORCH_ALLOC_CONF` inside the same process does NOT reconfigure the allocator

So what is actually possible?

- You can dynamically choose the allocator for the NEXT fresh process
- Pattern:
	1. current worker hits OOM / fragmentation issue
	2. free what you can in parent
	3. spawn a NEW subprocess with `PYTORCH_ALLOC_CONF=backend:cudaMallocAsync`
	4. import torch there
	5. rebuild model there
	6. rerun the request
- So the switching is dynamic at the service / worker-process level, not inside one running CUDA process

This is why the example looks "hacky"

- It catches OOM in parent
- serializes request to disk
- launches child with env var set BEFORE torch import
- child imports torch, creates fresh allocator, rebuilds model, retries generation

Visual

	live worker (default allocator)
				↓ OOM / fragmentation
	free cache + GC in parent
				↓
	spawn fresh child with PYTORCH_ALLOC_CONF=cudaMallocAsync
				↓
	child imports torch fresh
				↓
	child builds model and retries request

Why `cudaMallocAsync` helps

- Stream-ordered pooling
- Driver knows dependency order of frees
- Can recycle/coalesce memory more intelligently
- Often gives many benefits of custom allocator tuning with low manual effort

`max_split_size_mb` tuning

- PyTorch lets you tune splitting behavior via `PYTORCH_ALLOC_CONF=max_split_size_mb:<value>`
- Bigger split size:
	- fewer tiny blocks
	- lower metadata overhead
	- but can leave bigger unused holes
- Smaller split size:
	- more flexible reuse
	- but may flood allocator with tiny fragments

Operational policies

- If fragmentation grows over time:
	- purge unused cache occasionally
	- rolling-restart workers across fleet
- These are intrusive and should be used carefully
- If you need them often, root-cause allocator behavior instead of relying on resets forever

Practical mental model

- Default allocator is good for most workloads
- `cudaMallocAsync` is often a strong default if you want stream-ordered pooling
- "Dynamic switching" in practice means routing / retrying a request in a fresh worker configured differently
- Not hot-swapping allocator internals inside a live Python process

Mnemonic: allocator switching is usually a process-level failover, not a live in-process toggle — detect fragmentation from reserved-vs-allocated gaps, and if needed retry the request in a fresh worker started with `cudaMallocAsync`.

Slab allocators, hot-swappable kernels, and CUDA graph prewarming

Why slab allocators show up in inference engines

- Some allocations repeat at exactly the same size over and over
- Example: per-token buffers / activation tensors of a fixed size like 64 KB
- A slab allocator prepartitions memory into fixed-size blocks for that object size
- Result:
	- very fast reuse
	- near-zero fragmentation for that size class

Important nuance

- Slab memory is usually not returned to the general pool until the whole slab is free
- So slabs are great for stable repeated sizes, not for everything

Good design pattern

- Separate long-lived and short-lived allocations
- Example:
	- model weights → default caching allocator / large static pool
	- ephemeral per-token buffers → slab/custom pool
- This makes allocator choice much easier and cleaner

Fallback after OOM

- Production systems often do NOT fail immediately
- They may retry with:
	- GPU cache cleared
	- more CPU offload
	- more compression
	- alternate allocator / worker setup
- Goal: graceful degradation instead of request failure

Runtime kernel improvements and hot-swappable implementations

Core idea

- Better kernels appear constantly:
	- newer FlashAttention variants
	- megakernels
	- hardware-specific optimizations
- Hot swapping means replacing a slower implementation with a faster one WITHOUT full model reload / service restart

Simple Python-level approach

- Monkey-patch `forward` to call a new library implementation
- Or replace a method with a `torch.compile`'d version
- Effectively swaps the function pointer / callable used by the model

Examples

- old SDPA attention → new FlashAttention-3 style kernel
- raw `forward` → `torch.compile(..., backend="inductor")` compiled version

Why this matters

- 24/7 services can't restart every time a kernel improves
- Zero-downtime upgrades are valuable when reloads would hurt latency or availability

But there are strict conditions

- New kernel must be drop-in compatible
- Numerical output must match old kernel within tolerance
- Thread safety matters:
	- drain queue or use barrier
	- do not patch while another thread is inside the function

Operational rollout pattern

- Canary / shadow test a small traffic slice first
- Compare latency, throughput, and output drift
- If healthy, ramp traffic up
- If not, rollback using feature flag / impl registry

Feature-flag style implementation

- Maintain registry like:
	- `attention_impl = "fast"`
	- `attention_impl = "safe"`
- Toggle implementation through config/flags
- Lets you rollback quickly without code redeploy

Connection to autotuning

- Autotuner finds faster kernel variant for live workload
- Runtime patching promotes that implementation to the default
- This closes the loop:
	- discover better kernel
	- swap it in
	- keep measuring

Memory caution

- Multiple kernel implementations loaded at once consume code + memory
- Too many variants can hurt instruction cache and memory footprint

Per-request kernel choice

- Long sequences may use the fancy optimized kernel
- Short sequences may use the simpler one
- Runtime can choose implementation per request / per shape

Continuous prewarming of CUDA Graphs and caches

Why prewarm

- Cold starts cost latency:
	- JIT compilation
	- cache coldness
	- CUDA Graph capture/setup
- If you can predict demand, you can prepare before traffic arrives

What gets prewarmed

- model weights into caches
- JIT-compiled kernels
- CUDA Graphs for common batch sizes / shapes

How prediction helps

- Use time-series forecasting (ARIMA / Prophet / historical schedules)
- Example:
	- traffic spike every day at 9 a.m.
	- run warm-up shortly before 9 a.m.
	- capture graphs for batch sizes likely during the spike

Why CUDA Graphs fit this well

- They are static / shape-specific
- Inference often uses a small set of common batch sizes: 1, 2, 4, 8, 16...
- So you can maintain a pool of pre-captured graphs for common shapes

Visual

	before spike:
		predict common batch sizes → capture graphs for 1/2/4/8/16
		warm kernels + caches

	during spike:
		incoming batch=16 → reuse prewarmed graph
		one graph launch instead of many individual kernel launches

Benefits

- Less Python→CUDA dispatch overhead
- Faster first execution for common shapes
- Better latency under predictable bursty traffic

Trade-off

- If prediction is wrong, you prewarm graphs that go unused
- So monitor prediction hit rate to make sure prewarming is worth it

Mnemonic: use slabs for repeated tiny buffers, hot-swap kernels behind clean module boundaries, and prewarm CUDA graphs for the shapes tomorrow's traffic is most likely to need.

Graph-pool management, predictive warm-up, and adaptive batching

CUDA Graph pool caveat

- Each captured graph consumes GPU memory for workspace/state
- Too many graphs → memory pressure
- If memory gets tight:
	- evict rarely used graphs
	- keep only common batch/sequence shapes
- Graph patching can adjust small shape differences
- But graph pools are usually faster and simpler in practice

Why graphs reduce CPU overhead

- Without graph:
	- CPU launches many individual kernels
- With graph:
	- CPU launches one `cudaGraphLaunch`
- CPU is freed for preprocessing / routing / other real work

Time-series driven prewarming

- Forecast:
	- RPS
	- average prompt length
	- expected batch sizes
	- long-sequence spikes
- Tools: ARIMA, Prophet, historical schedules
- If forecast says traffic spike is coming:
	- prewarm CUDA Graphs
	- prefetch weights
	- prepare memory pools
	- scale prefill/decode workers
	- raise continuous-batching target size

Important: forecasts drift

- Retrain/update predictors with recent traffic
- Weird calendar effects can break old patterns
- Monitor prediction hit rate and warm-up usefulness

Scale-out warm-up

- When autoscaler starts new GPU workers:
	- run warm-up API calls before live traffic
	- validate engine health
	- trigger JIT compilation
	- allocate memory pools
	- capture CUDA Graph variants
	- warm KV/model caches
- Only add worker to live pool after warm-up completes

Warm both sides of PD

- Prefill workers:
	- representative long prompts
	- graph capture for expected prefill shapes
	- KV cache prep
- Decode workers:
	- graph capture for token-step shapes
	- speculative draft model loaded
	- draft KV cache warmed
	- loop-unrolled decode paths for fixed token counts

Low-priority prewarming

- Prewarm only when GPUs are underutilized when possible
- Use lower-priority CUDA streams
- Live inference streams should preempt warm-up work
- Idle GPU time is free only if warm-up does not collide with real traffic

Grace Blackwell note

- Unified/coherent CPU-GPU memory enables extra tricks
- CPU can stage data into unified memory before GPU needs it
- Can reduce explicit copy calls later

Why this matters

- Moves one-time costs out of the critical path:
	- JIT compile
	- graph capture
	- memory pool setup
	- cache cold misses
- Result: less jitter, fewer latency spikes, more predictable p99

Adaptive batching

Core idea

- Batching increases throughput but can hurt individual latency
- Adaptive batching changes batch size / wait thresholds as load changes

Simple heuristic

	┌────────────────────┬──────────────────────────────┐
	│ Condition          │ Batch behavior               │
	├────────────────────┼──────────────────────────────┤
	│ GPU util > 80%     │ allow larger batches (8/16)  │
	│ GPU util < 20%     │ use batch size 1             │
	└────────────────────┴──────────────────────────────┘

- High load → throughput matters
- Low load → latency matters
- More advanced systems use RL / prediction instead of fixed thresholds

Prefill and decode need separate batching

- Prefill:
	- compute-heavy
	- large prompt matmuls
- Decode:
	- memory-bandwidth-heavy
	- one/few tokens per step
- Modern engines batch them separately
- A prefill batch and decode batch can run independently

Chunked prefill

- Problem:
	- one huge prefill can block decode tasks
	- pipeline bubbles appear because prefill and decode have mismatched durations
- Fix:
	- slice big prefill into smaller chunks
	- interleave decode work between chunks
- Effect:
	- decode stays responsive
	- prefill still progresses
	- pipeline stages stay busier

Visual

	Naive:
		[ huge 10k-token prefill -------------------- ] [decode][decode][decode]
		decode waits behind the giant prefill

	Chunked:
		[prefill chunk][decode][prefill chunk][decode][prefill chunk][decode]
		long prefill is time-sliced so decode can slip through

Decode-maximal scheduling

- vLLM-style scheduling can prioritize keeping decode moving
- Chunked prefill creates gaps where decode batches can run
- This reduces TPOT spikes while maintaining prefill throughput

SARATHI result

- Chunked prefill + piggybacked decode improves scheduling
- Reported ~1.3-1.9× throughput over naive scheduling
- Key idea: steer prefill and decode together instead of letting one block the other

Mnemonic: prewarm the shapes before traffic arrives, batch bigger only when load demands it, and chunk giant prefills so decode can keep slipping through the schedule.

Adaptive chunked prefill sizing

Why chunk size matters

- Huge prefill example: 10,000 tokens
- Naive:
	- one giant prefill monopolizes pipeline
	- decode waits
	- bubbles / latency spikes appear
- Chunked:
	- split into five 2,000-token chunks
	- run decode batches between chunks
	- smoother GPU utilization

Rule of thumb

- Pick chunk size so each prefill chunk takes ~50-100 ms
- This gives frequent scheduling points for decode
- Actual token count depends on:
	- model size
	- GPU hardware
	- attention kernel
	- current load

vLLM-style adaptive scheduling

- Scheduler continuously watches:
	- token queues
	- GPU utilization
	- prefill backlog
	- decode backlog
- Then chooses:
	- process another prefill chunk
	- OR drain a decode batch
- Goal: keep decode responsive while still progressing long prefills

Occupancy-aware chunk sizing

- Chunk size should respect GPU shared-memory and SM occupancy limits
- Scheduler computes tile width `T`
- Shared memory roughly:
	- `shared_bytes = 3 * T * T * sizeof(float)`
	- 3 = Q, K, V tiles
- Then it asks: how many thread blocks can fit per SM with this `T`?

Adaptive rule

- If occupancy < 50%:
	- shrink `T` (often halve it)
	- frees shared memory
	- more blocks can co-reside per SM
- If DRAM traffic is too high and occupancy is healthy:
	- grow `T`
	- more KV reuse
	- fewer global-memory loads

Visual

	T too large:
		big reuse, but only 1 block/SM → low occupancy

	T too small:
		many blocks/SM, but reloads KV too often → DRAM pressure

	good T:
		enough reuse + enough active blocks

Scheduler loop intuition

	while work exists:
		if GPU util < target and prefill pending:
			choose largest prefill
			choose T from occupancy table
			launch one prefill chunk
		elif decode pending:
			form decode-maximal batch
			launch decode batch
		else:
			poll event / wait for work

Important thresholds in example

- `TARGET_UTIL = 0.85`
- `OCC_THRESHOLD = 0.5`
- `BLOCK_THREADS = 256`
- `SHMEM_LIMIT = 256 KB`

Decode-maximal piggybacking

- Between prefill chunks, scheduler forms one decode batch from ready requests
- Short interactive decodes slip into gaps
- Users with short prompts do not wait behind a giant long-prefill request

What to log

- chosen tile/chunk size `T`
- occupancy
- GPU utilization
- prefill queue depth
- decode queue depth
- TPOT / TTFT impact
- This verifies the scheduler is actually keeping utilization high

Policy name

- Utilization-maximization policy
- Similar to OS scheduler trying to keep CPU busy
- Here: keep SM slots filled and minimize dead GPU time

Mnemonic: choose prefill chunks that create decode opportunities every ~50-100 ms, shrink T when occupancy is poor, grow T when DRAM is thrashing, and let decode batches piggyback through the gaps.

Adaptive batching fairness and topology-aware scheduling

Why chunked prefill helps user experience

- Large-context prefills cannot monopolize the pipeline
- Interactive decode steps get frequent chances to run
- Small latency-critical requests are served quickly
- GPU stays busy without making users wait behind giant context builds

Scheduling rule of thumb

- Treat prefill and decode as separate queues with separate SLAs
- Decode is user-facing and latency-sensitive
- Prefill is bulk context construction
- So:
	- run decode in dedicated time slices / streams first
	- use leftover cycles for prefill chunks

Example time budget

- Reserve ~1-5 ms windows for ready decode work
- Then schedule a prefill chunk if decode queue is drained or under control
- This reduces perceived lag while still processing large prompts

Producer-consumer execution model

- Common engine pattern:
	- one thread prepares decode inputs
	- another thread prepares new prefills
	- scheduler merges them into one optimized execution stream
- Goal: overlap CPU prep, GPU scheduling, and actual GPU execution

Multinode PD deployment

- Prefill requests can go to prefill nodes
- Decode requests can go to decode nodes
- Each node pool can use phase-specific hardware

Hardware specialization

	┌──────────┬─────────────────────────────────────────────┐
	│ Phase    │ Hardware preference                         │
	├──────────┼─────────────────────────────────────────────┤
	│ Prefill  │ high FLOPS / Tensor Cores, compute-heavy    │
	│ Decode   │ high HBM capacity + bandwidth, memory-heavy │
	└──────────┴─────────────────────────────────────────────┘

Trade-off

- Heterogeneous nodes can save cost
- But they complicate dynamic load balancing
- If traffic ratio shifts, decode-optimized GPUs may be bad at sudden prefill bursts
- Homogeneous nodes simplify scheduling
- Specialize hardware only when workload ratio is predictable enough

Batching rules

- Decode:
	- batch ready decode calls whenever possible
	- improves throughput and reduces launch overhead
- Prefill:
	- avoid mixing very short and very long prompts
	- use length buckets, e.g. nearest 512-token bucket
	- reduces padding waste

Token-level scheduler

- Forms decode batches at each generation step
- Waits a few ms to gather ready tokens
- Caps batch size by memory and occupancy limits
- Uses fairness rules:
	- round robin = everyone gets turns
	- maximum delay = nobody waits beyond a deadline

Net effect

- Higher GPU utilization
- Better aggregate latency
- Fewer idle gaps
- Less head-of-line blocking from huge prefills

Congestion-aware and topology-aware scheduling

Why topology matters

- Modern racks like GB200 NVL72 / GB300 NVL72 connect 72 GPUs with NVLink/NVSwitch
- Each GPU can reach peers through high-bandwidth fabric
- But raw bandwidth alone is not enough
- Bad traffic patterns can still oversubscribe links / switches

NVL72 numbers to remember

- 72 GPUs in one NVSwitch domain
- B200: 180 GB HBM per GPU
- B300: 288 GB HBM per GPU
- Up to ~1.8 TB/s bidirectional NVLink per GPU
- Over 130 TB/s aggregate cross-sectional bandwidth

NVLink / NVSwitch mental model

- NVLink = high-speed point-to-point GPU links
- NVSwitch = rack-scale switch fabric connecting GPUs all-to-all
- In NVL72:
	- each GPU has many NVLink ports
	- traffic goes through NVSwitch
	- any GPU can reach any other GPU in roughly one fabric hop

Visual

	GPU0 ─┐
	GPU1 ─┤
	GPU2 ─┼── NVSwitch fabric ── GPU37
	...  ─┤                    ── GPU71
	GPU71─┘

Important caveat

- All-to-all topology does NOT mean infinite bandwidth
- Per-port and per-switch limits still exist
- Many-to-one traffic can oversubscribe ingress

Example bottleneck

	many GPUs sending KV to one decode GPU:

	GPU1 ─┐
	GPU2 ─┼──► GPU9
	GPU3 ─┘

	→ GPU9 ingress link / switch path becomes bottleneck

What topology-aware scheduling tries to do

- Use link-utilization telemetry
- Avoid congested NVLink/NVSwitch paths
- Route transfers to less-busy peers
- Schedule collective operations in waves instead of all at once
- Keep latency low while preserving throughput

Mnemonic: adaptive batching keeps decode from starving behind prefill, while topology-aware scheduling keeps GPU traffic from piling onto the same NVLink/NVSwitch path.

NVLink congestion telemetry and adaptive process-GPU mapping

Finite bandwidth reminder

- NVLink/NVSwitch is huge, but not infinite
- NVLink 5 port: ~100 GB/s per direction
- GB200/GB300: 18 NVLink links per GPU
- Per GPU: up to ~1.8 TB/s bidirectional throughput
- Under balanced load, NVSwitch looks nonblocking
- Under skewed load, links/switches can still congest

Congestion patterns

- Many-to-one:
	- many GPUs send KV/activations to one GPU
	- receiver ingress becomes bottleneck
- All-at-once exchange:
	- many GPUs communicate at the same time
	- switch paths can oversubscribe
- Cross-rack / cross-node:
	- leaves NVLink domain
	- uses InfiniBand/Ethernet
	- higher latency and lower bandwidth

Latency rough numbers

- NVSwitch hop: <1 µs
- InfiniBand NDR hop: ~5-10 µs
- So locality matters a lot

Do not guess topology

- Query it programmatically:
	- CUDA topology APIs
	- NCCL topology hints
	- NVML
	- DCGM
	- Fabric Manager / NVSwitch tooling
- Goal: know which GPUs are close, which links are shared, and which paths are hot

Real-time link telemetry

- Need to observe congestion before scheduling around it
- Useful counters:
	- bytes per NVLink port
	- per-link throughput
	- error rates
	- GPU-pair traffic
	- NVSwitch/uplink hotspots
- Preferred tools:
	- NVML / `nvmlDeviceGetNvLinkUtilizationCounter`
	- DCGM
	- DCGM exporter → Prometheus → Grafana
	- Nsight Systems for timeline views

Monitoring caveat

- NVLink counters have overhead
- Sample at a reasonable interval
- Do not poll so aggressively that monitoring hurts serving

What Nsight can reveal

- Transfers overlapping badly
- Pipeline stages all sending at the same time
- Kernels waiting on communication
- Specific phases blocked on NVLink/NVSwitch traffic

Scheduler reactions

- If link hot:
	- insert slight delay
	- reschedule transfer later
	- reroute to less-busy GPU/path
	- reassign GPU roles
	- schedule collectives in waves
- This is feedback-driven communication scheduling

Adaptive process-GPU mapping

Core idea

- Map model processes/layers to GPUs based on communication cost
- Keep heavy tensor transfers local and balanced
- Avoid placing communicating layers on far/congested GPUs

Problem example

	Bad mapping:
		layer 0 on GPU0  ───── activation ─────► layer 2 on GPU2
		path is far / congested

	Better mapping:
		layer 0 on GPU0  ──► layer 2 on nearby GPU1
		shorter / faster / less congested path

NVTAGS

- NVIDIA Topology-Aware GPU Selection
- Automatically maps processes to GPUs using:
	- fabric distance
	- link metrics
	- observed communication patterns
- Profiles topology and assigns GPU affinity to minimize communication cost

When remapping helps

- Large activation tensors between pipeline stages
- Heavy KV transfers between specific GPUs
- One link/switch path becoming saturated
- Naive process placement crossing expensive paths

If not using NVTAGS

- Build/use a topology map manually
- Group close GPUs together for strongly communicating processes
- Keep pipeline-adjacent layers on nearby GPUs where possible

Visual

	Model graph:       Hardware graph:

	L0 → L1 → L2       GPU0 --fast-- GPU1 --fast-- GPU2
											 \                       /
												---- slower path ------

	Good placement:
		L0 on GPU0, L1 on GPU1, L2 on GPU2
		→ activations follow fast neighboring links

Mnemonic: NVSwitch is an enormous highway, not teleportation — watch the link counters, avoid many-to-one traffic, and place communicating layers on nearby GPUs.

Adaptive remapping and NCCL all-reduce tuning

Adaptive process-GPU remapping

- Remapping can happen:
	- at initialization
	- between inferences
	- periodically after profiling
- Goal: put high-traffic process pairs close together
- If process 3 and process 4 exchange tens of GB/s:
	- keep them on same node / same fast fabric when possible
- Compute-heavy, low-communication processes can tolerate more distance

Optimization view

- Model processes/layers = graph nodes
- Data movement between them = graph edges
- Edge weight = GB/s transferred
- Goal = graph partitioning / min-cut:
	- keep heavy edges local
	- push light edges across slower links if needed

Remapping caveat

- Moving a process/layer means moving model weights
- Layers can be many GB
- Do not remap too frequently
- Best times:
	- between large batches
	- during quiet periods
	- when mapping will remain stable long enough to pay back the move cost

Visual

	Before:
		process 3 ── 50 GB/s cross-node ── process 4
		→ slow / congested path

	After:
		process 3 and process 4 placed closer
		→ traffic stays within node / NVSwitch domain
		→ less cross-node bandwidth, lower latency

NCCL collectives

- NCCL = NVIDIA Collective Communications Library
- Handles GPU collectives like:
	- all-reduce
	- all-gather
	- broadcast
	- reduce-scatter
- In inference, collectives show up in:
	- tensor-parallel all-reduce
	- MoE expert output gathering
	- parameter / activation broadcasts

What all-reduce means

- Each GPU starts with one partial result
- All-reduce combines all partial results
- Then every GPU receives the final combined result

Visual: all-reduce sum

	Start:
		GPU0 has A
		GPU1 has B
		GPU2 has C
		GPU3 has D

	After all-reduce SUM:
		GPU0 has A+B+C+D
		GPU1 has A+B+C+D
		GPU2 has A+B+C+D
		GPU3 has A+B+C+D

Why inference needs it

- In tensor parallelism, each GPU computes a slice of a layer output
- All-reduce combines slices so every GPU has the needed combined result

Ring all-reduce

- GPUs form a loop
- Data chunks circulate around the ring
- Every GPU sends to one neighbor and receives from another
- Very bandwidth-efficient for large messages
- But latency grows roughly linearly with GPU count

Visual: ring

	GPU0 ──► GPU1 ──► GPU2 ──► GPU3
	 ▲                              │
	 └──────────────────────────────┘

	Good for:
		large messages
		bandwidth-dominated transfers
		balanced, uncongested fabric

	Bad for:
		small latency-sensitive messages
		very large GPU counts
		congested long ring paths

Tree all-reduce

- GPUs form a logical tree
- Reduce upward, then broadcast downward
- Fewer sequential steps: about log2(N)
- Lower latency for many GPUs / small messages
- May not saturate every link as fully as ring

Visual: tree reduce + broadcast

				GPU0
			 /    \
		GPU1    GPU2
		/  \    /  \
	GPU3 GPU4 GPU5 GPU6

	reduce: leaves → root
	broadcast: root → leaves

Ring vs tree

	┌──────────────┬────────────────────────────┬────────────────────────────┐
	│ Algorithm    │ Strength                   │ Weakness                   │
	├──────────────┼────────────────────────────┼────────────────────────────┤
	│ ring         │ max bandwidth for big data │ latency grows with N       │
	│ tree         │ low latency, O(log N)      │ may underuse bandwidth     │
	│ hierarchical │ mixes local ring/tree      │ more complex tuning        │
	└──────────────┴────────────────────────────┴────────────────────────────┘

72-GPU intuition

- Ring: ~71 sequential hops
- Tree: ~log2(72) ≈ 6-ish sequential levels
- So tree often wins for latency-sensitive reductions at huge GPU count

NCCL tuning

- NCCL normally chooses ring/tree/hierarchical heuristically
- Scheduler can override/tune based on:
	- message size
	- topology
	- congestion
	- GPU count
- Example knob:
	- `NCCL_ALGO=Tree`
- General rule:
	- small messages / latency-sensitive → tree
	- huge messages / bandwidth-dominated → ring if fabric is not congested
	- multi-node / mixed topology → hierarchical

Overlap with compute

- NCCL communication can run on one CUDA stream
- GEMM compute can run on another
- Good schedule overlaps all-reduce with useful compute when dependencies allow

Visual

	stream 0: [GEMM compute --------------]
	stream 1:        [NCCL all-reduce ----]
									 overlap hides comm cost

Mnemonic: all-reduce means everyone contributes a partial and everyone gets the total; ring maximizes bandwidth by walking the loop, tree minimizes latency by reducing up and broadcasting down.

Rotating rings, wave scheduling, and multirail RDMA

Rotating ring endpoints

- Ring is simple and bandwidth-efficient
- But fixed ring order can overload the same links repeatedly
- Especially the wraparound / critical GPU-pair links
- Fix: rotate or shuffle ring ordering across collectives

Visual: fixed ring

	collective 1:
		GPU0 → GPU1 → GPU2 → GPU3 → GPU0

	collective 2:
		GPU0 → GPU1 → GPU2 → GPU3 → GPU0

	→ same neighbor pairs get hammered every time

Visual: rotated ring

	collective 1:
		GPU0 → GPU1 → GPU2 → GPU3 → GPU0

	collective 2:
		GPU1 → GPU2 → GPU3 → GPU0 → GPU1

	collective 3:
		GPU2 → GPU3 → GPU0 → GPU1 → GPU2

	→ heavy link duty moves around over time

NCCL note

- NCCL already alternates inside/outside ring directions on successive calls
- Extra rank shuffling helps if workload is persistently imbalanced
- Scheduler can reindex GPUs in communicator with permuted rank order
- Effect: no single NVLink/GPU stays on the critical path forever

Wave scheduling of collectives

- Instead of one giant all-to-all across every GPU at once
- Split communication into smaller phased waves
- Reduces instantaneous fabric load

Problem example

- 72 GPUs all-to-all exchange
- Naive:
	- each GPU sends to 71 others at once
	- 72 × 71 messages flood NVSwitch/NIC paths
	- huge traffic spike

Wave version

- Split into 4 waves of 18 GPUs
- Each wave does a smaller exchange
- Stagger start times so one wave partly drains before next starts
- This is temporal multiplexing: spread traffic over time

Visual

	Naive:
		t0: [all 72 GPUs communicate at once]  → giant spike

	Waves:
		t0: [wave 1: 18 GPUs]
		t1:        [wave 2: 18 GPUs]
		t2:               [wave 3: 18 GPUs]
		t3:                      [wave 4: 18 GPUs]
		→ smoother traffic, less congestion

NCCL relation

- NCCL internally slices data for pipelining, especially in rings
- Scheduler can also orchestrate smaller collectives externally
- Useful when whole-cluster collectives would create a burst

Overlap compute and communication

- If reduction happens in waves:
	- later communication waves can overlap with next-layer compute
- Some GPUs compute while others finish communication
- This time-shifts network traffic into otherwise idle gaps

Visual

	time →
	comm wave 1: [reduce]
	compute:          [next layer compute -----]
	comm wave 2:          [reduce]
	compute:                       [next compute]

Why wave scheduling helps

- Avoids massive one-time NVSwitch/NIC spike
- Improves fairness between transfers
- Keeps bandwidth busy without overwhelming it
- Similar to pacing network traffic to avoid burstiness

Multinode / multirack communication

- Beyond one NVL72 rack, GPUs communicate through NICs and network switches
- NVLink/NVSwitch no longer connects every GPU directly
- Internode paths use InfiniBand / Ethernet
- These are slower and higher-latency than intrarack NVLink

GPUDirect RDMA

- Lets remote GPUs read/write GPU memory directly through NICs
- Avoids CPU host-memory bounce buffers
- Still limited by network bandwidth and congestion

Visual

	Bad path:
		GPU → CPU RAM → NIC → network → CPU RAM → GPU

	GPUDirect RDMA:
		GPU HBM → NIC → network → NIC → GPU HBM

Multirail networking

- High-end GPU servers often have multiple NICs / IB ports
- Using two NICs can approach ~2× throughput vs one NIC
- NCCL can split rings/trees across NICs
- Separate NICs often connect to separate network rails / switches

NCCL_CROSS_NIC

- Controls whether NCCL can use different NICs across nodes for one collective
- With good topology, enable for large collectives
- Example effect:
	- half traffic exits NIC1
	- half traffic exits NIC2
	- less bottleneck on one rail

Congestion-aware internode scheduling

- If one NIC/path is saturated:
	- move traffic to another NIC
	- use alternate route
	- split transfers into smaller chunks
	- let network adaptive routing balance flows
- Application can influence this with NCCL channel/NIC assignment

NIC affinity

- Bind each GPU to the closest NIC
- Usually same PCIe root complex / same CPU or NVSwitch complex
- Reduces local PCIe/CPU contention before data even reaches the network

Mnemonic: rotate rings so one link is not always hot, send collectives in waves so the fabric is not flooded all at once, and stripe multinode traffic across NIC rails with GPUDirect RDMA.

GPU-NIC affinity, MoE expert placement, and dynamic congestion control

GPU-NIC affinity

- Map each GPU to its closest NIC
- Use NVML / NCCL / topology files to learn locality
- Example mapping:
	- GPUs 0-3 → NIC 0
	- GPUs 4-7 → NIC 1
- Goal: shortest path out of the node

Why affinity matters

- Bad mapping:
	- GPU traffic crosses PCIe/CPU/NVSwitch boundaries before reaching NIC
	- adds local contention before the data even enters the network
- Good mapping:
	- GPU sends through nearest NIC
	- lower latency, less internal node congestion

Manual overrides

- NCCL autotuning handles many cases
- But persistent hotspots may require manual/static topology hints
- If NIC 0 is saturated:
	- scheduler can move some GPU traffic to NIC 1 next round
	- or assign NCCL channels to less-used NICs

Extreme cluster case: massive MoE

- MoE inference creates heavy cross-node traffic
- Token router sends tokens to expert GPUs
- Results are gathered back
- Without tuning, network becomes the bottleneck

Two ways to reduce cross-node traffic

- Replicate hot data/experts
	- popular expert exists on multiple nodes
	- tokens can use nearer copy
- Hierarchical aggregation
	- aggregate within node first
	- exchange compact summaries across nodes
	- avoid full all-to-all across every GPU

NVIDIA SHARP

- Offloads some aggregation into switch hardware
- Helps reduce communication bottlenecks
- Works well with adaptive routing in large clusters

Network as a schedulable resource

- Treat network like GPUs / HBM:
	- monitored
	- allocated
	- balanced
	- adapted dynamically
- At large scale, one slow link can throttle the whole distributed inference pipeline

MoE expert rebalancing

Why MoE creates special traffic

- Experts live on different GPUs
- Each token is routed to top-k experts
- This creates all-to-all-like traffic:
	- tokens go to expert GPUs
	- expert outputs come back

Problems with static expert placement

- Popular experts become hotspots
- Experts often used together may be placed far apart
- Tokens repeatedly travel long NVLink/NVSwitch/network paths

Expert rebalancing

- Periodically move experts to better GPUs/nodes
- Based on logged routing/traffic stats
- Goals:
	- spread popular experts
	- colocate experts that are often used together
	- keep traffic inside local NVLink/NVSwitch groups when possible

Visual

	Bad placement:
		expert 5 on GPU0
		expert 19 on GPU71
		same tokens often need both → long fabric path every time

	Better placement:
		expert 5 + expert 19 on same GPU/node/group
		→ traffic stays local

Hot expert example

- Expert 7 receives too many tokens
- Scheduler can:
	- move it to a less-busy GPU
	- duplicate it if supported
	- split token load across copies

Rebalancing caveat

- Experts are model weights
- Moving them can mean transferring GBs
- Do it infrequently:
	- maintenance windows
	- between large batches
	- after enough routing stats prove the change is worth it

Expert grouping / bucketing

- Put commonly co-used experts in same group of GPUs
- Group can be:
	- same server
	- same NVSwitch island
	- same rack-local domain
- Reduces cross-group traffic

Graph partitioning view

- Nodes = experts / GPUs
- Edges = token traffic between router and experts, or expert-expert co-use
- Edge weight = traffic volume
- Partition graph to minimize heavy edges crossing topology boundaries

Key idea

- MoE regrouping rearranges the workload to fit the network
- Not the network to fit the workload

Dynamic congestion-aware scheduling

- Continuously monitor link/NIC/switch telemetry
- Adjust in real time:
	- throttle
	- reroute
	- reorder
	- delay
	- remap

Temporal load balancing

- If a link is maxed out by one big transfer
- And another transfer wants same link
- Scheduler delays the second by a few ms
- Better to queue briefly than overload the fabric

Visual

	Bad:
		t0: transfer A uses link 0
		t0: transfer B also uses link 0
		→ queue buildup / congestion

	Better:
		t0: transfer A uses link 0
		t1: transfer B uses link 0 after A drains
		→ smoother utilization

Analogy

- Like network packet routing / backpressure
- Small scheduling delays prevent bigger congestion stalls

Mnemonic: bind GPUs to nearby NICs, colocate experts that co-activate, replicate hot experts when needed, and delay/reroute transfers before one saturated link slows the whole cluster.


