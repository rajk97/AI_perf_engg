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

