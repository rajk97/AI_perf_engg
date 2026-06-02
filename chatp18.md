Chapter 18 — Advanced prefill-decode and KV cache tuning

Where this fits

- Builds on chapter 17 (high-level PD scaling)
- Goes lower level: kernels, KV cache layout, GPU-to-GPU prompt transfer, adaptive scheduling
- Goal: lower decode latency, higher per-GPU throughput, hit strict SLOs

Optimized decode kernels

Why decode needs special kernels

- Decode = single-token forward pass per step
- Memory-bound + many tiny launches
- Standard kernels leave GPU underused
- Fixes target: fewer launches, fused ops, better memory hierarchy use

Three notable kernels

  ┌──────────────┬───────────┬────────────────────────────────────────────┐
  │ Kernel       │ Origin    │ Idea                                       │
  ├──────────────┼───────────┼────────────────────────────────────────────┤
  │ FlashMLA     │ DeepSeek  │ fused MLA decode kernel                    │
  ├──────────────┼───────────┼────────────────────────────────────────────┤
  │ ThunderMLA   │ Stanford  │ "megakernel" + dynamic packing             │
  ├──────────────┼───────────┼────────────────────────────────────────────┤
  │ FlexDecoding │ PyTorch   │ JIT-compiled decode kernel via Triton      │
  └──────────────┴───────────┴────────────────────────────────────────────┘

FlashMLA (DeepSeek)

- "FlashAttention for decode": targets the single-token step
- Fuses attention ops → one kernel handles many heads + time steps
- Increases arithmetic intensity → math units stay busy at small batch
- Cuts kernel-launch overhead + improves memory access pattern
- Pushes MLA closer to compute-bound regime (vs GQA / MQA)
- Open source, integrated in SGLang and vLLM
- Use it to boost per-token throughput without architectural changes

Visual: arithmetic intensity ladder

  MQA   ──┐
  GQA   ──┤  memory-bound region
  MHA   ──┘
       MLA → moves toward compute-bound (math units saturated)

ThunderMLA (Stanford)

- Builds on FlashMLA
- A single fused decode "megakernel" for attention + scheduling
- Eliminates the "tail effect":
  - in a batch of variable-length decodes, some sequences finish early
  - GPU partially idles waiting on stragglers
  - ThunderMLA dynamically repacks remaining streams in same kernel
- 20–35% faster decode throughput vs FlashMLA
- Benefits amplified by big L2 caches + FP8/FP4 Tensor Cores on modern GPUs

Visual: tail effect

  Without ThunderMLA:
  seq A: ████████░░░░░░  (idle after finishing)
  seq B: ████████████░░  (idle)
  seq C: ██████████████  (still running, GPU half-used)

  With ThunderMLA:
  remaining streams get repacked → GPU stays full

FlexDecoding (PyTorch)

- Decode backend of `torch.nn.attention.flex_attention`
- JIT-compiles a specialized kernel for decode (Q_len = 1) over growing KV
- Uses TorchInductor + Triton → warp specialization, async copies
- Reuses compiled kernel across decode steps as long as shapes/dtypes match
- Supports masks, biases, GQA, PagedAttention
- In place KV management

Recommended PyTorch settings

- `torch.compile(mode="max-autotune")` for stable latency-critical decode
- Keep capture boundary narrow (per-layer / per-attn block)
  - reduces graph invalidations from ragged batches
- Prefer Transformer Engine FP8 (MXFP8) for prefill + decode
- FP4 (NVFP4) only when accuracy permits and perf wins (still maturing)
- `torch.set_float32_matmul_precision("high")` for TF32 fallback on FP32 ops

Nested jagged tensors (NJT)

- Native support for ragged batches of variable-length sequences
- Three sequences of different lengths → one tensor + offsets array
- No padding waste → ideal for decode-time batching

Visual: jagged batching

  seq1: [t1 t2 t3]
  seq2: [t1 t2 t3 t4 t5 t6]
  seq3: [t1 t2]

  NJT representation:
    values:  [t1 t2 t3 | t1 t2 t3 t4 t5 t6 | t1 t2]
    offsets: [0, 3, 9, 11]

PagedAttention integration

- FlexDecoding maps logical KV blocks → physical cache layout via BlockMask
- No extra copies; works with vLLM-style paged caches (e.g. LMCache)
- Captured tensors let mask/bias values vary per iteration without recompile
- Particularly useful for MoE inference and custom attention sparsity

Why this matters

- Full Python flexibility for arbitrary attention patterns
- No custom CUDA needed
- Performance close to hand-tuned dense attention kernels
- Future-proof: new sparsity ideas can be tried in Python

Mnemonic: FlashMLA fuses, ThunderMLA repacks the tail, FlexDecoding JIT-compiles per pattern — three answers to "decode is small, slow, and irregular."

Tuning KV cache: disaggregated pools and prefix sharing

Profiling note

- Verify kernel overlap with Nsight Systems (CUDA hw traces)
- Use Nsight Compute for memory + link metrics
- Custom kernels are usually picked up by PyTorch / vLLM / SGLang quickly

Why KV cache becomes a first-class shared resource

- With disaggregation, KV outlives a single request and can move between nodes
- Treat it like a cluster-wide store, not per-GPU scratch
- Two big techniques: distributed KV pool, prefix reuse across requests

Disaggregated KV cache pool

- Decouple KV storage from individual GPUs
- Spread KV blocks across the whole cluster's GPU memory
- Spill to CPU DRAM (Grace Blackwell / Vera Rubin unified memory) or NVMe
- Multitier hierarchy ≈ OS virtual memory:
  - hot KV  → GPU HBM
  - warm KV → CPU RAM
  - cold KV → NVMe

Visual

  ┌─────────────────────────────────────────────────────────┐
  │  Disaggregated KV pool                                  │
  ├──────────────┬──────────────┬───────────────────────────┤
  │ GPU HBM      │ CPU DRAM     │ NVMe                      │
  │ active KV    │ overflow     │ long-tail / cold context  │
  └──────────────┴──────────────┴───────────────────────────┘
                ↑ async paging, overlapped with compute

Worked KV size example

- Model: 70B, 80 layers, 32 heads × 128 dim → hidden 4096
- Per token per layer: 2 × 4096 floats = 8192 floats (K + V)
- Per token total: 80 × 8192 = 655,360 floats
- FP16 (2 B/float) → ~1.31 MB per token
- 250,000-token chat session → ~328 GB just for KV
- FP8 + selective-layer caching → ~100–150 GB
- Still won't fit on one GPU with weights → need to spill or recompute

Why this is a big deal

- Without pool: must truncate context OR recompute (O(N²) attention)
- With pool: page older blocks out, fetch back when needed
- Async paging hides under compute → near-zero stall

Other wins from a global pool

- Any decode node can pick up any request → failover + load balance
- Crash mid-gen → KV survives in pool, another node continues
- Cross-request reuse: shared prefix computed once, used everywhere
- Locality-aware scheduling: send decode to node closest to its KV

Trade-off

- Extra hop to fetch KV from pool
- But still cheaper than recomputing quadratic attention
- Trades memory + cold storage for compute

KV cache reuse and prefix sharing

When prefixes overlap

- Multiturn conversations
- Shared system prompts ("You are a helpful assistant…")
- Attached documents reused across requests
- Skipping prefill on the matched prefix saves huge compute

vLLM automatic prefix caching

- Built on PagedAttention
- Each 16-token block has a content hash
- Global hash table maps hash → KV block
- Match → copy the cached KV instead of recomputing

Visual

  new prompt: [block1][block2][block3][block4]
                hash    hash    hash    hash
                  ↓       ↓       ↓       ↓
               cache   cache   cache    miss
              ────────────────────────► only block4 needs prefill

Cache management

- Match strategy: exact prefix match (simple, robust)
- Partial-overlap merging is hard — usually skipped
- Eviction: LRU or "likelihood of reuse" heuristics
- Prompt-tree structures track shared prefix subtrees

Locality-aware routing for prefix hits

- If node A already holds the prefix's KV → route to A
- Avoids cross-node KV transfer
- "Send compute to the data" (classic distributed-systems rule)

Why a global KV view matters

- Without it: same conversation hitting different nodes recomputes prefix
- With it: any node sees any cached prefix → fewer redundant prefills
- Especially valuable in disaggregated clusters where requests bounce around

Trade-offs of caching

- More cached KV = more memory used
- Need eviction policies
- Hot prefixes worth keeping, rare ones worth dropping
- Goal: maximize hit rate within memory budget

Mnemonic: KV cache wants to be a cluster-wide tiered storage system — pool it across GPUs/RAM/NVMe, hash prefixes for reuse, and route compute toward where the KV already lives.

KV cache memory layout, POD-Attention, GB200, and fast PD transfer

KV cache size grows fast

- Per stream: ≈ num_layers × 2 × seq_len × d_head
- Many concurrent decodes → huge HBM footprint
- Keep active KV in GPU memory (latency)
- Spill older KV to CPU / NVMe / compress when possible

Paged KV layout (FlashMLA style)

- Allocate KV in fixed-size pages
- Active sequence's pages live contiguously
- Benefits: fewer cache misses, less DRAM traffic, better coalesced access

Prefix compression / eviction

- Long conversations → context window slides
- Old tokens won't be attended to → drop or compress their KV
- Saves memory + HBM bandwidth on long sequences
- ⚠ Safe only when attention is sliding-window or otherwise restricted
- ⚠ NOT safe for layers with full-context attention or retrieval hooks unless evaluated

POD-Attention (SM-aware CTA scheduling)

- Reorganizes attention to reduce HBM traffic
- One kernel launches enough CTAs to cover BOTH prefill + decode work
- Each CTA at runtime inspects its SM and per-SM counters
- Picks role: prefill or decode, based on what's already running there
- Result: prefill + decode colocate on the same SM

Visual

  Without POD:
  SM0: ████prefill████      (bursty HBM reads)
  SM1: ████prefill████
  SM2:                ░░decode░░░  (bursty later, mem-bound)
  → bursty mem traffic, low overlap

  With POD-Attention:
  SM0: prefill+decode mixed
  SM1: prefill+decode mixed
  SM2: prefill+decode mixed
  → smoother memory pressure, shared KV in L2, ~29% speedup

Key insight

- Decouples HW CTA→SM assignment from SW CTA→role assignment
- Hardware/software codesign to minimize data movement
- Reuses KV in L2 across phases on the same SM

GPU + CPU-GPU superchip improvements

- Higher HBM bandwidth + bigger L2 = direct win for memory-bound decode
- Grace Blackwell GB200 NVL72:
  - 36 Grace CPUs + 72 Blackwell GPUs in one rack
  - ~30 TB unified memory across CPU + GPU
  - One logical "decode unit" can hold millions of tokens of KV
- Memory tiers on NVL72:
  - HBM (Blackwell GPU)     → active KV
  - LPDDR5X (Grace CPU)     → cooler / older KV
  - NVMe                    → cold context
- Even on NVL72, prefill and KV offload still matter for million-token contexts

Macro + micro must combine

- Macro: disaggregation, routing, KV pool
- Micro: FlashMLA / ThunderMLA / FlexDecoding, paged layouts, POD-Attention
- Hardware: GB200-class memory bandwidth + capacity
- All three layers together = real decode efficiency

Fast KV cache transfer between prefill and decode

Why transfer speed is critical

- Prefill output = KV for all prompt tokens
- Has to land on the decode worker before decode can run
- Slow transfer → wipes out the parallelism gain of disaggregation
- Target: a few ms, not hundreds

KV size rough formula

- KV size ≈ 2 × L × N × (h × d)
  - L = layers, N = prompt tokens, h = heads, d = head_dim
  - factor 2 = keys AND values
- Example: L=40, h=16, d=64, N=1000 → ~40K KV vectors, hundreds of MB
- N=5000 → 5× more (transfer cost grows linearly in tokens)

Naive (bad) transfer path

  prefill GPU → CPU memcpy → TCP socket → CPU on decode → GPU memcpy
  → adds hundreds of ms for large prompts

Recommended (fast) path

  prefill GPU HBM ──── GPUDirect RDMA ────► decode GPU HBM
  - no CPU bounce
  - NIC reads/writes GPU memory directly (zero-copy)
  - latency ~few ms even for big prompts

Practical tip from the book

- Collate small PagedAttention blocks into LARGER buffers before sending
  - lots of tiny RDMA ops = high overhead
  - one big op = bandwidth-dominated, latency-bounded
- Send via GPUDirect RDMA, not CPU sockets

Mnemonic: pack KV pages into contiguous buffers, ship via GPUDirect RDMA, colocate prefill+decode on the same SM (POD), and lean on NVL72 unified memory for the cold tail.

5/21/26

Zero-copy GPU-to-GPU KV transfer

Core idea

- Use RDMA over high-speed fabrics — no CPU bounce, no extra copies
- Within a node: NVLink / NVSwitch
- Across nodes: InfiniBand (or RoCE) via GPUDirect RDMA
- Prefill GPU writes KV directly into decode GPU's HBM

NVIDIA NIXL (NVIDIA Inference Xfer Library)

- Plugin architecture for zero-copy data movement
- Backends: NVLink, UCX fabrics, GPUDirect Storage
- Used by NVIDIA Dynamo and vLLM + LMCache integration
- Writes KV tensors directly into remote GPU memory

Important: a full RDMA write still needs the data to LAND before decode can use it for that request

- "Async" overlaps with OTHER ongoing decodes, not within the same request
- New request's first token waits until its KV is fully landed
- But that's only ~few ms over RDMA vs hundreds over CPU+TCP

Five common transfer strategies

  ┌─────────────────────┬───────────────────────────────────────────────┐
  │ Strategy            │ How it works                                  │
  ├─────────────────────┼───────────────────────────────────────────────┤
  │ Prefill-side push   │ prefill RDMA-writes KV into decode buffer,    │
  │                     │ then moves on to other work                   │
  ├─────────────────────┼───────────────────────────────────────────────┤
  │ Decode-side pull    │ decode RDMA-reads KV from prefill GPU when    │
  │                     │ ready; receiver controls timing               │
  ├─────────────────────┼───────────────────────────────────────────────┤
  │ Shared-mem (IPC)    │ same host: CUDA IPC handle, NVLink/NVSwitch   │
  │                     │ memcpy — zero-copy, no network                │
  ├─────────────────────┼───────────────────────────────────────────────┤
  │ Connector / queue   │ vLLM Pipe/LookupBuffer abstracts transport;   │
  │                     │ swap RDMA, IPC, pub-sub (NATS for control)    │
  ├─────────────────────┼───────────────────────────────────────────────┤
  │ Nonblocking overlap │ KV write happens while decode keeps producing │
  │                     │ tokens for OTHER requests                     │
  └─────────────────────┴───────────────────────────────────────────────┘

Visual: nonblocking overlap

  decode GPU timeline:
    [A tok][B tok][C tok][A tok][B tok][C tok][A tok]...
                                                  ▲
  RDMA in background: ──── D's KV streaming in ───┘
  once D's KV lands → D joins the rotation
    [A tok][B tok][C tok][D tok][A tok][B tok]...

  → KV transfer ~5 ms is hidden behind active decode → near-zero added latency to other requests

Why we package KV before sending

- vLLM stores KV in 16-token PagedAttention blocks → many small pieces
- Naively RDMA-ing each block:
  - each transfer has fixed protocol overhead
  - thousands of tiny ops → poor bandwidth utilization
- Fix: collate small blocks into one big buffer → one (or few) RDMA ops
- Bandwidth-bounded instead of latency-bounded

Push vs pull tradeoff

- Push: sender (prefill) decides when, frees prefill earlier, simpler producer
- Pull: receiver (decode) decides when, can rate-limit incoming, smoother HBM use
- Both achieve zero-copy; choice is design preference

Why this works in the pipeline

- Prefill compute: ~hundreds of ms (depends on prompt length)
- KV transfer over RDMA: ~few ms
- Decode start: nearly immediate after prefill finishes
- Total parallelism preserved → goodput stays high

Mnemonic: zero-copy = NIC/NVLink writes straight into decode HBM; package small KV pages into one big buffer; overlap the transfer with other requests' decoding so only the new request pays the (tiny) RDMA cost.

KV page collation, LMCache + NIXL configuration, UCX tuning

Page size matters for RDMA throughput

- Engines support 8 / 16 / 32 / 64 / 128 token blocks
- Bigger pages → bigger collated buffers → fewer Work Queue Elements (WQEs)
- Sustained RDMA bandwidth needs LARGE transfers
- Tip: collate ≥ 128-token pages per RDMA write
- Use dedicated CUDA stream (nonblocking) + event fences
- Always confirm overlap with Nsight Systems

LMCache measured win

- 7500-token KV transferred as 470 small ops → ~20 ms
- Same KV collated into 128-token pages → ~8 ms
- ~2.5× faster handoff, same hardware

Visual: small vs large transfer

  Many small ops:
  |req|req|req|req|req|req|...|req|     ← 470 RDMA submits
  per-op overhead dominates → 20 ms

  Few large ops:
  |══════════ one big op ══════════|     ← collated 128-token slabs
  bandwidth-bound → 8 ms

LMCache + NIXL config (sender side, prefill)

  ┌──────────────────────────────────────────────┐
  │ enable_pd: true                              │
  │ transfer_channel: nixl                       │
  │ pd_role: sender                              │
  │ pd_proxy_host: decode-host                   │
  │ pd_proxy_port: 7500                          │
  │ pd_buffer_size: 1 GiB                        │
  │ pd_buffer_device: cuda    ← stays in HBM     │
  └──────────────────────────────────────────────┘

LMCache + NIXL config (receiver side, decode)

  ┌──────────────────────────────────────────────┐
  │ enable_pd: true                              │
  │ transfer_channel: nixl                       │
  │ pd_role: receiver                            │
  │ pd_peer_host: 0.0.0.0                        │
  │ pd_peer_init_port: 7300  ← handshake/control │
  │ pd_peer_alloc_port: 7400  ← data             │
  │ pd_buffer_size: 1 GiB                        │
  │ pd_buffer_device: cuda                       │
  │ nixl_backends: [UCX]                         │
  └──────────────────────────────────────────────┘

Sizing the transfer buffer

- Start at 1 GiB
- ≈ FP16 KV for 4-8K tokens on 70B / 80 layers / 32 heads / d_head=128
- For prompts > ~7.5K tokens → 2 GiB
- Formula: bytes ≈ 2 × L × N × (H × Dh) × bytes_per_val
- FP8 / FP4 KV → shrink buffer proportionally
- Round to 256 MB boundary
- Always validate against your largest collated page group

Decode launch with UCX tuning

  UCX_RNDV_THRESH=16384         ← large bufs use rendezvous, small use eager
  UCX_MAX_EAGER_RAILS=1
  UCX_TLS=cuda_ipc,rc,rdmacm,cuda_copy,cuda_ipc,tcp
  CUDA_VISIBLE_DEVICES=1
  LMCACHE_CONFIG_FILE=lmcache-decoder-config.yaml
  python run_vllm_decoder.py --port 8200

Transport selection rule

- Single-node multi-GPU → enable CUDA IPC (NVLink/NVSwitch p2p)
- Across nodes → prefer RDMA (RoCE/IB)
- Typical UCX_TLS: rc, rdmacm, cuda_copy, cuda_ipc, tcp
- For RoCE/IB: ensure lossless ECN/PFC on the fabric
- Validate transports with `ucx_info -f`

Eager vs rendezvous in UCX

- Eager   = small messages, sent immediately (low setup overhead)
- Rendezvous = large messages, handshake first, then RDMA bulk move
- UCX_RNDV_THRESH sets the cutoff (here 16384 bytes)
- Large KV buffers fall into rendezvous → efficient bandwidth use

Deterministic hashing for KV routing

- In multiprocess runs, Python's `hash()` is randomized per process
- KV-chunk routing needs same key → same shard on every process
- Fix: `export PYTHONHASHSEED=0`
- Otherwise different workers see different "owners" for the same block → cache misses

Why handoff speed matters (recap)

- If KV transfer is slow, parallel PD pipeline collapses back to serial
- Goal: keep handoff in single-digit to tens of ms
- Combination of: RDMA + page collation + GPU-resident buffers + UCX tuning

Mnemonic: collate pages to ≥128 tokens, keep buffers in HBM, prefer rendezvous over eager for big transfers, and PYTHONHASHSEED=0 so every process agrees on where each KV chunk lives.

Connector design, fault handling, and heterogeneous hardware

Two coordination patterns

  ┌──────────────────┬─────────────────────────────────────────────────┐
  │ Pattern          │ How it works                                    │
  ├──────────────────┼─────────────────────────────────────────────────┤
  │ Global queue     │ decode pushes prompt tasks into a central queue │
  │ (NVIDIA Dynamo)  │ prefill workers pull from it                    │
  │                  │ each task carries reply-to id of the decode     │
  │                  │ prefill returns KV via NIXL RDMA                │
  ├──────────────────┼─────────────────────────────────────────────────┤
  │ Per-request      │ decode + prefill open a direct channel for      │
  │ direct channel   │ each request (TCP or RDMA negotiated at start)  │
  │ (vLLM+LMCache)   │ no shared queue                                 │
  └──────────────────┴─────────────────────────────────────────────────┘

Trade-offs

- Global queue:
  - simpler load balancing
  - easier failover
  - small queueing delay
  - good fit: multitenant, robust ops
- Direct channel:
  - fewer hops, less jitter
  - shines under stable PD pairs + fast fabric
  - good fit: strict tail-latency SLOs
- Recommendation: benchmark both under your actual prompt mix, concurrency, and failure scenarios

Pipeline must be nonblocking

- At any moment:
  - request A is decoding
  - request B's prompt is prefilling
  - request C's KV is in flight (RDMA)
- No stage idle while work exists elsewhere
- This is the whole point of disaggregation — parallel stages

Visual

  Stage    | t=0     t=1     t=2     t=3
  ─────────┼──────────────────────────────────
  prefill  | promptB promptC promptD ...
  KV xfer  |  →B?    →C?     →D?    ...
  decode   | tokenA  tokenA  tokenA tokenA  (B/C/D join as their KV lands)

About the first token

- Prefill DOES produce logits for the first token
- Often NOT transferred — decode worker can recompute from the KV cheaply
- Some systems do transfer it to save a few hundred μs
- Trade-off: simplicity vs micro-optimization

Robustness to failures

- Decode crash mid-gen → global KV pool lets another node resume
- Prefill crash mid-prompt → retry the prompt elsewhere
- Router uses heartbeats + timeouts on PD transfers
  - stalled transfer → reassign or abort cleanly
- One node failure should NOT fail the whole request

Heterogeneous hardware per phase

- Disaggregation lets each phase pick optimal HW + parallelism
- Monolithic deployments must use one type for both → compromise

Phase needs

  ┌──────────┬─────────────────────────────────────────────────────┐
  │ Phase    │ Wants                                               │
  ├──────────┼─────────────────────────────────────────────────────┤
  │ Prefill  │ high TFLOPS, fresh Tensor Cores, fast clocks        │
  │          │ moderate HBM (just for prompt KV)                   │
  ├──────────┼─────────────────────────────────────────────────────┤
  │ Decode   │ huge HBM capacity + bandwidth                       │
  │          │ doesn't need top-end compute                        │
  └──────────┴─────────────────────────────────────────────────────┘

Pairing example (Splitwise study)

- prefill: 4× H100   (compute-heavy)
- decode:  4× A100   (memory-heavy, cheaper)
- result vs homogeneous 8-GPU baseline:
  - 1.4× throughput at 20% lower cost (one config)
  - 2.35× throughput at same cost/power (other config)
- Alternative: match baseline throughput with fewer GPUs (5-6 vs 8)
- KV transfer across mixed GPUs goes over NVSwitch — minimal overhead

Visual: cost/throughput shift

  Homogeneous:
  [H100][H100][H100][H100][H100][H100][H100][H100]   8× expensive

  Heterogeneous (mixed-gen):
  [H100][H100][H100][H100] prefill (compute-bound)
  [A100][A100][A100][A100] decode  (memory-bound)
  → 2.35× RPS at same $ / W

Rule of thumb

- Compute-bound work → highest compute/$ GPU (Blackwell / Rubin)
- Memory-bound work → cost-efficient older GPUs with enough HBM BW (Hopper / Ampere)
- High-bandwidth interconnect (NVLink/NVSwitch) makes mixed setups viable

HexGen-2

- Distributed inference framework
- Treats heterogeneous-GPU allocation as an optimization problem
- Co-optimizes:
  - resource allocation (which GPU does what)
  - per-phase parallelism strategy (TP/PP/DP)
  - communication efficiency (placement vs interconnect)

Mnemonic: pick the connector pattern (queue vs direct channel) to match your SLOs, keep the pipeline busy in all three stages at once, and use cheap memory-rich GPUs for decode while saving the expensive compute-rich GPUs for prefill.

HexGen-2 results and phase-specific model parallelism

HexGen-2 results

- Llama 2 70B on mixed GPUs:
  - up to 2× serving throughput (~1.3× avg) vs SOTA at same price
  - matches high-end baseline with ~30% lower cost
- Effectively automates what Splitwise did by hand
- Disaggregation isn't just speed — it's $/query and W/query

Cost intuition

- Same traffic with 6 mixed GPUs instead of 8 top-tier GPUs → ~25% hardware cost cut
- Power efficiency: decode on lower-power GPUs (slight speed hit, big watts saved)
- Big deal when supply of newest GPUs is limited

Trade-offs of heterogeneity

- More system complexity (manage multiple GPU types)
- Less flexibility to reshuffle GPUs across phases dynamically
- Usually worth it for cost-sensitive deployments

Phase-specific model parallelism (why)

- Optimal parallelism for prefill ≠ optimal for decode
- Disaggregation lets you choose them independently

Prefill characteristics

- One big forward pass over N prompt tokens
- Compute-bound, amortizes comm overhead over many tokens
- TP works (big matmuls split across GPUs)
- PP also works (stream prompt through layer stages)
- Goal: minimize TTFT

Decode characteristics

- Tiny per-token forward passes
- Latency-sensitive (TPOT / ITL)
- More GPUs per request can HURT due to comm overhead
- TP=1 (single GPU) often best per request
- DP across requests gives throughput

Common phase-specific config

  ┌──────────┬────────────────────────────────────────────────┐
  │ Phase    │ Parallelism                                    │
  ├──────────┼────────────────────────────────────────────────┤
  │ Prefill  │ PP=4 (or TP=8) — minimize TTFT                 │
  │ Decode   │ TP=1 — minimize per-token latency              │
  └──────────┴────────────────────────────────────────────────┘

- Note: TP on prefill adds all-reduce overhead during prompt processing
- PP avoids that for prefill; TP wins inside small decode steps

KV layout mismatch problem

- Prefill TP=1 (uses PP) on 4 GPUs → each prefill GPU holds full-size KV for its layers
- Decode TP=4 → each decode GPU expects 1/4 of KV sliced along hidden dim
- Layouts don't match between phases

Visual: layout mismatch

  Prefill side (PP=4, TP=1):
    GPU0: KV for layers 0-9    (full hidden dim)
    GPU1: KV for layers 10-19  (full hidden dim)
    GPU2: KV for layers 20-29  (full hidden dim)
    GPU3: KV for layers 30-39  (full hidden dim)

  Decode side (TP=4, one stage):
    GPU0: KV for all layers, hidden dim slice [0..H/4]
    GPU1: KV for all layers, hidden dim slice [H/4..H/2]
    GPU2: KV for all layers, hidden dim slice [H/2..3H/4]
    GPU3: KV for all layers, hidden dim slice [3H/4..H]

  → need to reshape KV: split by layer ↔ split by head

NVIDIA Dynamo's KV transpose

- High-perf on-the-fly KV transpose kernel
- Runs AFTER NIXL read, BEFORE writing into decode HBM
- Converts [TP_p parts] → [TP_d parts]
- Cost is small compared to network transfer
- NVLink BW absorbs the reorg easily
- Trade: tiny transpose cost vs big phase-specific perf wins

Available parallelism dimensions to mix

- TP (tensor): split matmul across GPUs
- PP (pipeline): split layers across GPUs
- DP (data): replicate model, split requests
- SP (sequence): split sequence tokens across GPUs

Mnemonic: prefill = PP or large TP for TTFT; decode = TP=1 (or small TP) for low TPOT; KV transpose bridges the layout mismatch between phases.

Parallelism tables, mixed precision, and hybrid CPU-GPU prefill

Example prefill parallelism (per-phase config)

  ┌────────┬──────┬──────────────────────────────────────────────┐
  │ Sym    │ Val  │ Why                                          │
  ├────────┼──────┼──────────────────────────────────────────────┤
  │ TP_p   │ 2    │ split weights across 2 GPUs → halve TTFT     │
  │ PP_p   │ 2    │ 2 pipeline stages for deep models            │
  │ SP_p   │ 1    │ no sequence sharding unless huge context     │
  │ CP     │ 1    │ keep whole context on one GPU                │
  │ DP_p   │ 1 (or 2) │ 1 replica/GPU; 2 doubles batched prompts │
  └────────┴──────┴──────────────────────────────────────────────┘

Example decode parallelism

  ┌────────┬───────────────┬─────────────────────────────────────────┐
  │ Sym    │ Val           │ Why                                     │
  ├────────┼───────────────┼─────────────────────────────────────────┤
  │ TP_d   │ 1 (default)   │ minimal sync overhead                   │
  │        │ N (max)       │ helps tiny GEMMs / model too big for 1  │
  │ PP_d   │ 1             │ avoid pipeline bubbles per token        │
  │ SP_d   │ 1             │ keep stream local unless huge output    │
  │ DP_d   │ 1             │ replicas handle parallel REQUESTS, not  │
  │        │               │ a single stream                         │
  └────────┴───────────────┴─────────────────────────────────────────┘

- If model can't fit on one B200 → use TP_d = 2 or 4, NOT PP (no bubbles)

Decode parallelism intuition

- 1 decode stream wants 1 GPU (lowest comm overhead)
- TP_d > 1 only when:
  - model too big for 1 GPU
  - GEMMs are tiny enough that comm hides under compute
- DP across requests = throughput; not for a single stream

Mixed precision per phase

- Prefill: FP8 / INT8 / FP4 → faster compute, smaller KV
- Decode: same precision OR higher for output quality
- Catch: KV layout/precision must match decode expectations
- Fix: convert KV during transfer (same idea as TP transpose)
  - quantize before send, dequant on receive
  - or send low-precision over network → less BW used
- Lets you tune precision per phase independently

Hybrid prefill with CPU-GPU collaboration

Why CPUs come into play

- Capacity, not speed:
  - CPU DDR/LPDDR5X = hundreds of GB to TBs
  - GPU HBM = 80-288 GB
- CPUs do NOT replace GPU compute — way slower for matmuls
- They serve as a tier for cold KV / huge prompts / preprocessing

Memory tier visual

  ┌──────────────────────────────────────────────────────────────────┐
  │ tier      │ size       │ bandwidth     │ role                    │
  ├──────────────────────────────────────────────────────────────────┤
  │ HBM (GPU) │ 80-288 GB  │ 3-8 TB/s      │ active KV, matmul       │
  │ CPU DDR/  │ 0.5-6 TB   │ 300-500 GB/s  │ cold KV, long prompts,  │
  │ LPDDR5X   │            │               │ preprocess              │
  │ NVMe      │ TBs+       │ GB/s          │ long-term cold storage  │
  └──────────────────────────────────────────────────────────────────┘

When to offload prefill to a CPU

- Ultralong prompts (10K+ tokens) that don't fit on GPU
- Offline / batch / low-priority jobs
- Background work that mustn't tie up interactive GPUs
- Burst overflow when GPUs are saturated (graceful degradation)

Grace Blackwell (and similar superchips)

- Fast CPU-GPU interconnect (NVLink-C2C)
- CPU handles huge memory + initial preprocessing
- GPU handles dense attention
- Can spill KV to CPU DDR with low-ish overhead
- Effectively extends usable context length

Layer-partition pattern (rare, complex)

  prompt
     ↓
  GPU runs first N layers (compress sequence)
     ↓
  CPU runs middle M layers (lots of memory available)
     ↓
  GPU runs final layers (dense attention, generate KV)

- Trade-off: heavy orchestration + data movement
- Only worth it for extreme contexts / severe HBM limits

Third worker type in the system

- Now 3 prefill options the router picks from:
  - GPU prefill worker (fast, normal path)
  - CPU prefill worker (slow, big-memory, offline)
  - Local prefill on the decode GPU (when offload doesn't pay)
- Policy example: prompt_length > 5000 → CPU prefill worker

Things to watch

- CPU offload raises TTFT — not for interactive paths
- Monitor frequency of CPU-offload events
  - frequent use → you actually need more GPU capacity
- "Fail fast" if CPU path would blow SLO anyway

Cost angle

- CPU hours << GPU hours
- Hybrid clusters (GPU + CPU instances) cut $ for:
  - tokenization, padding, preprocessing
  - small-model / non-LLM inference
  - large offline prompts

Mnemonic: CPUs aren't faster — they're roomier. Use HBM for hot KV and matmuls, CPU DDR/LPDDR5X for cold KV and ultralong prompts, and let a third worker type handle the slow-but-cheap path.

6/1/26

SLO-aware request management and fault tolerance

Why SLO-aware serving needs more than scaling

- Scaling + scheduling improve capacity, but they cannot save every request during overload
- If accepting a request will likely miss TTFT / TPOT targets, reject or defer it early
- Goal: maximize goodput, not raw accepted-request count

Visual

  Bad overload behavior:
    accept everything → queues grow → many requests time out → low goodput

  SLO-aware behavior:
    admit only feasible work → reject/defer excess → accepted requests meet SLO

Early rejection / admission control

- Admission control = decide at request arrival whether the system can serve it within the latency target
- Prediction can use:
  - current queue length
  - recent throughput
  - prefill utilization
  - decode utilization
  - estimated prompt length + output length
  - lightweight latency model
- If predicted latency > SLO → return fast "too busy / retry later" instead of silently missing the deadline
- This is like HTTP 503 load shedding: painful, but better than timing out everyone

Mooncake-style example

- Router estimates pressure on BOTH prefill and decode clusters
- Long prompt → mainly stresses prefill
- Long expected generation → mainly stresses decode
- If decode cluster is already saturated with long sequences, a new long-output request is rejected/deferred
- This avoids wasting GPU compute + HBM bandwidth on a request that would miss its SLO anyway

Visual: admission gate

  new request
      ↓
  estimate prefill load + decode load + expected length
      ↓
  ┌───────────────────────────────┐
  │ would TTFT / TPOT stay in SLO? │
  └───────────────┬───────────────┘
          yes ────┴──── no
          ↓             ↓
       admit       reject / defer / retry later

Goodput intuition

- Throughput = how much work system attempts
- Goodput = how much work finishes within SLO
- Under overload, accepting fewer requests can increase goodput
- Reason: queues stay short, tail latency stays bounded, GPU memory bandwidth is not wasted on doomed requests

Quality of Service (QoS)

- QoS = not all requests are treated equally
- Examples:
  - premium users > free users
  - interactive chat > offline batch
  - short latency-sensitive requests > long best-effort jobs
- Scheduler can reserve capacity by tier:
  - premium: 10%
  - standard: 30%
  - free: 60%
- Higher-priority requests get headroom even when lower-priority traffic spikes

Visual: tiered capacity

  total cluster capacity
  ┌──────────┬────────────────────────────┬──────────────────────────────┐
  │ premium  │ standard                   │ free                         │
  │ 10%      │ 30%                        │ 60%                          │
  └──────────┴────────────────────────────┴──────────────────────────────┘
     protected  protected                    best-effort / easiest to shed

QoS actions when latency approaches SLO

- scale out if capacity can arrive fast enough
- reject new low-priority requests
- defer batch/offline work
- cap max output length for long generations
- route high-priority traffic to reserved workers
- shed load with circuit breakers before p99 / p99.9 explodes

Disaggregation helps diagnosis

- Separate prefill queue + decode queue reveal the real bottleneck
- Prefill backed up → stop accepting huge prompts or add prefill workers
- Decode saturated → limit long-output / reasoning requests or add decode workers
- This is better than one generic "GPU busy" signal

Visual: bottleneck-specific response

  prefill queue high → prompt admission ↓ / prefill scale ↑
  decode queue high  → max output ↓ / decode scale ↑ / long gens rejected

Fault tolerance in disaggregated inference

- Main failure question: what happens if a worker dies mid-request?
- If KV cache is in a shared pool, another decode worker can resume generation from saved KV
- If KV is not saved, system must rerun prefill on another node, then continue decode
- KV snapshots trade extra memory/copy overhead for faster recovery

Failure recovery cases

  ┌──────────────────────┬──────────────────────────────────────────────┐
  │ Failure              │ Recovery                                      │
  ├──────────────────────┼──────────────────────────────────────────────┤
  │ decode node dies     │ reload KV from pool, continue on new decoder  │
  │ prefill node dies    │ retry prefill elsewhere                       │
  │ KV missing           │ recompute prefill, then decode                │
  │ transfer stalls      │ timeout, reassign, or abort cleanly           │
  └──────────────────────┴──────────────────────────────────────────────┘

Why periodic KV checkpointing matters

- KV cache is the expensive state created by prefill
- Saving it periodically means failures lose fewer tokens / less compute
- This is useful even without strict prefill-decode disaggregation
- vLLM-like systems can checkpoint/copy KV to a pool to protect against worker loss

Process-level checkpointing

- `cuda-checkpoint` + CRIU can snapshot a Linux GPU worker process
- Restore works best onto another node with the same GPU chip type
- Useful for preemption, failure recovery, and reducing cold-start cost
- Prewarmed checkpoint restore can avoid reloading weights + recompiling CUDA graphs from scratch

Cold-start intuition

  normal start:
    load weights → allocate GPU memory → compile/capture graphs → warm caches → serve

  checkpoint restore:
    restore prewarmed process/GPU state → serve sooner

Mnemonic: SLO serving is a nightclub bouncer plus an ambulance — admission control keeps overload out, QoS lets VIP traffic through, and KV/process checkpoints recover when workers crash.

Dynamic scheduling and load balancing

Why static prefill/decode splits fail

- Workload mix changes over time
- Long prompts + short answers → prefill-heavy
- Short prompts + long reasoning answers → decode-heavy
- A fixed `X_p : Y_d` worker ratio becomes wrong as traffic shifts
- Goal: keep both phases near capacity WITHOUT growing queues

Visual: changing bottleneck

  summarization hour:
    long input  → heavy prefill  → need more prefill workers
    short output → light decode

  reasoning hour:
    short input → light prefill
    long output → heavy decode → need more decode workers

Static misconfiguration patterns

  ┌──────────────────────────────┬──────────────────────────────────────────┐
  │ Misconfiguration             │ Symptom                                  │
  ├──────────────────────────────┼──────────────────────────────────────────┤
  │ too few decode workers       │ TPOT rises; decode queue grows           │
  │ too few prefill workers      │ TTFT rises; requests wait to start       │
  │ overprovisioned one side     │ expensive GPUs idle on the other phase   │
  │ balanced for old workload    │ breaks when prompt/output mix changes    │
  └──────────────────────────────┴──────────────────────────────────────────┘

Adaptive scheduling core idea

- Continuously observe prefill queue, decode queue, GPU utilization, and latency
- Predict where bottlenecks will form before queues explode
- Route requests to less-loaded workers
- Repurpose capacity between prefill and decode when possible
- Combine with autoscaling, admission control, and QoS shedding

Visual: feedback loop

  metrics → predict bottleneck → adjust routing / worker split → measure again
     ↑                                                        ↓
     └──────────────────── closed-loop scheduler ────────────┘

Good target operating regime

- Prefill busy but not overloaded
- Decode busy but not overloaded
- TTFT stable
- TPOT stable
- No single node becomes a hotspot
- Accepted requests meet SLO; excess load is rejected/deferred early

TetriInfer two-level scheduler

- Level 1: request-level routing
  - assigns each request to specific prefill + decode instances based on current load
  - normal per-request placement decision
- Level 2: cluster-level hotspot prevention
  - monitors queue lengths, GPU utilization trends, and predicted resource usage
  - proactively shifts work away from nodes likely to become overloaded

TetriInfer intuition

- Name hints at packing requests like Tetris pieces
- Different requests have different prompt lengths, output lengths, and resource shapes
- Scheduler tries to fit these shapes into available GPU time without interference
- This smooths load across the cluster instead of letting one node become the long-sequence graveyard

Visual: hotspot prevention

  naive routing:
    decode0: [long][long][long][long]  → hotspot, TPOT spike
    decode1: [short]                   → underused
    decode2: [short][short]            → underused

  TetriInfer-style routing:
    decode0: [long][short]
    decode1: [long][short]
    decode2: [long][short]
    → smoother queues, better p99 latency

How this connects to SLO-aware control

- Admission control decides whether to accept work
- QoS decides which work matters most
- Dynamic scheduling decides where accepted work should go
- Fault tolerance makes sure failures do not drop the request
- Together: stable service under overload, traffic shifts, and node failures

First-principles summary

- Prefill and decode are two different factories
- If one factory is backed up while the other is idle, global throughput suffers
- Dynamic scheduling is the dispatcher that keeps both factories fed at the right rate
- SLO-aware admission is the gate that prevents the dispatcher from accepting impossible work

Mnemonic: static splits are brittle; dynamic schedulers watch queues, pack requests like Tetris, and move work before hotspots become SLO misses.

Arrow, Mooncake, and dynamic resource scaling

Why fixed PD ratios fail

- Workload mix shifts during the day
  - input-heavy (summaries, RAG) → prefill bottleneck
  - output-heavy (reasoning, long answers) → decode bottleneck
- Static # of prefill vs decode workers → temporary goodput loss
- One side idles while the other piles up queue

Arrow — adaptive instance scaling

- Continuously measures:
  - input token rate vs output token rate
  - backlog per worker pool
  - TTFT / TPOT percentiles
- Then BOTH:
  - reschedules requests (where to send next request)
  - rescales instances (how many prefill vs decode workers)
- Treats # of prefill / # of decode as a tunable parameter
- Result: up to 5.6× higher RPS vs nonadaptive in highly shifting workloads

Two scaling situations

  ┌────────────────────────┬─────────────────────────────────────┐
  │ Detected condition     │ Arrow action                        │
  ├────────────────────────┼─────────────────────────────────────┤
  │ prefill queue growing  │ convert decode → prefill            │
  │ TTFT rising            │ or launch new prefill instances     │
  ├────────────────────────┼─────────────────────────────────────┤
  │ decode queue growing   │ shift prefill → decode              │
  │ TPOT rising            │ or launch new decode instances      │
  └────────────────────────┴─────────────────────────────────────┘

Visual

  Workload shifts → input-heavy
  ┌─────────────────────────┐
  │ prefill: ▇▇▇▇▇▇▇▇ busy  │  ← bottleneck
  │ decode:  ▇▇▇    idle    │
  └─────────────────────────┘
              ↓ Arrow
  ┌─────────────────────────┐
  │ prefill: ▇▇▇▇▇▇▇▇       │  more workers here
  │ decode:  ▇▇             │  fewer workers
  └─────────────────────────┘

Role-flip mechanics

- On-prem (fixed GPUs): instruct a GPU to switch roles
  - overhead: may need to load different sharded weights / quant
  - some designs keep all weights loaded → just route different tasks
- Cloud: trigger Kubernetes HPA / Cluster Autoscaler to add/remove pods
- New pods take tens of seconds → may need load-shedding in the gap

Mooncake — supply-side scaling's demand-side twin

- While Arrow scales SUPPLY (more/less workers), Mooncake manages DEMAND
- Predictive early rejection (admission control):
  - if predicted that SLO can't be met → reject before accepting
  - prevents overload cascades
- Two sides of one coin:
  - supply-side: spin up / reallocate workers (Arrow)
  - demand-side: throttle / reject low-prio (Mooncake)

Dynamic resource scaling techniques

- Elastic instances (k8s HPA-style)
  - rule: keep prefill GPU util ≈ 70%
  - exceed → add a pod; under → move pod to decode
- Instance "flip" (TetriInfer)
  - nodes can switch roles
  - needs weights / sharding / quant compatible with both roles
- Statelessness for elasticity (Arrow)
  - workers don't keep long-lived session state
  - free to reassign
  - mid-decode is sticky though — wait for it to finish before flipping
- Anti-oscillation guards
  - min residency time per role
  - hysteresis on thresholds
  - prevents thrashing/flapping between roles

Predictive scheduling

- ARIMA-style forecasting on traffic patterns
- e.g. "long-output spike every night at 9pm" → preallocate decode capacity
- Reduces lag between workload shift and resource shift
- Combine reactive + predictive for best stability

Metrics to feed the controller

- Prefill queue length, decode queue length
- TTFT p50/p95/p99
- TPOT (ITL) p50/p95/p99
- Per-pool GPU util + HBM util
- Usually exported via Prometheus + custom k8s controllers

Multitenant / mixed-workload reuse

- Disaggregation modularity lets idle decode GPUs run
  a different smaller model temporarily
- Not mainstream yet, but architecturally feasible

Layered insight

  ┌─────────────────────────────────────────────────────────┐
  │ disaggregation        removes PHASE INTERFERENCE        │
  │ adaptive disaggregation removes PHASE IMBALANCE         │
  │ admission control     removes SLO VIOLATION CASCADES    │
  └─────────────────────────────────────────────────────────┘

Mnemonic: Arrow scales the supply (flip GPUs and add pods), Mooncake throttles the demand (reject early), forecasting preallocates for known peaks — and anti-oscillation guards stop everything from thrashing.

