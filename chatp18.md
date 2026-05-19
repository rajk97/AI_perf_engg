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
