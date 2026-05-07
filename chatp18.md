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
