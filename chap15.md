Multinode Inference, Parallelism, Decoding, and Routing Optimization: 

3/10/26

Disaggregated prefill and decode

- Traditional: prefill + decode colocated on same GPUs → prefill-decode interference
  - Long prefill blocks time-sensitive decode work (and vice versa)
  - One scheduling strategy forced on two very different workloads

- Disaggregated: separate GPU pools for each phase
  - Prefill pool: processes input prompt → produces KV cache
  - Decode pool: autoregressive token generation using KV cache
  - Communication: only KV cache transfer between pools (NVLink/NVSwitch intra-node, GPUDirect RDMA inter-node)

- Why it works — the two phases have opposite profiles
  - Prefill: large parallel matmuls, compute-bound, optimize for TTFT
  - Decode: many small sequential steps, memory-bound, optimize for TPOT

- Benefits
  - Scale each pool independently (add prefill GPUs or decode GPUs as needed)
  - No cross-phase interference → less GPU idle time → higher goodput
  - Prefill: batch aggressively for throughput without hurting decode latency
  - Decode: run at lower batch sizes with priority scheduling for stable TPOT
  - Both TTFT and TPOT SLOs met simultaneously (no compromise)
  - DistServe reported up to 7.4x more goodput (12.6x under tighter latency SLOs)

  Colocated vs disaggregated

  Colocated (traditional):
    GPU pool ──► prefill req A ──► decode req B ──► prefill req C ──► decode req A
                 ▲ decode B waits behind prefill A = interference

  Disaggregated:
    Prefill pool ──► prefill A ──► prefill C ──► prefill E    (batch aggressively)
                          │              │
                      KV transfer    KV transfer
                          ▼              ▼
    Decode pool  ──► decode A ──► decode B ──► decode C       (low latency, priority scheduled)

  No interference — each pool runs its own workload at full utilization 

Disaggregated prefill → decode via NIXL + LMCache

  Prefill worker (GPU pool A)
      │
      │ computes KV cache for input prompt
      │
      ▼
  NIXL transport ──── moves KV cache bytes ────► LMCache (on decode side)
  (NVLink / RDMA)     GPU→GPU or GPU→CPU→GPU     (stores/indexes KV cache)
                                                       │
                                                       ▼
                                                  vLLM decode worker (GPU pool B)
                                                       │
                                                       reads KV cache from LMCache
                                                       │
                                                       ▼
                                                  autoregressive token generation

KV cache transfer and NIXL

- NIXL auto-selects the fastest transport path based on topology:
  - Intra-node: NVLink/NVSwitch (device-to-device, no host staging)
  - Inter-node: GPUDirect RDMA over InfiniBand or RoCEv2
  - Fallback: host-staged paths (non-optimal, when peer access unavailable)
- Integrated with vLLM (via LMCache), NVIDIA Dynamo, and TensorRT-LLM
- 1 NIC per GPU recommended for optimal disaggregated prefill/decode and MoE all-to-all
- ConnectX-8 SuperNICs: up to 800 Gb/s per port with GPUDirect RDMA
- Overlap KV transfers with compute using separate streams and events
- Always validate end-to-end KV transfer time on your actual deployment

- Worker placement follows the fabric:
  - Same node → NVLink/NVSwitch via CUDA peer access
  - Cross node → GPUDirect RDMA over InfiniBand/RoCEv2

Kubernetes deployment (llm-d project)

  ┌──────────────────────────────────────────────────────────┐
  │                    Kubernetes cluster                     │
  │                                                          │
  │  Variant autoscaler ← monitors prompt-response mix       │
  │       │                                                  │
  │       ├── scale prefill replicas (long prompts? add more)│
  │       └── scale decode replicas (long outputs? add more) │
  │                                                          │
  │  Prefill pool ──► LMCache + NIXL ──► Decode pool         │
  │  (vLLM instances)    KV transfer     (vLLM instances)    │
  └──────────────────────────────────────────────────────────┘

  - All nodes run same runtime (vLLM/SGLang/Dynamo) — role assigned by orchestrator
  - Roles can be assigned statically at startup or dynamically during lifecycle
  - Long prompts, short outputs → more prefill GPUs
  - Short prompts, long outputs → more decode GPUs

Disaggregated PD — trade-offs summary

  Benefits                                  Costs
  ─────────────────────────────────────     ─────────────────────────────────────
  No prefill-decode interference            KV cache transfer overhead (few GB)
  Scale each phase independently            More complex scheduling
  Meet both TTFT and TPOT SLOs             Need high-bandwidth interconnect
  Up to 7.4x goodput (DistServe)           Fabric-aware worker placement required

Tensor parallelism (TP) for inference

- Splits each layer's computation (matmuls) across multiple GPUs
- GPUs compute partial results in parallel → all-reduce to aggregate
- Use when: single layer too large for 1 GPU, or want lower latency via parallel compute
- Requires high-bandwidth interconnect — all GPUs in lockstep per layer

- Applied to: attention projections and MLP matmuls in transformers
- Activations per token are small → all-reduce cost is low on NVLink
- Near-linear speedup as long as comms are fast relative to compute

- Scaling limits follow the fabric:

 Scope                       Interconnect           TP efficiency
 ──────────────────────────  ─────────────────────  ─────────────
 Within NVL72 rack (≤72)     NVLink Switch 130TB/s  excellent
 Across racks (≤576)         NVLink Switch Systems  good (viable, not full speed)
 Across nodes (no NVLink)    InfiniBand/Ethernet    poor (avoid for TP)

- Best practice: keep TP within a single NVLink/NVSwitch domain
- NVLink Switch Systems make cross-rack TP viable (up to 576 GPUs) but intra-rack is always faster

Pipeline parallelism (PP) for inference

- Splits model depth-wise: GPU 0 = layers 1-20, GPU 1 = layers 21-40, GPU 2 = layers 41-60
- Data flows sequentially through stages — GPU 0 → GPU 1 → GPU 2
- Use when: model too deep to fit on 1 GPU (memory scaling)
- PP increases throughput via microbatching, but adds latency per single request
- Prefill benefits more (long sequence, can stream portions in staggered fashion)
- Decode benefits less (1 token at a time → pipeline bubbles)
- Usually combined with TP: PP splits depth, TP splits width

if data flows sequentially, how is anything parallel?

Without microbatching (purely sequential — no parallelism)

  Time →    T0        T1        T2        T3        T4        T5
  GPU 0:   req A      idle      idle      req B      idle      idle
           L1-20                           L1-20
  GPU 1:   idle      req A      idle      idle      req B      idle
                     L21-40                          L21-40
  GPU 2:   idle      idle      req A      idle      idle      req B
                               L41-60                          L41-60

  Only 1 GPU busy at a time — 2 GPUs always idle = pipeline bubbles


With microbatching (assembly line — THIS is the parallelism)

  Time →    T0        T1        T2        T3        T4
  GPU 0:   req A      req B      req C      req D      idle
           L1-20     L1-20     L1-20     L1-20
  GPU 1:   idle      req A      req B      req C      req D
                     L21-40    L21-40    L21-40    L21-40
  GPU 2:   idle      idle      req A      req B      req C
                               L41-60    L41-60    L41-60

  At T2: ALL 3 GPUs busy simultaneously
    GPU 0 runs layers 1-20 on req C
    GPU 1 runs layers 21-40 on req B (received from GPU 0 at T1)
    GPU 2 runs layers 41-60 on req A (received from GPU 1 at T1)