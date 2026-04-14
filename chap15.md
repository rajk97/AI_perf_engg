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

3/17/26:

Expert parallelism (EP) for inference

- MoE-specific: distribute different experts across different GPUs
- Example: 16 experts, 4 GPUs → 4 experts per GPU
- Each token activates only top-k experts (typically top-2) → token routed to those GPUs

- Communication pattern: all-to-all (twice per MoE layer)
  1. Tokens → shuffled to GPUs owning their assigned experts
  2. Experts compute
  3. Results → shuffled back to original order

  Token arrives ──► gate picks top-2 experts ──► all-to-all dispatch to expert GPUs
                                                        │
                                                  expert compute
                                                        │
                                                  all-to-all gather back ──► combine results

- Scales model capacity linearly with GPUs (100 experts on 100 GPUs, but each token only touches 2)
- Individual token = few experts; aggregate across all users = ALL experts active concurrently

- Load balancing is critical
  - Hot expert (receives disproportionate tokens) → straggler effect → pipeline stalls
  - Fixes:
    - Capacity factor (1.2-1.5x avg load) — cap tokens per expert, overflow to second-choice expert
    - Load-balancing loss + gating noise during training (GLaM-style)
    - Hot-expert replication — copy overloaded expert to extra GPUs (costs memory)
    - Group frequently paired experts on same GPU/node → localize traffic

- Communication optimization
  - Top-1 gating: less comms (1 expert per token) but lower quality, uneven load
  - Top-2 gating: more comms but better quality, balanced load — production default
  - High-bandwidth interconnects essential: NVLink intra-rack, InfiniBand inter-node

EP load balancing

  Balanced (good):                     Imbalanced (bad — straggler):
  GPU 0: ████ (4 tokens)              GPU 0: ████████████ (12 tokens) ← hot expert
  GPU 1: █████ (5 tokens)             GPU 1: ██ (2 tokens) idle...
  GPU 2: ████ (4 tokens)              GPU 2: █ (1 token) idle...
  GPU 3: █████ (5 tokens)             GPU 3: ███ (3 tokens) idle...
  All finish ~same time                All wait for GPU 0 to finish

  Fix: capacity factor caps GPU 0 → overflow tokens go to second-choice expert


Data parallelism — training vs inference

 Training DP:
   GPU 0: full model copy ──► forward(batch 0) ──► backward ──► gradients ─┐
   GPU 1: full model copy ──► forward(batch 1) ──► backward ──► gradients ─┤── all-reduce
   GPU 2: full model copy ──► forward(batch 2) ──► backward ──► gradients ─┘   (sync)
                                                                                 │
                                                                         averaged gradients
                                                                                 │
                                                                         all GPUs update weights
                                                                         (must stay in sync)

 Inference DP:
   GPU 0: full model copy ──► forward(req A) ──► output A    ← independent
   GPU 1: full model copy ──► forward(req B) ──► output B    ← independent
   GPU 2: full model copy ──► forward(req C) ──► output C    ← independent

                             NO communication between GPUs
                             no gradients, no sync, no all-reduce

Data parallelism (DP) for inference

- Replicate entire model on multiple GPUs — each GPU handles different requests independently
- No inter-GPU communication during inference (unlike training — no gradient sync)
- Linear throughput scaling: 8 GPUs = ~8x requests/second
- Does NOT reduce latency for a single request — only increases throughput
- Cost: 8 replicas = 8x GPU memory = 8x hardware cost
- Usually combined with TP/PP/EP when model doesn't fit on 1 GPU
  - Example: 2 DP groups × 8 GPUs with TP = 16 GPUs total, 2x throughput
- Production: replicas sit behind a load balancer, each is a separate model instance

Context parallelism (CP) for inference

- Splits a single long sequence across multiple GPUs (sequence dimension, not layers or weights)
- Each GPU handles a chunk of the input positions at each layer
- KV cache split across GPUs — enables contexts too large for 1 GPU's memory
- Near-linear speedup for prefill on very long inputs (2x GPUs ≈ half the prefill time)
- Does NOT speed up decode (still sequential token-by-token)
- Challenge: self-attention is global (every token attends to every earlier token)
  - Uses ring parallelism + blocked attention to limit cross-GPU communication
- Short prompts: CP overhead not worth it
- Long prompts (10K-1M+ tokens): CP significantly reduces TTFT

All 5 parallelism strategies — what they split

  Strategy    Splits along        Use when                          Communication
  ─────────   ─────────────────   ──────────────────────────────    ──────────────────
  TP          within each layer   layer too wide for 1 GPU          all-reduce per layer
  PP          across layers       model too deep for 1 GPU          activation transfer between stages
  EP          across experts      MoE experts across GPUs           all-to-all per MoE layer
  DP          across requests     need more throughput               none (independent replicas)
  CP          across sequence     context too long for 1 GPU        ring attention at boundaries

  5D parallelism = TP + PP + EP + DP + CP all combined

Hybrid parallelism — combining all strategies

- No single parallelism method is enough for massive MoE models — combine them

- Guiding principle (priority order):
 1. TP first — within NVLink domain, up to diminishing returns
 2. PP minimally — just enough to fit model in memory
 3. EP maximized — distribute MoE experts across GPUs
 4. DP replicas — add copies to scale throughput for more users
 5. CP optionally — layer on top for extremely long inputs

- Example: 64 GPUs, 60-layer MoE with 64 experts
 - 4-way PP (15 layers per stage) → splits depth
 - 2-way TP within each stage → splits width
 - 16 experts per 4-GPU group via EP → splits experts
 - 2x DP replicas (2 × 64 GPUs = 128 total) → doubles throughput

- Align parallelism with hardware topology:
 - NVL72 (72 GPUs, full NVLink): TP + EP groups within the domain
 - 2 × 8-GPU nodes (InfiniBand cross-node): keep TP within each 8-GPU node, avoid inter-node TP
 - Even inside NVL72, keep TP groups smaller than 72 — all-reduce latency has diminishing returns

 Speculative decoding — breaking the sequential bottleneck

- Problem: decode generates 1 token at a time, each depends on previous → serial bottleneck

- Speculative decoding (two-model approach):
  - Small fast draft model → proposes several tokens in one batch
  - Large target model → verifies all candidates in parallel
  - Accepted tokens skip sequential steps → higher throughput
  - Cost: two models to deploy, verification pass can bottleneck

- Medusa (single-model approach):
  - Attaches multiple lightweight decoding heads to one LLM
  - Tree-based attention: generate + verify several candidates in ONE forward pass
  - No cross-model token transfers, no second model to deploy
  - Improvement over two-model speculative decoding

Speculative decoding — two-model vs Medusa

  Two-model:
    Draft model ──► propose 5 tokens ──► Target model ──► verify all 5
    (small, fast)                        (large, slow)     accept 3, reject 2
                                                           ← 2 models, 2 passes

  Medusa (single-model):
    LLM + multiple heads ──► propose + verify 5 tokens in 1 forward pass
                              accept 3, reject 2
                              ← 1 model, 1 pass

Two-model speculative decoding

- Draft model (small, fast): generates k tokens speculatively ahead
- Target model (large, slow): verifies all k tokens in one batched forward pass
- If all k match → k tokens generated in ~1 target model call (k× speedup)
- If divergence at token j → keep tokens 1..j, discard j+1..k, resume from j
- At least 1 token always accepted per cycle
- Practical speedup: 1.5-2.5x (theoretical: kx)
- Draft model requirements: same tokenizer, high fidelity to target distribution, 4x+ faster
- Built into vLLM, SGLang, TensorRT-LLM

How verification works — step by step

  Context so far: "The cat sat on the"

  Step 1: Draft model generates k=4 tokens (fast, sequential):
    "The cat sat on the" → "warm"  → "sunny" → "roof" → "top"
    Draft sequence: [warm, sunny, roof, top]

  Step 2: Target model verifies ALL 4 in ONE forward pass:
    Feed: "The cat sat on the warm sunny roof top"
    Target model produces next-token probabilities at EVERY position:

    Position:  "the ___"   "warm ___"   "sunny ___"   "roof ___"
    Draft said: "warm"      "sunny"      "roof"         "top"
    Target says: "warm" ✓   "sunny" ✓    "roof" ✓       "top" ✓
                                                         All accepted!
    Result: 4 tokens in 1 target call


  What if target disagrees?

    Position:  "the ___"   "warm ___"   "sunny ___"   "roof ___"
    Draft said: "warm"      "sunny"      "roof"         "top"
    Target says: "warm" ✓   "cozy"  ✖    ─── stop here, discard rest
                accept       reject

    Result: keep "warm", use target's "cozy", discard "roof" and "top"
    Resume drafting from: "The cat sat on the warm cozy"

Key: second token verification for sunny just takes first token to be warm into the sentence -- there's a final sequential verification afterwards.     

Two separate phases

  Phase 1: Forward pass (parallel, on GPU)
  ──────────────────────────────────────────
  Feed ALL draft tokens into the target model at once.
  The model computes as if every draft token is correct.
  This is just matrix math — the GPU doesn't "know" about verification.

  Phase 2: Acceptance check (sequential, on CPU)
  ──────────────────────────────────────────
  AFTER the forward pass finishes, compare outputs left-to-right.
  Stop at first mismatch.


  Phase 1 (GPU, parallel):
    Input:  [The cat sat on the | warm | sunny | roof | top]
                                  all draft tokens fed in as if correct
    Output: predictions at every position (all computed simultaneously)

  Phase 2 (CPU, sequential):
    pos 0: target="warm"  vs draft="warm"   ✓ accept
    pos 1: target="sunny" vs draft="sunny"  ✓ accept
    pos 2: target="cozy"  vs draft="roof"   ✖ STOP
    pos 3: never checked

  The "waste": positions 2 and 3 were computed in the forward pass
               but their results are thrown away.
               That's the GAMBLE of speculative decoding.

- EAGLE: operates at feature level instead of token level
  - Extrapolates target model's own intermediate representations
  - Higher acceptance rates → up to 3.5x speedup with 4-token draft
  - Preserves target model's output distribution

3/18/26:

EAGLE evolution

  EAGLE-1: draft model predicts internal feature vectors → decode to tokens (up to 3.5x)
  EAGLE-2: adds dynamic draft TREE of possibilities → 20-40% faster than EAGLE-1
  EAGLE-3: skips feature prediction, predicts tokens directly using fused multi-layer features
           → up to 1.4x over EAGLE-2, up to 6.5x over vanilla baseline

- Other tricks: skip every Nth layer, FP8/NVFP4 for draft, smaller hidden size during draft

Self-speculative decoding (single model, no draft model)

- Same target model does both draft and verify
- Draft pass: skip half the layers (or lower precision) → fast approximate output of k tokens
- Verify pass: full forward pass on all k tokens → accept/reject left-to-right
- No separate model to train/maintain/deploy
- ~2x speedup, similar to two-model speculative decoding  

Medusa:

Standard LLM: 1 head → 1 token per forward pass

  Input: "The cat sat on the"
                │
                ▼
  ┌──────────────────────┐
  │   Transformer body    │
  │   (all layers)        │
  └──────────┬───────────┘
             │
         [head 0]  → "warm"         ← 1 token per step
             │
  Next step: "The cat sat on the warm"
             → another full forward pass for "sunny"
             → another for "roof"
             → 3 forward passes for 3 tokens


Medusa LLM: multiple heads → multiple tokens per forward pass

  Input: "The cat sat on the"
                │
                ▼
  ┌──────────────────────┐
  │   Transformer body    │
  │   (all layers)        │    same body, shared computation
  └──┬───────┬───────┬───┘
     │       │       │
  [head 0] [head 1] [head 2]     ← extra heads added during training
     │       │       │
   "warm"  "sunny"  "roof"        ← 3 tokens from 1 forward pass!

How the tree-based verification works

  Medusa generates a TREE of candidates, not a single sequence:

                      "warm"   (head 0: next token)
                     /      \
               "sunny"      "cozy"    (head 1: token+2 candidates)
              /     \       /    \
          "roof" "deck" "little" "big"  (head 2: token+3 candidates)

  This gives multiple candidate PATHS through the tree:
    Path A: warm → sunny → roof
    Path B: warm → sunny → deck
    Path C: warm → cozy → little
    Path D: warm → cozy → big

  Verification: tree-based attention checks all paths in ONE forward pass
    Target agrees with: warm ✓ → sunny ✓ → roof ✓  (Path A wins!)
    Accept 3 tokens from 1 forward pass

  If Path A fails at "roof":
    warm ✓ → sunny ✓ → roof ✖ but deck ✓ → accept warm, sunny, deck
    The TREE gives fallback options that a single chain wouldn't have

Why tree > chain

  Single chain (speculative decoding):
    warm → sunny → roof → top
    If "roof" wrong → discard "roof" and "top" → only 2 tokens accepted

  Tree (Medusa):
    warm → sunny → roof
                 → deck    ← backup branch!
         → cozy → little
                → big
    If "roof" wrong → try "deck" → if correct → still 3 tokens accepted
    More branches = more chances to match = higher acceptance rate

        Speculative decoding comparison

  Method                Model(s)    Tokens/pass   Speedup    Trade-off
  ────────────────────  ──────────  ────────────  ─────────  ─────────────────
  Two-model spec        draft+target  1 (draft)   1.5-2.5x   deploy 2 models
  EAGLE-1               draft head    1 (feature)  ~3.5x     feature prediction
  EAGLE-3               draft head    1 (direct)   ~6.5x     fused multi-layer
  Self-speculative      target only   1 (skip layers) ~2x    no extra model
  Medusa                1 model+heads MULTIPLE      2-3.6x   must retrain model

Interleaving decode steps across requests

- Inference engine batches token-level decode steps from different users on same GPU
- While one request waits on I/O or dependency → GPU runs another request's next token
- Doesn't speed up single-request latency (may add tiny overhead)
- Greatly improves throughput + GPU utilization under heavy concurrent load
- Implemented via continuous batching + token scheduling in vLLM, SGLang, etc.
- Verify with Nsight Systems that token-step kernels overlap with NIC/NVLink transfers

Constrained decoding (structured outputs)

- Force output to match a format (JSON schema, grammar, allowed tokens)
- At each step, only valid tokens allowed → scan down vocabulary until valid one found
- Cost: few ms per token for grammar checking + vocabulary filtering

- Mitigation: compile grammar → precompute valid token masks per state
  - Mask invalid softmax outputs at runtime → reduces backtracking
  - Simple JSON schemas: low single-digit % overhead
  - Complex grammars / small batches: can hit double-digit % overhead

- Implementations: TensorRT-LLM (XGrammar backend), vLLM, Hugging Face Transformers
- Best practice: keep grammars compact, avoid overly restrictive constraints
- Alternative: let LLM decode freely → postprocess/filter outputs (not always viable)

MoE dynamic routing and expert communication

- Every MoE layer: all-to-all shuffle sends tokens to GPUs hosting their assigned experts
- Happens per layer → can dominate inference time if not optimized

- Hierarchical routing to reduce cross-node traffic:

  Step 1: route within node (NVLink/NVSwitch — fast)
    GPU 0 ←→ GPU 1 ←→ GPU 2 ←→ GPU 3    (intra-node, ~1.8 TB/s)

  Step 2: route across nodes only for tokens needing non-local experts
    Node 0 ←→ Node 1    (inter-node, InfiniBand — slower)

  Two-stage all-to-all: most tokens stay local, only stragglers cross the node boundary

4/14/26:

MoE (Mixture of Experts) — how it works

- Dense model: every token passes through ONE large FFN per layer
- MoE model: replace that one FFN with N smaller expert FFNs + a router
- Router = small linear layer that scores each expert per token → softmax → top-k gating
- Only top-k experts (usually 2) activate per token → rest stay idle
- Output = weighted sum of active experts' outputs (weights from router softmax)

  Dense:    token → [FFN] → output

  MoE:      token → [Router] → picks Expert 2 (weight 0.6) + Expert 5 (weight 0.4)
                         │
                         ├→ Expert 2: token × 0.6
                         └→ Expert 5: token × 0.4
                                  │
                                  ▼
                          weighted sum → output

- Benefit: model has N experts worth of parameters but only runs k per token → more capacity, same compute
- Used in: Mixtral, DeepSeek, GPT-4 (rumored), Switch Transformer

MoE with expert parallelism — tokens must travel

- Each GPU hosts DIFFERENT experts (Expert 0 on GPU 0, Expert 1 on GPU 1, etc.)
- Each GPU also has a local batch of tokens from the scheduler
- Router runs locally → tells each token which expert (= which GPU) to visit
- Problem: tokens must physically move to the GPU hosting their assigned expert

  Per MoE layer, two all-to-all communications happen:

  ALL-TO-ALL #1 (dispatch): send each token to the GPU hosting its expert
  → each GPU runs its local expert on all tokens that arrived
  ALL-TO-ALL #2 (combine): send results BACK to origin GPU

  Why send back? The next layer (attention, layernorm) needs ALL tokens together again
  MoE is a temporary detour: together → scatter → compute → gather → together

  GPU 0 owns [A,B,C,D] for the entire forward pass
       │
       │ attention, norms (all local)
       ▼
    ┌─ MoE layer ──────────────────────────────┐
    │ Router: A→Expert2, B→Expert0, C→Expert3  │
    │ ALL-TO-ALL #1: ship A to GPU2, C to GPU3 │
    │ COMPUTE: each GPU runs its expert         │
    │ ALL-TO-ALL #2: results back to origin     │
    └──────────────────────────────────────────┘
       │
       ▼ attention, norms continue (all local)

┌─────────────────────────────────────────────────────────────────┐
│            MoE COMMUNICATION OPTIMIZATION — BIG PICTURE         │
│                                                                 │
│  Problem: all-to-all is the BOTTLENECK of MoE layers            │
│  Goal:    minimize time tokens spend in transit, maximize GPU    │
│           compute utilization                                    │
└─────────────────────────────────────────────────────────────────┘

There are 5 independent levers. Each attacks a different angle:

  ┌──────────────────────────────────────────────────────────────┐
  │  LEVER 1: OVERLAP (hide latency)                             │
  │  ─────────────────────────────                               │
  │  Double-buffer: send batch N+1 while computing batch N       │
  │                                                              │
  │  Time: ──[send B1][send B2][send B3]──                       │
  │         ─────────[comp B1][comp B2]──   ← overlap            │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  LEVER 2: SCHEDULE (avoid global barrier)                    │
  │  ────────────────────────────────────                        │
  │  Butterfly / shifted schedule: phased partial exchanges      │
  │                                                              │
  │  Naive:      everyone → BARRIER → go         (1 big sync)   │
  │  Butterfly:  round1(partial) → round2(partial) → done       │
  │              every link busy every round, no idle waiting    │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  LEVER 3: HIERARCHY (use fast links first)                   │
  │  ────────────────────────────────────────                    │
  │  Stage 1: intra-rack shuffle via NVLink (fast, 130 TB/s)    │
  │  Stage 2: inter-rack only for RESIDUAL tokens (slow NIC)    │
  │                                                              │
  │  ┌─Rack A──────┐     ┌─Rack B──────┐                        │
  │  │ GPU↔GPU     │ ──→ │ GPU↔GPU     │                        │
  │  │ NVLink ⚡    │ NIC │ NVLink ⚡    │                        │
  │  │ (most done  │ 🐌  │ (most done  │                        │
  │  │  here)      │     │  here)      │                        │
  │  └─────────────┘     └─────────────┘                        │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  LEVER 4: PLACEMENT (avoid sending at all)                   │
  │  ─────────────────────────────────────                       │
  │  Expert collocation: put commonly co-activated experts       │
  │  on the SAME GPU                                             │
  │                                                              │
  │  Profile shows: Expert 5 and Expert 7 fire together 80%     │
  │                                                              │
  │  Before: token → GPU 2 (Expert 5) → GPU 4 (Expert 7)        │
  │          = 2 network hops                                    │
  │                                                              │
  │  After:  token → GPU 2 (Expert 5 + Expert 7)                │
  │          = 0 extra hops, both run locally                    │
  │                                                              │
  │  How to find pairs: gating-frequency analysis (profile which │
  │  experts the router picks together)                          │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  LEVER 5: COMPRESSION (send fewer bytes)                     │
  │  ───────────────────────────────────────                     │
  │  Cast activations to FP8 or NVFP4 before sending            │
  │                                                              │
  │  FP16 activation: 2 bytes per value                          │
  │  FP8  activation: 1 byte per value  → 2x less traffic       │
  │  NVFP4 activation: 0.5 bytes        → 4x less traffic       │
  │                                                              │
  │  Tradeoff: tiny precision loss, but network is the           │
  │  bottleneck so pack/unpack cost is negligible                │
  │                                                              │
  │  Done via NVIDIA Transformer Engine on Tensor Cores          │
  └──────────────────────────────────────────────────────────────┘


INFRASTRUCTURE CHECKLIST (making the hardware ready for MoE):
─────────────────────────────────────────────────────────────
  ┌─────────────────────────┬────────────────────────────────┐
  │ What                    │ Why                            │
  ├─────────────────────────┼────────────────────────────────┤
  │ NVLink Switch mesh      │ Full bandwidth intra-rack      │
  │ (e.g. NVL72)            │ all-to-all (up to 72 GPUs)    │
  ├─────────────────────────┼────────────────────────────────┤
  │ GPUDirect RDMA enabled  │ GPU→NIC→GPU without CPU copy  │
  │ on internode paths      │ for cross-rack transfers       │
  ├─────────────────────────┼────────────────────────────────┤
  │ InfiniBand link bonding │ Multiple physical ports → one  │
  │                         │ fat logical pipe + failover    │
  ├─────────────────────────┼────────────────────────────────┤
  │ NCCL algorithm choice   │ ncclAllToAll vs grouped        │
  │                         │ send/recv — profile both!      │
  ├─────────────────────────┼────────────────────────────────┤
  │ Profile link utilization│ Check BOTH NVLink AND NIC      │
  │                         │ paths are saturated            │
  └─────────────────────────┴────────────────────────────────┘


HOW THEY ALL FIT TOGETHER:
──────────────────────────
  Token arrives at MoE layer
         │
         ▼
  Router assigns expert ──→ Is expert LOCAL?
         │                      │
         │ YES                  │ NO
         ▼                      ▼
    Just compute         ┌─ COMPRESS (FP8/NVFP4) ─┐
                         │                         │
                         ▼                         │
                  Expert on same rack?             │
                    │          │                    │
                   YES         NO                  │
                    │          │                    │
                    ▼          ▼                    │
               NVLink      NVLink first            │
               shuffle     then NIC for            │
               (fast)      residual (HIERARCHY)    │
                    │          │                    │
                    ▼          ▼                    │
                  ┌─ BUTTERFLY SCHEDULE ─┐         │
                  │  (phased rounds,     │         │
                  │   no global barrier) │         │
                  └──────────────────────┘         │
                         │                         │
                         ▼                         │
                  ┌─ DOUBLE-BUFFER ──────┐         │
                  │  overlap with prev   │         │
                  │  batch's compute     │         │
                  └──────────────────────┘         │
                         │                         │
                         ▼                         │
                   Expert computes                 │
                         │                         │
                         ▼                         │
                  DECOMPRESS + send back           │
                  (another all-to-all)             │
                         │                         │
                         ▼                         │
                   Continue to next layer

Load balancing, capacity factor, and expert replication

- Problem: router may send disproportionate tokens to a few hot experts
  - Hot GPU at 99% while others sit at 60% → bottlenecks the entire cluster

- Training-time fix: load-balancing loss
  - Extra loss term penalizes gate for overusing some experts / underusing others
  - Result: trained model distributes tokens more evenly

- Inference-time fix 1: capacity factor
  - Each expert has a max token cap (e.g. capacity factor 1.2 = 120% of average load)
  - Overflow tokens → forwarded to next-best expert (by routing score) or serialized in a second pass
  - Common config: capacity factor 1.2 with top-2 gating

  Expert 3 capacity = 32 tokens, receives 40:
    first 32 → processed by Expert 3
    remaining 8 → rerouted to Expert with next-highest score

- Inference-time fix 2: expert replication
  - Clone hot expert onto another GPU → router can split tokens across original + clone
  - Replicas are transparent to the model — engine registers them as separate indices
  - Only replicate the few hot experts, not all → targeted cost increase
  - Replicas loaded from same checkpoint, never updated independently (prevents divergence)

  Before:  Expert 5 on GPU 2 only → overloaded
  After:   Expert 5 on GPU 2 + Expert 5 clone on GPU 6 → load split

Adaptive expert routing and real-time monitoring

- Standard gating: fixed routing scores from training, no runtime awareness
- Adaptive routing: inference engine intercepts gating softmax, adjusts based on live load

  How it works:
    1. Each expert GPU emits utilization metrics (Prometheus/Grafana)
    2. Engine sees GPU 3 at 99%, GPU 7 at 60%
    3. Engine biases gating scores: lowers Expert 3 weight, raises Expert 7 weight
    4. Some tokens diverted to less-loaded expert (slightly lower score, but available)

- Profiling: Nsight Systems timeline traces — if one GPU's all-to-all segment is much longer,
  it is processing more tokens than others → rebalance signal

- Actions the engine can take dynamically:
  - Adjust gating probabilities (bias scores)
  - Reassign experts to different GPUs
  - Spawn additional expert replicas (requires pre-provisioned capacity or fast model loading)

- Tradeoff: adds monitoring overhead, decision logic, configuration complexity
  - Not always worth it — profile your specific workload to decide

Dynamic routing — summary and operational notes

- Two core objectives of dynamic MoE routing:
  1. Reduce routing overhead (fast interconnects, overlap comms with compute, co-location)
  2. Evenly distribute work across expert GPUs (gating, capacity factor, replication)

- Expert placement matters: grouping experts to produce uniform token distribution
  → GPUs finish each layer more synchronously → better parallelism

- Modern MoE schedulers can spawn additional replicas on the fly if persistent imbalance detected

- Gating options (can combine):
  - Simple: top-1 or top-2 gating
  - Advanced: capacity-aware gates, dynamic replication, expert reassignment

- Scaling: with balanced load + minimal routing overhead → near-linear throughput scaling
  - Doubling GPUs can nearly double inference throughput

- Advanced optimizations:
  - Bypass / turn off underutilized experts entirely
  - Cache expert outputs → skip redundant compute for repeated token patterns

- Operational best practices:
  - Log all dynamic changes (replica spawns, gating bias adjustments, expert reassignments)
  - Alert when any expert utilization exceeds threshold (e.g. 80%)
  - Continuous profiling + adaptive algorithms to keep GPUs computing, not idling
