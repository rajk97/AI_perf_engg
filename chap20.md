Chapter 20: AI-assisted performance optimization and scaling

Big theme

- AI helps optimize AI systems
- Not just model code:
	- algorithms
	- kernels
	- compilers
	- scheduling
	- future research loops
- Goal: faster systems without only buying more hardware

Visual

	Old loop:
		human writes kernel → profiles → tunes → repeats

	AI-assisted loop:
		AI proposes code/algorithm
		→ verifier tests correctness + speed
		→ feedback improves next attempt
		→ best variant is kept

Core idea

- Performance optimization becomes search
- AI explores more variants than humans can try manually
- Human role shifts toward:
	- setting goals
	- building verifiers
	- checking edge cases
	- integrating winning ideas

AlphaTensor: AI-discovered matrix multiplication

What it did

- DeepMind used RL to search for faster GEMM algorithms
- Treated matrix multiplication as a single-player game
- Rediscovered Strassen-like ideas
- Found new algorithms for larger matrix sizes

Why GEMM matters

- GEMM powers nearly everything:
	- training forward pass
	- training backward pass
	- inference layers
	- attention/MLP blocks
- Small GEMM speedup → huge system-wide gain

Result

- On NVIDIA V100-era hardware
- Some discovered algorithms were ~10%-20% faster than cuBLAS at the time
- Like getting extra compute without new GPUs

Visual

	Hardware upgrade path:
		buy more GPUs → more speed

	Algorithm upgrade path:
		same GPUs + better math → more speed

Takeaway

- Even basic operations may still hide efficiency
- AI can search huge algorithm spaces without human bias
- Future targets:
	- convolutions
	- sorting
	- attention
	- communication patterns

Caveat

- AlphaTensor methods remain experimental
- Mainstream libraries need more validation/generalization first

DeepSeek-R1: automated CUDA attention kernel tuning

Setup

- NVIDIA tested DeepSeek-R1 for GPU kernel generation
- Task: write optimized attention kernel with relative positional encoding
- Model was given time to think/iterate
- Ran on H100

Loop

	for each attempt:
		R1 writes kernel
		verifier compiles + tests
		measure runtime
		feedback refines prompt
		try again

Key detail

- Verifier is the safety rail
- It checks:
	- correctness
	- speed
	- edge cases

Result

- Generated functionally correct CUDA attention kernel
- Speedup vs PyTorch FlexAttention: ~1.1x-2.1x
- KernelBench attention tests:
	- 100% basic pass
	- 96% complex pass

Why this matters

- Expert CUDA tuning can take hours/days
- AI produced strong variant in ~15 minutes
- Human still supervises, tests, and integrates

Takeaway

- LLM + verifier + profiler = practical optimization loop
- AI becomes a performance copilot
- Best use: generate variants quickly, then let tests/profilers judge

Predibase: RL-generated Triton kernels

Question

- Can an LLM learn to write high-performance Triton kernels?

Setup

- Model: Qwen2.5-Coder-32B-Instruct
- Training method: GRPO reinforcement learning
- Hardware: H100 cluster
- Target: replace PyTorch code with faster Triton kernels

Reward function

- Candidate kernel gets reward if it:
	- compiles
	- produces correct output
	- runs faster than baseline

Visual

	generate Triton kernel
		→ compile
		→ test correctness
		→ benchmark speed
		→ reward / penalty
		→ model improves

Result

- Correct kernels for all 13 tasks
- Success rose from near 0% to ~40% after ~5,000 steps
- Some kernels up to ~3x faster than baseline

Takeaway

- RL can align code generation with real performance metrics
- AI learns tricks like:
	- warp-level parallelism
	- fewer global memory reads
	- better tiling
	- less overhead

Why Triton matters

- Python-like GPU programming
- Lower barrier than raw CUDA
- Future path: AI writes/optimizes kernels continuously

Self-improving AI agents and future AI factories

Big direction

- Frontier labs are building enormous AI data centers
- Future training runs may use ~10^27-10^28 FLOPs
- Roughly 100x GPT-4-scale compute in some scenarios

Compute scale intuition

	GPT-3: ~3 × 10^23 FLOPs
	GPT-4: ~2 × 10^25 FLOPs
	Agent-1 idea: ~10^27-10^28 FLOPs

Agent-1 idea

- Self-improving model for research/code optimization
- Can generate + optimize code in real time
- Speeds up:
	- debugging
	- kernel fusion
	- experiment iteration
	- systems research

Agent-2 idea

- Always-learning system
- Instead of static train → checkpoint → deploy
- Continuously updates from fresh synthetic data
- Moves toward perpetual model improvement

Performance engineer takeaway

- Future bottleneck is not just hardware
- It is the loop speed:
	- generate idea
	- test it
	- measure it
	- improve it
- AI agents compress this loop

Human + AI role split

	AI:
		searches variants
		writes kernels
		tunes parameters
		finds surprising algorithms

	Human:
		sets constraints
		defines rewards
		builds verifiers
		checks safety/edge cases
		ships robust systems

Mnemonic: let AI search the optimization space, let verifiers judge correctness and speed, and let humans define the rewards, constraints, and production guardrails.

Agent-2 to Agent-4, smart compilers, and AI cluster operations

Agent-2 caveat: continual learning is hard

- Agent-2 idea = always retraining / always improving
- Big risk: catastrophic forgetting
- Meaning:
	- model learns new tasks
	- but older abilities degrade
- Stability is still an active research problem

Visual

	Good continual learning:
		old skills stay + new skills added

	Catastrophic forgetting:
		new skill improves, old skill fades

Agent-3 idea: superhuman coding workforce

- Hypothetical AI Futures Project scenario
- Uses:
	- neural scratchpads
	- iterated distillation/amplification
	- algorithmic breakthroughs
- Goal: fast, cheap, superhuman coding

Agent-3 scale thought experiment

- 200,000 copies in parallel
- Like tens of thousands of elite programmers
- Claimed ~30x faster operating speed
- Far beyond today, but useful directionally

What massive AI coding parallelism changes

- More ideas tested per day
- Faster R&D loop
- More kernels / algorithms / systems variants explored
- Human teams shift toward oversight + strategy

Agent-4 idea: self-rewriting researcher

- Hypothetical AGI-like system
- Can improve its own code / methods
- Uses mechanistic interpretability to inspect reasoning
- Pushes scientific + AI research faster than humans alone

Agent ladder

	Agent-1: better research/coding assistant
	Agent-2: continually learning model
	Agent-3: massively parallel superhuman coder
	Agent-4: self-improving researcher

Key takeaway

- AI systems performance is not just an engineering detail
- It controls how fast future AI research can improve
- Faster infrastructure → faster experiments → faster models

Smart compilers and automated code optimization

Trend

- Less hand-tuning every CUDA detail
- More compiler/AI automation
- Performance engineer guides tools instead of touching every knob

Framework direction

- PyTorch / TensorFlow / JAX increasingly automate:
	- graph optimization
	- op fusion
	- Tensor Core use
	- async memory movement
	- compute/communication overlap
	- precision choices

Triton

- Python-like GPU programming language
- Compiler emits optimized CUDA-style kernels
- Hides much low-level CUDA/PTX complexity
- New GPU features can be exposed through compiler updates

Visual

	Old path:
		write CUDA/PTX manually → tune for each GPU

	New path:
		write high-level Triton/PyTorch
		→ compiler maps to GPU features
		→ autotuner searches fast configs

CUDA Graphs

- Capture repeated GPU operation sequences
- Launch captured graph with low CPU overhead
- Useful when loop structure repeats
- APIs:
	- `cudaGraphInstantiate()`
	- `cudaGraphLaunch()`

Why graph execution helps

- Fewer CPU-GPU sync/launch overheads
- Enables whole-graph optimization
- Repeated training/inference steps run cheaper

Automatic distributed scheduling: future direction

- Framework may eventually infer:
	- pipeline parallelism
	- layer grouping
	- chunk sizes
	- TP/PP/DP/EP strategy
- Today, full automatic pipeline parallelism still needs human guidance

Possible future advisor

	User: model has 500B parameters
	Advisor: use 8-way TP per node,
	         4-way PP across nodes,
	         these layer groups,
	         these chunk sizes

Performance engineer role shift

- From manually trying endless configs
- To:
	- asking better optimization questions
	- setting constraints
	- verifying compiler/AI choices
	- handling edge cases

AI-assisted cluster operations

Big idea

- Automation moves from code → whole cluster
- AI scheduler watches live cluster state
- Optimizes jobs, requests, memory, KV cache, and failures

Current orchestrators

- Kubernetes / SLURM often use static heuristics
- Future schedulers can learn from telemetry:
	- GPU utilization
	- queue wait time
	- memory pressure
	- network load
	- job interference

Smart colocating

- Put compatible workloads together
- Example:
	- job A is compute-heavy
	- job B is memory-bandwidth-heavy
	- colocate if they do not fight badly
- Goal: maximize useful work, not just utilization

Goodput vs utilization

- 100% GPU busy is not always good
- Bad busy:
	- redundant transfers
	- stalled communication
	- useless retries
- Good busy:
	- useful neural compute
	- productive memory movement
	- progress on requests/jobs

NVIDIA Dynamo example

- Distributed inference framework
- Coordinates:
	- request scheduling
	- KV cache placement
	- data movement
	- microbatch assignment
	- failure rerouting
- Integrates with Kubernetes-style infrastructure

Weight streaming / activation offloading

- Stream model layers from host/storage when needed
- Keep less in GPU memory at once
- Helps serve extremely large models
- Useful for cheaper storage + huge model capacity

AI performance copilot

- Watches logs + metrics + traces
- Suggests fixes:
	- increase batch size if memory is underused
	- change learning rate if loss diverges
	- inspect data pipeline if throughput stalls
	- flag node/NIC/GPU anomalies

Anomaly examples

	Loss becomes NaN:
		maybe unstable gradients → try clipping / LR reduction

	GPU memory drops suddenly:
		maybe data stall / failed prefetch / memory leak pattern

	Job crashes after ECC errors:
		likely GPU HBM/channel hardware issue

Metrics pipeline idea

	Prometheus metrics + logs + traces
		→ LLM/AI assistant
		→ likely root cause
		→ suggested action

RL for live system control

- Power agent:
	- tune clocks / frequencies / allocations
	- maximize performance per watt
- Memory agent:
	- decide tensors to keep on GPU
	- decide what to offload to CPU/NVMe
- Network/cache agent:
	- congestion control
	- cache eviction
	- adaptive routing

Why AI helps operations

- Ultrascale clusters have too many knobs
- Workloads change constantly
- Human static rules miss nonobvious patterns
- AI can adapt 24/7 from live feedback

Human role in AI-operated clusters

- Set objectives
- Define safety/fairness guardrails
- Supervise novel incidents
- Let AI handle routine:
	- load balancing
	- failure recovery
	- memory tuning
	- resource scheduling

Final idea

- Hand-tuning everything will not scale
- AI-friendly automation lets humans focus on novel work
- Best performance teams use AI as an always-on optimizer, not just a chatbot

Mnemonic: future performance work is guide, verify, and guardrail; let compilers tune kernels, let agents operate clusters, and let humans steer the objectives.

Scaling to multimillion GPU clusters and 100T-parameter models

Big question

- We crossed trillion-parameter models
- Next target: 10T-100T+ parameters
- Needs full-stack scaling:
	- hardware
	- memory
	- networking
	- software
	- algorithms
	- AI-assisted optimization

Hardware pressure

- 100T parameters means huge storage + movement cost
- Need more:
	- HBM capacity
	- HBM bandwidth
	- interconnect bandwidth
	- cluster-scale memory

HBM trend

- HBM3e: Blackwell generation
- HBM4: Rubin generation
- HBM4 roughly doubles bandwidth per stack again
- Possible future GPU boards:
	- 512 GB HBM
	- 1 TB HBM
- More model state fits locally
- Less swapping / sharding pressure

Visual

	Before:
		model shard spread across many GPUs
		→ lots of communication

	After:
		more params fit per GPU
		→ fewer shards, less traffic, lower latency

Mega-GPU cluster idea

- Link many NVL72 racks together
- Goal: many GPUs act like one giant accelerator
- Needs NVLink/NVSwitch or similar fabric to keep scaling
- Communication wall is the real enemy

Low precision is mandatory

- FP8 / FP4 already matter
- Future may use even lower precision for some parts
- Hybrid strategy:
	- low precision for most layers
	- higher precision for sensitive layers
- Without low precision, 100T training is too slow/expensive

Sparse and conditional compute

- Do not activate all 100T parameters per token
- MoE is the main pattern:
	- huge total model
	- small active subset per token
- Throughput depends on:
	- GEMM speed
	- expert routing
	- expert placement
	- expert-output caching
	- sparse communication

MoE placement intuition

	Good:
		right expert near right tokens
		→ less network traffic

	Bad:
		experts far from tokens
		→ cross-rack traffic dominates

Memory-efficient algorithms

- Adam-like optimizers keep extra state:
	- momentum
	- variance
- This can triple memory use
- 100T weights can imply 200T extra optimizer values
- Need memory-saving optimizers:
	- Adafactor
	- Shampoo-style methods
	- sharded optimizer states

Activation checkpointing

- Store fewer activations
- Recompute during backward pass
- Trades compute for memory
- At 100T scale: likely aggressive/default

Rotating weight updates

- Radical idea: do not update every weight every step
- Update subsets over time
- Reduces per-step compute and communication
- Question to keep asking:
	- do we need this operation this often?
	- do we need this precision here?

Network and infrastructure

- 10,000+ GPUs means fabric matters as much as GPUs
- Need:
	- high bisection bandwidth
	- adaptive routing
	- fault tolerance
	- fast checkpoint/restart
	- topology-aware scheduling
- Spectrum-X example:
	- Ethernet/RoCE fabric optimized for AI
	- adaptive routing
	- reduced congestion

Unified memory at giant scale

- Goal: huge memory space across GPU/CPU/storage
- Programmer sees something closer to one pool
- Runtime handles placement/prefetch
- Tools:
	- Unified Memory
	- `cudaMemPrefetchAsync()`
	- on-demand paging
- At 100T scale, page placement becomes critical

Datacenter-scale thinking

- Frontier clusters may reach 1M+ GPUs
- One training job may span a datacenter
- Performance engineer must think at:
	- node scale
	- rack scale
	- datacenter scale
	- maybe multidatacenter scale

Socio-technical scaling

- Biggest models may exceed one team/company
- More like big science projects
- Needs shared standards:
	- checkpoint formats
	- training code
	- reproducibility
	- fault-tolerant snapshots
	- multiparty ownership

100T model recipe

	More HBM + faster fabric
	+ low precision
	+ sparse MoE
	+ memory-efficient optimizers
	+ checkpointing
	+ topology-aware scheduling
	+ AI-assisted compilers/agents
	= feasible ultrascale model

Chapter 20 key takeaways

Codesign wins

- Real performance comes from hardware + software + algorithms together
- Not one isolated trick

AI-assisted optimization works

- AlphaTensor: algorithm discovery for GEMM
- NVIDIA / DeepSeek-R1: CUDA attention kernels
- Predibase: RL-generated Triton kernels
- Pattern:
	- generate
	- verify
	- benchmark
	- refine

100T strategies

- Aggressive quantization
- Multidimensional parallelism:
	- data
	- tensor
	- pipeline
	- expert
	- context/sequence
- Careful inter-rack communication
- Smart scheduling is as important as raw FLOPs

Compute scaling

- Future AI datacenters aim for 100x-1000x more training FLOPs
- Bigger compute enables bigger models
- But efficiency decides cost and feasibility

Future AI agents

- Generate code
- Optimize kernels
- Continually learn
- Debug systems
- Possibly rewrite/improve their own methods
- Compress the time between breakthroughs

AI-assisted troubleshooting

- Watch logs, metrics, traces
- Detect:
	- accuracy/loss spikes
	- NaNs
	- hardware errors
	- memory stalls
	- network anomalies
- Suggest fixes or trigger automated actions

Performance per watt

- Core metric:
	- tokens/sec/$/watt
- More useful than raw peak FLOPs alone
- Grace Blackwell NVL72 claims major perf-per-watt gains over Hopper-era systems
- Lower watts per token = lower cost per token

Book-level conclusion

- AI systems performance engineering is now full-stack
- Scope expanded from single kernels to:
	- rack-scale systems
	- inference engines
	- network fabrics
	- memory hierarchy
	- AI-assisted optimization loops
	- datacenter-level scheduling

Mechanical sympathy

- Best systems understand the hardware deeply
- Tensor Cores, Transformer Engine, NVLink, NVSwitch, SHARP, HBM, KV cache, schedulers all matter
- Hardware/software/algorithm codesign is the winning pattern

Smart scaling over brute-force scaling

- Do more useful work per cycle
- Avoid wasted communication
- Use lower precision when safe
- Let AI tools handle routine tuning
- Humans focus on architecture, constraints, and new ideas

Final role of performance engineer

- Spot bottlenecks
- Guide AI tools
- Verify recommendations
- Build robust systems
- Balance:
	- efficiency
	- reliability
	- cost
	- sustainability

Final mindset

- Keep fundamentals strong
- Stay curious
- Experiment with new hardware/software
- Trust AI recommendations only with verification
- Adapt as the field moves into larger, stranger systems

Mnemonic: 100T-scale AI needs bigger memory, faster fabric, lower precision, sparse activation, smarter algorithms, AI-assisted tuning, and humans who can verify the whole stack.
