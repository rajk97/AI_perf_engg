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
