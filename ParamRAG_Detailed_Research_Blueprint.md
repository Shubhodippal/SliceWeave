# ParamRAG / Fine-Grained Conditional Parameter Activation
## Detailed Research Blueprint

**Working title:** ParamRAG: Fine-Grained Retrieval-Augmented Parameter Activation for Dense Language Models

**Primary research target:** Qwen3.8-27B

**Goal:** Build a method that preserves the practical quality of a large dense 27B model while activating only the small subset of parameters that matter for the current prompt/token.

---

# 1. Core Research Goal

The project is NOT about:

- sub-4-bit quantization,
- distillation,
- training a smaller student model,
- hardware-specific optimizations,
- replacing Qwen with a different 7B/13B model,
- ordinary static pruning,
- or simply routing between several independently trained small models.

The project is about taking ONE large pretrained dense model and reorganizing its existing computation into many tiny, fine-grained experts.

For any prompt, only the experts that matter should be used.

The desired behavior is:

```text
Qwen3.8-27B
    |
    |-- all original knowledge remains available
    |
    +--> prompt-level capability retrieval
            |
            +--> 8B–12B candidate parameter pool
                    |
                    +--> token/layer-level micro-expert routing
                            |
                            +--> ~4B–7B parameters actually executed per token
```

The model can retain the knowledge capacity of a 27B model while paying the compute cost of a much smaller model for many tasks.

---

# 2. Hard Constraints

The project should respect the following constraints.

## Model quality

The quality target is the practical quality of Qwen3.8-27B.

The goal is NOT:

> "make a decent small model."

The goal is:

> "make the large model execute only the relevant subset of itself while preserving its capabilities."

## Quantization

The final runtime should use at least 4-bit weights.

Preferred target:

```text
Q4 / Q4_K_M class precision
```

Do not rely on 2-bit or 3-bit quantization to reach the desired memory/computation target.

## No distillation

Do not train a smaller student model to imitate Qwen3.8-27B.

There should be:

- no teacher/student output matching as the main method,
- no model replacement,
- no knowledge distillation pipeline.

## No hardware-specific scientific claim

The methodology should work conceptually on:

- CPU,
- NVIDIA GPU,
- AMD GPU,
- Apple Silicon,
- ARM,
- or other hardware.

The local machine is only the final deployment target.

Final demonstration hardware:

```text
CPU: AMD Ryzen 7 5800H
RAM: 32 GB
GPU: RTX 3060 Laptop GPU
```

Cloud resources such as Kaggle or Colab may be used for research experiments.

---

# 3. Central Hypothesis

A 27B dense model does not require all 27B parameters to contribute significantly to every token.

Different inputs activate different computational pathways.

For example:

```text
Prompt:
"Debug this FastAPI JWT authentication bug."

Likely useful capabilities:
- English/language
- Python
- debugging
- FastAPI
- HTTP
- authentication
- security
- program reasoning
```

Knowledge related to unrelated domains should contribute much less:

```text
- poetry
- chemistry
- ancient history
- geography
- unrelated programming languages
- etc.
```

Therefore the main hypothesis is:

> A large dense LLM contains many latent fine-grained capability subnetworks, and only a small subset is necessary for each prompt/token.

If this is true, we can convert the dense model into a conditionally active model without training a completely new model.

---

# 4. Why Fine-Grained Experts

Do NOT create only coarse experts like:

```text
Coding expert
Math expert
General expert
Writing expert
```

These are too large and too crude.

Instead create thousands of very small micro-experts.

Examples of latent capabilities might correspond to things like:

```text
Python syntax
Python exceptions
SQL joins
pointer reasoning
graph algorithms
dynamic programming
regex
HTTP
authentication
symbolic algebra
proof reasoning
JSON
React
Git
Linux shell
memory management
recursion
sorting
API design
```

However, the system does not need to assign human-readable labels.

The experts may remain latent numerical clusters.

The important property is:

> Different experts become useful for different hidden states.

---

# 5. First Target: FFN / MLP Blocks

The first implementation should target the feed-forward network (FFN / MLP) layers.

Do NOT attempt full-model routing immediately.

Why start with FFNs?

1. They contain a large fraction of model parameters.
2. Their intermediate neurons are naturally separable.
3. They are easier to partition than attention.
4. If FFN sparsity does not exist, the whole project becomes much less promising.
5. FFN routing can be tested without redesigning the entire architecture.

A typical gated FFN can be written approximately as:

```text
z = SiLU(W_gate x) * (W_up x)
y = W_down z
```

The intermediate vector `z` contains many neurons.

Split these neurons into groups.

Example:

```text
Intermediate size: 17,408
Micro-expert group size: 128 neurons

17,408 / 128 = 136 micro-experts per layer
```

If the target model has 64 layers:

```text
136 * 64 = 8,704 FFN micro-experts
```

This provides the minute-level granularity required by the project.

---

# 6. Exact Micro-Expert Definition

Suppose the FFN intermediate vector is:

```text
z = [z_1, z_2, ..., z_d]
```

Partition neuron indices into groups:

```text
I_1, I_2, ..., I_E
```

For expert `e`, use only the neuron indices belonging to `I_e`.

Conceptually:

```text
y_e = W_down[:, I_e] * z[I_e]
```

Then the full FFN output is:

```text
y = y_1 + y_2 + ... + y_E
```

This is important because initially the decomposition can be exact.

No model quality is lost when all experts are active.

The only approximation appears when the router activates only some experts.

---

# 7. Research Phase 0 — Environment Setup

Use two environments.

## Local VS Code machine

Use for:

- writing code,
- debugging,
- unit tests,
- smaller models,
- analysis scripts,
- plotting,
- repository management,
- final runtime development.

Recommended local software:

```text
Python 3.11
VS Code
Git
PyTorch
Transformers
Accelerate
Safetensors
NumPy
Pandas
scikit-learn
matplotlib
```

Later:

```text
FAISS
llama.cpp
GGUF tooling
C++ / CUDA if needed for final runtime
```

## Kaggle / Colab / cloud notebook

Use for:

- full 27B experiments,
- activation collection,
- large benchmark sweeps,
- experiments requiring more VRAM,
- large-scale expert analysis.

The laptop should NOT be forced to perform every 27B research experiment.

The laptop is primarily the final deployment machine.

---

# 8. Development Model Progression

Do not start on the 27B model.

Use progressively larger models.

Recommended progression:

```text
Stage 1: Qwen-family ~4B model
Stage 2: Qwen-family ~9B model
Stage 3: Qwen3.8-27B
```

Purpose of 4B:

- verify code,
- inspect model structure,
- test hooks,
- confirm expert decomposition,
- test metrics,
- avoid wasting GPU time.

Purpose of 9B:

- stress-test scalability,
- detect memory problems,
- test routing across many layers.

Purpose of 27B:

- final scientific evaluation.

---

# 9. Experiment 1 — Oracle FFN Sparsity Test

This is the most important first experiment.

Do NOT build a router first.

First determine whether useful sparsity actually exists.

## Question

> If an oracle knew exactly which micro-experts were most important for a token, how many experts would be required to preserve the layer/model output?

## Procedure

For a real prompt:

1. Run the model normally.
2. Capture one FFN intermediate activation vector `z`.
3. Split `z` into micro-expert groups.
4. Measure the contribution/energy of each group.
5. Rank groups by importance.
6. Keep only a fraction of groups.
7. Zero the rest.
8. Run the remaining layer/model.
9. Compare logits to the baseline.

Test:

```text
100%
70%
50%
40%
30%
20%
15%
10%
5%
```

of micro-experts active.

## Initial importance score

A simple first score:

```text
score_e = mean(z_e^2)
```

where `z_e` is the activation vector for micro-expert `e`.

Later, improve the oracle score to include the downstream projection:

```text
score_e = || W_down[:, I_e] z[I_e] ||
```

This is more accurate because it measures actual output contribution.

---

# 10. Experiment 1 Metrics

Compare the sparse run against the full model.

Measure:

## Logit cosine similarity

```text
cos(logits_full, logits_sparse)
```

## Relative L2 error

```text
||logits_sparse - logits_full|| / ||logits_full||
```

## Top-1 next-token agreement

```text
argmax(logits_sparse) == argmax(logits_full)
```

## Top-k token overlap

For example:

```text
top-10 overlap
top-50 overlap
```

## KL divergence

Measure probability distribution distortion.

## Longer generation divergence

Later measure:

```text
number of tokens generated before sparse model first diverges from full model
```

---

# 11. Experiment 1 Success / Failure Criteria

Do not continue based on intuition.

Use explicit thresholds.

## Excellent signal

If around:

```text
20–30% active FFN experts
```

maintains extremely high logit similarity and next-token agreement across many prompts/layers.

This strongly supports continuing.

## Interesting signal

If around:

```text
30–50%
```

of experts are sufficient.

Still worth investigating.

## Weak signal

If:

```text
>60–70%
```

of experts are required most of the time.

Then the potential compute savings may be too small.

## Failure

If even 50% removal destroys behavior across many layers/tasks.

Then the original hypothesis is weak and should be reconsidered.

---

# 12. Do Not Confuse Oracle Routing With a Real Router

Experiment 1 is deliberately cheating.

It computes the true intermediate activations first.

That means it already spent the compute that the final system is supposed to save.

This is acceptable because Experiment 1 is not the final algorithm.

It answers only:

> Is sparse expert execution theoretically useful?

If the oracle fails, stop.

If the oracle succeeds, build a predictor.

---

# 13. Experiment 2 — Predict Experts Before Computing Them

Now create a real router.

Input:

```text
current hidden state x
```

Output:

```text
list of micro-experts likely to matter
```

The router must be much cheaper than the experts it saves.

Possible first router:

```text
score_e = cosine(x, centroid_e)
```

where `centroid_e` represents hidden states that historically made expert `e` important.

Select the highest-scoring experts.

This gives:

```text
hidden state
    |
    v
cheap router
    |
    v
selected micro-experts
    |
    v
compute only selected experts
```

---

# 14. Building Expert Centroids

During calibration:

1. Run many prompts.
2. For every layer/token:
   - record hidden state `x`,
   - calculate which experts were important.
3. For each expert:
   - collect hidden states where the expert ranked highly.
4. Compute an average or learned embedding.

Example:

```text
centroid_e = mean(hidden states where expert e is important)
```

Then at inference:

```text
score_e = cosine(x, centroid_e)
```

This is a simple baseline.

Do not begin with a neural router unless necessary.

---

# 15. Experiment 3 — Activation-Based Neuron Clustering

Arbitrary contiguous groups of 128 neurons are only the first test.

Later, build better micro-experts.

## Goal

Group neurons that tend to activate together.

For neuron `j`, collect its activation signature:

```text
s_j = [a_j1, a_j2, ..., a_jN]
```

over many calibration tokens.

Then cluster neurons using:

```text
MiniBatchKMeans
spectral clustering
hierarchical clustering
or another scalable method
```

Possible objectives:

- similar activation patterns,
- correlated contribution,
- similar task dependence,
- similar downstream projection behavior.

After clustering, physically reorder neurons so each expert is contiguous in memory.

---

# 16. Exact Neuron Reordering

For a gated FFN, neuron permutations can be applied consistently.

If the same permutation is applied to:

```text
rows of gate projection
rows of up projection
columns of down projection
```

the full dense FFN computation remains unchanged.

This is useful because:

- related neurons become contiguous,
- micro-experts can be loaded efficiently,
- final kernels can operate on expert blocks.

Always verify numerically that the reordered dense FFN matches the original before introducing sparsity.

---

# 17. Experiment 4 — Adaptive Number of Experts

Do NOT use a fixed Top-K forever.

Some tokens are easy.

Some are difficult.

Let the system dynamically choose how many experts to activate.

Example:

```text
easy token       -> 8 experts
normal token     -> 16 experts
difficult token  -> 32 experts
very uncertain   -> 64 experts or fallback
```

Possible confidence signals:

- router score entropy,
- gap between selected and unselected scores,
- prediction uncertainty,
- logit margin,
- hidden-state novelty,
- disagreement between multiple router estimates.

Define:

```text
K_t = f(router uncertainty)
```

This gives adaptive compute.

---

# 18. Experiment 5 — Prompt-Level Capability Retrieval

Token-level routing alone still considers all micro-experts.

Add a first-level gateway.

Before generation, analyze the prompt and select a candidate expert pool.

Example:

```text
Prompt:
"Implement Dijkstra in Rust using a binary heap."

Prompt-level retrieval may select experts related to:

- code
- Rust
- graphs
- algorithms
- data structures
- program reasoning
```

Then token routing only searches inside that pool.

Conceptually:

```text
27B parameter universe
        |
        v
prompt-level retrieval
        |
        v
8B–12B candidate parameter pool
        |
        v
token/layer router
        |
        v
4B–7B executed parameters/token
```

---

# 19. ParamRAG Analogy

The project can be explained using an analogy to retrieval-augmented generation.

Traditional RAG:

```text
query
  |
retrieve relevant documents
  |
LLM
```

ParamRAG:

```text
query
  |
retrieve relevant model parameters / experts
  |
run only those parameters
```

The retrieval database is not external text.

It is the model's own parameter space.

This is why the name "ParamRAG" is useful as a working title.

---

# 20. Expert Addressing

Every micro-expert should have a compact address/embedding.

Example:

```text
expert_embedding[e] in R^d
```

The hidden state or prompt produces:

```text
query_embedding
```

Then retrieve:

```text
TopK(query_embedding dot expert_embedding[e])
```

Possible addressing systems:

- centroid similarity,
- learned low-rank router,
- locality-sensitive hashing,
- FAISS nearest-neighbor search,
- hierarchical expert trees,
- product quantization for router embeddings.

Start simple.

Only add complexity if the simple method fails.

---

# 21. Hierarchical Routing

A scalable design may use multiple routing levels.

Example:

```text
Level 1: task family
Level 2: capability cluster
Level 3: layer-specific experts
```

Instead of comparing against thousands of experts directly:

```text
prompt
  |
choose 20 broad clusters
  |
choose 100 candidate micro-experts
  |
activate 10–30 experts per layer/token
```

This reduces routing cost.

---

# 22. Candidate Parameter Budget vs Active Parameter Budget

Keep these three quantities separate.

## Total parameter pool

All knowledge available:

```text
~27B
```

## Candidate parameters per request

Selected after prompt-level retrieval:

```text
target: ~8B–12B
```

## Actually executed parameters per token

Selected by token/layer router:

```text
stretch target: ~4B–7B
```

These are not the same thing.

The final paper must report all three.

---

# 23. Why the Full Model File May Still Be Large

If all 27B Q4 weights are stored, the total on-disk model can still be large.

Fine-grained routing mainly reduces:

- active memory working set,
- memory traffic,
- computation,
- latency.

If the project additionally requires the entire file to be around 10–12 GB, a second structural-storage method will be necessary.

Possible later research directions:

- shared parameter blocks,
- duplicate block elimination,
- lossless entropy coding,
- common-core storage,
- cross-expert dictionary sharing.

Do NOT add this complexity until conditional execution is proven useful.

---

# 24. Final Runtime Target

The eventual local demonstration should run on:

```text
Ryzen 7 5800H
32 GB RAM
RTX 3060 Laptop GPU
```

Desired final behavior:

```text
full model knowledge pool available locally
Q4 or better precision
prompt-level module selection
token-level micro-expert routing
adaptive compute budget
no cloud required during inference
```

Possible final memory organization:

```text
GPU VRAM:
- hot/shared blocks
- frequently selected experts
- current KV/state

System RAM:
- cold micro-experts
- additional task modules
- routing metadata
```

This is an implementation choice, not the scientific contribution.

---

# 25. Research Software Stack

## Development

```text
Python 3.11
PyTorch
Hugging Face Transformers
Accelerate
Safetensors
Hugging Face Datasets
NumPy
Pandas
scikit-learn
matplotlib
```

## Expert retrieval

Start with:

```text
PyTorch matrix similarity
```

Later:

```text
FAISS
```

## Evaluation

Use:

```text
lm-evaluation-harness
custom generation-agreement scripts
coding benchmarks
reasoning benchmarks
```

## Final inference runtime

Eventually:

```text
llama.cpp / GGML / GGUF
```

with custom support for:

```text
micro-expert parameter layout
routing metadata
conditional expert kernels
```

Do NOT begin with llama.cpp.

---

# 26. Repository Structure

Recommended repository:

```text
paramrag/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── qwen4b.yaml
│   ├── qwen9b.yaml
│   └── qwen27b.yaml
│
├── src/
│   ├── model_utils.py
│   ├── hooks.py
│   ├── micro_experts.py
│   ├── oracle.py
│   ├── routing.py
│   ├── clustering.py
│   ├── metrics.py
│   └── datasets.py
│
├── experiments/
│   ├── 01_oracle_ffn.py
│   ├── 02_oracle_all_layers.py
│   ├── 03_collect_activations.py
│   ├── 04_cluster_neurons.py
│   ├── 05_centroid_router.py
│   ├── 06_adaptive_k.py
│   ├── 07_prompt_gateway.py
│   └── 08_full_generation.py
│
├── evaluation/
│   ├── next_token.py
│   ├── generation_agreement.py
│   ├── coding.py
│   └── lm_eval.py
│
├── runtime/
│   ├── gguf/
│   └── llama_cpp/
│
├── results/
│   ├── csv/
│   ├── plots/
│   └── logs/
│
└── notebooks/
    ├── kaggle_27b.ipynb
    └── analysis.ipynb
```

---

# 27. Exact VS Code Starting Steps

After opening VS Code:

## Step 1

Open the project folder.

## Step 2

Open Terminal.

## Step 3

Create Python environment:

```powershell
py -3.11 -m venv .venv
```

## Step 4

Activate:

```powershell
.venv\Scripts\activate
```

## Step 5

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install torch transformers accelerate safetensors huggingface_hub numpy pandas scikit-learn
```

If NVIDIA CUDA-enabled PyTorch is not installed correctly, install the appropriate PyTorch build for the local CUDA environment.

## Step 6

Check GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Step 7

Run:

```powershell
python experiments/01_oracle_ffn.py
```

---

# 28. First Script Responsibilities

The first script should:

1. Load a small Qwen-family model.
2. Find FFN `down_proj` modules.
3. Select one middle layer.
4. Run a baseline prompt.
5. Capture the intermediate FFN vector.
6. Group neurons into blocks.
7. Rank blocks by true activation importance.
8. Mask low-ranked blocks.
9. rerun forward pass.
10. compare final logits.
11. save results to CSV.

This script is NOT yet intended to accelerate inference.

It is a scientific falsification tool.

---

# 29. Experiment 2 — All-Layer Oracle Sweep

Once one layer works, extend the experiment.

For every layer:

```text
layer 0
layer 1
...
layer N
```

test multiple retention fractions.

Create a result table:

```text
Layer | 10% | 20% | 30% | 40% | 50%
-------------------------------------
0     | ... | ... | ... | ... | ...
1     | ... | ... | ... | ... | ...
2     | ... | ... | ... | ... | ...
...
```

Metrics:

- logit cosine,
- top-1 match,
- KL divergence,
- output error,
- benchmark degradation.

Plot layer sensitivity.

This may reveal:

```text
early layers require 50%
middle layers require 20%
late layers require 35%
```

The final router should therefore use layer-specific budgets.

---

# 30. Experiment 3 — Dataset Diversity

Do not test only coding prompts.

Use categories such as:

```text
coding
math
reasoning
general knowledge
writing
instruction following
long context
```

Measure whether different task types activate different experts.

This is crucial for the "capability-addressable parameter" hypothesis.

If the same experts dominate all tasks, prompt-level expert retrieval may have little value.

If expert usage differs strongly by task, the hypothesis becomes much stronger.

---

# 31. Expert Specialization Metrics

Possible measurements:

## Expert activation frequency

```text
P(expert active)
```

## Task conditional activation

```text
P(expert active | coding)
P(expert active | math)
P(expert active | writing)
```

## Mutual information

Measure how predictive expert activation is of task type.

## Expert entropy

Experts with low task entropy are specialized.

Experts with high task entropy may form the universal core.

This could naturally reveal:

```text
shared core experts
+
specialized capability experts
```

without manually defining them.

---

# 32. Universal Core Discovery

Some experts may be active almost all the time.

Keep them always active.

Example:

```text
always-active core = 2B–4B parameters
```

Then route only the remaining experts.

Conceptually:

```text
Universal core
      +
retrieved capability modules
      +
token-specific micro-experts
```

This may dramatically simplify routing.

---

# 33. Possible Final Architecture

A mature version may look like:

```text
                Prompt
                  |
                  v
        Prompt capability encoder
                  |
                  v
       Candidate capability modules
                  |
         +--------+--------+
         |                 |
         v                 v
  Always-active core   retrieved pool
         |                 |
         +--------+--------+
                  |
                  v
           Transformer layer
                  |
                  v
        token-level micro-router
                  |
                  v
      selected FFN/attention experts
                  |
                  v
              next layer
```

---

# 34. Later Extension — Attention / DeltaNet Routing

Do not begin here.

Once FFN routing works, investigate whether additional computation can be conditionally skipped in:

- attention heads,
- projection blocks,
- DeltaNet/linear-attention components,
- state update components.

Possible ideas:

```text
head-level routing
head-group routing
projection block routing
conditional state updates
```

This is required if the final target is extremely low active parameter counts such as 4B–7B total.

---

# 35. Quality Protection

The router should have a fallback mechanism.

If confidence is low:

```text
increase active expert count
```

If still uncertain:

```text
activate larger candidate pool
```

Extreme fallback:

```text
use full layer/model computation
```

Therefore the system can trade compute for confidence.

Conceptually:

```text
easy token -> 20% experts
normal token -> 35%
hard token -> 60%
uncertain token -> 100%
```

This may preserve quality better than fixed sparsity.

---

# 36. Main Baselines

The final paper should compare against:

```text
Full Qwen3.8-27B Q4
Static structured pruning
Random expert selection
Magnitude-based selection
Activation magnitude pruning
Fixed Top-K routing
Prompt-only routing
Token-only routing
Existing dense-to-MoE conversion baselines
```

Ablations:

```text
contiguous experts vs clustered experts
64 vs 128 vs 256 neurons/expert
fixed K vs adaptive K
centroid router vs learned router
prompt routing only
token routing only
both combined
always-active core size
```

---

# 37. Evaluation Categories

## Coding

Possible benchmarks:

```text
HumanEval+
MBPP+
LiveCodeBench
MultiPL-E
```

## Reasoning

Possible benchmarks:

```text
GSM-style math
reasoning suites
logic benchmarks
```

## General capability

Use standard language-model evaluation tasks.

## Generation agreement

Measure agreement with full Q4 parent.

## Efficiency

Measure:

```text
active parameters/token
candidate parameters/request
bytes read/token
tokens/sec
VRAM
RAM
router latency
energy if practical
```

---

# 38. Most Important New Metric

Report:

```text
Active Parameter Ratio (APR)
```

Define:

```text
APR = parameters executed per token / total model parameters
```

Example:

```text
27B model
6B average active

APR = 6 / 27 = 22.2%
```

Also report:

```text
Candidate Parameter Ratio
```

and:

```text
Quality Retention
```

Example:

```text
quality retention = sparse benchmark score / full model score
```

---

# 39. Desired Research Result

A very strong outcome would look conceptually like:

```text
Parent:
Qwen3.8-27B Q4
27B active parameters/token
baseline quality = 100%

ParamRAG:
8–12B candidate parameters/request
5–7B average executed parameters/token
quality retention >= 98–99%
large reduction in bytes/token
large speed improvement
```

These are research targets, not promised results.

---

# 40. Stop / Pivot Criteria

The project should be falsifiable.

Stop or radically change direction if:

1. Oracle routing still requires >60–70% of FFN experts.
2. Expert importance changes so unpredictably that cheap routing cannot predict it.
3. Router overhead consumes most of the saved compute.
4. Sparse memory access destroys hardware efficiency.
5. Quality collapses before meaningful compute savings are achieved.
6. Expert clustering does not produce meaningful specialization.
7. Prompt-level expert usage is nearly identical across task categories.

Do not keep building just because the idea sounds attractive.

---

# 41. Continue Criteria

Continue aggressively if:

1. Oracle routing preserves logits with <=30–40% FFN experts.
2. Expert usage differs strongly across tasks.
3. A simple centroid router reaches close to oracle selection quality.
4. Adaptive K substantially protects hard tokens.
5. Full-generation quality remains near the parent model.
6. Sparse execution provides real wall-clock savings.

---

# 42. Paper Contribution Candidates

If successful, possible contributions include:

## 1. Fine-Grained Parameter Retrieval

Treat model parameter blocks as retrievable objects.

## 2. Capability-Addressable Micro-Experts

Discover latent capability modules from a dense pretrained model.

## 3. Hierarchical Prompt + Token Routing

Prompt-level retrieval narrows the model; token-level routing performs fine-grained activation.

## 4. Adaptive Parameter Budget

Spend more model capacity only when uncertainty requires it.

## 5. Post-Training Dense-to-Conditional Conversion

Convert an existing dense LLM without distillation or training a new architecture from scratch.

## 6. Active Parameter Metrics

Measure effective model size dynamically rather than only stored parameter count.

---

# 43. Important Novelty Warning

The following ideas alone are NOT enough for a paper:

```text
convert dense FFN to MoE
use Top-K experts
use magnitude pruning
split FFN neurons into groups
route with a classifier
make coding/math/general experts
```

These areas already have substantial prior work.

The paper must combine them into a genuinely distinct mechanism.

The strongest potential novelty is:

```text
fine-grained capability-indexed parameter retrieval
+
hierarchical prompt/token routing
+
adaptive parameter budget
+
post-training conversion of one dense parent
```

A thorough literature review is mandatory before claiming novelty.

---

# 44. Immediate Development Roadmap

## Milestone 1

One-layer oracle test.

Deliverable:

```text
oracle_ffn.csv
```

## Milestone 2

All-layer oracle sweep.

Deliverables:

```text
layer_sensitivity.csv
layer_sensitivity.png
```

## Milestone 3

Multi-task activation collection.

Deliverable:

```text
expert_task_statistics.parquet
```

## Milestone 4

Neuron clustering.

Deliverables:

```text
expert_assignments.npy
expert_centroids.npy
```

## Milestone 5

Centroid router.

Deliverable:

```text
router_vs_oracle.csv
```

## Milestone 6

Adaptive-K router.

Deliverable:

```text
quality_vs_active_params.csv
```

## Milestone 7

Prompt gateway.

Deliverable:

```text
prompt_retrieval_results.csv
```

## Milestone 8

Full-model generation.

Deliverables:

```text
generation_agreement.csv
benchmark_results.csv
```

## Milestone 9

27B cloud experiment.

Deliverable:

```text
qwen38_27b_results/
```

## Milestone 10

Optimized local runtime.

Deliverable:

```text
custom GGUF / llama.cpp branch
```

---

# 45. Recommended Research Order

The correct order is:

```text
1. prove sparsity exists
2. prove experts can be predicted
3. prove routing generalizes
4. prove quality survives generation
5. prove computational savings
6. optimize runtime
7. test 27B
8. final benchmarks
9. write paper
```

The wrong order is:

```text
1. modify llama.cpp
2. build a huge router
3. create a custom file format
4. discover that 80% of experts are required
```

Always test the scientific assumption first.

---

# 46. First Week Plan

## Day 1

Set up repository and environment.

Run 4B model.

Inspect FFN module names.

Verify baseline generation.

## Day 2

Implement micro-expert masking.

Verify 100% experts exactly matches baseline.

Run:

```text
50%
30%
20%
10%
```

retention.

## Day 3

Repeat across 5–10 layers.

Save results.

## Day 4

Test 20–50 prompts from coding, math and general tasks.

Measure sparsity differences.

## Day 5

Improve expert score from activation magnitude to true output contribution.

## Day 6

Create first plots.

Determine whether oracle sparsity is strong enough.

## Day 7

Decision:

```text
continue
pivot
or stop
```

based on evidence.

---

# 47. What NOT to Work On Yet

Do not work on:

- final GUI,
- llama.cpp modifications,
- custom CUDA kernels,
- 27B full benchmark suite,
- expert human-readable naming,
- perfect Q4 serialization,
- publication writing,
- patent work,
- multiple hardware platforms.

These are premature.

---

# 48. Immediate Next Action

Open VS Code.

Run the starter experiment.

The first scientific question is:

> How much of one real Qwen FFN can be removed for one token before the final model output changes significantly?

Everything else depends on that answer.

---

# 49. Final Vision

The final system should behave like:

```text
                  Qwen3.8-27B knowledge
                           |
                           v
                 parameter capability index
                           |
                 +---------+----------+
                 |                    |
                 v                    v
         prompt-level retrieval   shared core
                 |                    |
                 +---------+----------+
                           |
                           v
                 token-level routing
                           |
                           v
             only required micro-experts
                           |
                           v
                       output
```

Target conceptual result:

```text
27B knowledge capacity
8–12B request working set
4–7B average executed parameters/token
Q4 or better precision
near-parent-model quality
fully local final inference
```

The key scientific question is not:

> "Can we make Qwen smaller?"

It is:

> "Can we make a large dense LLM behave as a dynamically assembled sparse model whose active parameter count is determined by what the current problem actually requires?"

That is the research project.
