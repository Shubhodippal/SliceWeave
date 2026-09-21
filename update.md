# ParamStream / ParamRAG v2
## Unified Research Blueprint for Quality-Adaptive Parameter Retrieval, Conditional Execution, and Hierarchical Memory Streaming

**Status:** Updated combined research plan  
**Primary objective:** Build a high-quality coding LLM that runs completely offline on a consumer laptop while also producing a publishable systems/ML research contribution.  
**Primary deployment machine:** AMD Ryzen 7 5800H, 32 GB RAM, RTX 3060 Laptop GPU with 6 GB VRAM, 700-800 GB NVMe SSD.  
**Primary scientific target:** Qwen3.8-27B dense model.  
**Stretch deployment target:** Larger sparse/MoE coding models such as Qwen3-Coder-Next, once the runtime layer is proven.  
**Minimum desired interactive speed:** >= 10 tokens/second.  
**Quality target:** Preserve practical coding/reasoning quality as close as possible to the full quantized parent model.  
**Inference requirement:** Fully offline after preparation/calibration.

---

# 1. Updated Project Thesis

The project should no longer be treated as only "ParamRAG" or only "SSD streaming."

The combined research idea is:

> Convert a pretrained dense LLM into a fine-grained conditionally active model after training, retrieve only the parameter blocks needed for the current request/token, progressively activate additional blocks when confidence is low, and place/prefetch those blocks across SSD, RAM, CPU, and GPU according to predicted future use.

The complete system therefore attacks two separate bottlenecks:

1. **Too much computation per token**
   - solved by fine-grained conditional parameter activation.

2. **Too much memory movement / insufficient VRAM**
   - solved by hierarchical memory placement, caching, and predictive prefetch.

The project should attempt to preserve the full model as the knowledge pool while reducing the amount of the model that is actually executed and moved per token.

---

# 2. Proposed Working Name

## Preferred name

**ParamStream**

### Expanded title

**ParamStream: Quality-Adaptive Hierarchical Parameter Retrieval and Streaming for Dense Language Models**

Alternative titles:

- ParamRAG v2
- Dynamic Parameter Retrieval for Dense LLMs
- Capability-Addressable Parameter Streaming
- Adaptive Micro-Expert Retrieval for Memory-Constrained LLM Inference
- Hierarchical Conditional Parameter Execution for Consumer Hardware

The name can change later. The research mechanism matters more than branding.

---

# 3. Dual Objective

The project has two simultaneous goals.

## Goal A - Practical Offline Coding Assistant

Run a strong coding model entirely offline on:

```text
CPU  : AMD Ryzen 7 5800H
RAM  : 32 GB
GPU  : RTX 3060 Laptop GPU, 6 GB VRAM
SSD  : 700-800 GB NVMe
```

Desired behavior:

```text
high coding quality
+
>= 10 tok/s average generation target
+
no cloud during inference
+
Q4 or better practical precision
+
large context if memory permits
```

## Goal B - Publishable Research

The research contribution must be more than:

- ordinary quantization,
- ordinary pruning,
- existing MoE inference,
- simple CPU offload,
- simple SSD offload,
- static layer placement,
- fixed Top-K sparse FFN,
- or a basic router.

The paper should study whether one pretrained dense model can be transformed into a **quality-adaptive, conditionally executed, memory-hierarchical model** without replacing it with a smaller student.

---

# 4. Core Scientific Question

The central question becomes:

> Can a pretrained dense coding LLM be converted post-training into a dynamically assembled sparse model whose active parameter count, memory placement, and precision are determined by the current prompt/token while retaining near-parent-model quality?

This is stronger than asking:

> Can we make the model smaller?

The model remains large.

What changes is:

```text
how much is active
how much is moved
where it is stored
where it is executed
and how aggressively it is approximated
```

---

# 5. Why Keep Qwen3.8-27B as the Main Scientific Target

Qwen3.8-27B is useful for the paper because it is dense.

For a dense model:

```text
27B total
~27B nominally active/token
```

If the project can reduce this to:

```text
~4B-6B average active/token
```

without major quality loss, that is scientifically more interesting than merely streaming an already sparse MoE.

The dense target allows the paper to investigate:

> Does a large dense model contain latent conditional subnetworks that can be exposed after training?

This is the main ParamRAG/ParamStream hypothesis.

A native MoE model can be used later as:

- a systems comparison,
- an additional deployment target,
- or a second paper/extension.

---

# 6. Recommended Model Strategy

Use three model classes during the project.

## Track 1 - Small development model

A Qwen-family ~4B model.

Purpose:

- implement hooks,
- verify exact FFN decomposition,
- run oracle sparsity experiments,
- build routing code,
- test clustering,
- debug kernels.

## Track 2 - Intermediate model

A Qwen-family ~7B-9B model.

Purpose:

- test scalability,
- benchmark real wall-clock sparse execution,
- evaluate router overhead,
- validate memory layout.

## Track 3 - Main research model

**Qwen3.8-27B**

Purpose:

- final scientific evaluation,
- coding benchmark evaluation,
- local runtime target.

## Stretch Track - Native MoE coding model

Examples:

```text
Qwen3-Coder-30B-A3B
Qwen3-Coder-Next class models
```

Purpose:

- test whether ParamStream's memory scheduler also benefits native MoEs,
- compare dense-to-conditional conversion against native conditional computation,
- attempt a stronger practical coding assistant after the core method is validated.

---

# 7. High-Level Architecture

```text
                       Prompt
                         |
                         v
              +----------------------+
              | Prompt Capability    |
              | Retriever            |
              +----------+-----------+
                         |
                         v
                Candidate parameter pool
                     ~8B-12B
                         |
                         v
              +----------------------+
              | Token/Layer Router   |
              | micro-expert Top-K   |
              +----------+-----------+
                         |
                         v
                  Initial active set
                     ~3B-5B
                         |
                         v
              +----------------------+
              | Quality / Residual   |
              | Confidence Estimator |
              +----------+-----------+
                         |
               +---------+---------+
               |                   |
           confident           uncertain
               |                   |
               v                   v
            execute         activate more
                                   |
                                   v
                            5B -> 8B -> 12B
                                   |
                                   v
                           full layer fallback
```

At the same time:

```text
SSD  <---->  RAM  <---->  VRAM
 L3           L2           L1

cold          warm         hot
experts       experts      experts
```

The router and scheduler work together.

---

# 8. Main Research Components

The combined system has six major components.

## 8.1 Exact Dense-to-Micro-Expert Decomposition

Split dense FFN neurons into many small parameter blocks.

Initially:

```text
dense FFN
=
sum of all micro-expert contributions
```

When all blocks are active, the transformed model should numerically match the original model.

This gives an exact starting point before introducing approximation.

## 8.2 Capability-Aware Parameter Retrieval

Use prompt-level information to reduce the parameter search space before token generation.

Example:

```text
Prompt:
"Fix this async FastAPI authentication race condition."

Candidate capabilities:
Python
async programming
FastAPI
HTTP
authentication
security
debugging
program reasoning
```

The system does not need human-readable expert labels.

Latent numerical clusters are sufficient.

## 8.3 Token/Layer-Level Conditional Execution

For each hidden state, activate only micro-experts predicted to matter.

## 8.4 Quality-Adaptive Progressive Activation

Do not commit to one fixed Top-K.

Start small.

If uncertainty is high, activate more parameters.

## 8.5 Predictive Hierarchical Memory Placement

Predict which blocks will be needed later and move them before they are requested.

## 8.6 Heterogeneous Execution

Some blocks execute on GPU.

Some blocks may execute on CPU.

The scheduler should choose based on:

- block location,
- transfer cost,
- compute cost,
- reuse probability,
- latency.

---

# 9. Exact FFN Micro-Expert Formulation

For a gated FFN:

```text
z = SiLU(W_gate x) * (W_up x)
y = W_down z
```

Partition the intermediate dimension into groups:

```text
I_1, I_2, ..., I_E
```

For expert `e`:

```text
z_e = z[I_e]
y_e = W_down[:, I_e] z_e
```

The complete FFN is:

```text
y = sum_e y_e
```

If all micro-experts are active:

```text
y_sparse(all experts) == y_dense
```

up to normal numerical tolerance.

This exact decomposition is important.

The approximation begins only when some experts are skipped.

---

# 10. Micro-Expert Granularity

Test multiple block sizes.

Example:

```text
64 neurons/expert
128 neurons/expert
256 neurons/expert
512 neurons/expert
```

Tradeoff:

```text
small blocks
    -> fine control
    -> higher routing/indexing overhead

large blocks
    -> lower routing overhead
    -> less precise sparsity
```

The paper should determine the best granularity empirically.

Do not assume 128 neurons is optimal.

---

# 11. Neuron Reordering

Contiguous neuron groups are easy to implement but may not be semantically meaningful.

Later, cluster neurons by activation behavior.

For a permutation `P`, apply the same permutation consistently to:

```text
rows of W_gate
rows of W_up
columns of W_down
```

This preserves the dense computation exactly.

Then physically store related neurons together.

Benefits:

- better specialization,
- contiguous block reads,
- simpler sparse kernels,
- improved cache behavior.

---

# 12. Phase 1 - Oracle Sparsity Test

Do not build the final router first.

First answer:

> If an oracle knew which micro-experts mattered, how many are actually necessary?

Procedure:

1. Run the normal model.
2. Capture FFN intermediate activations.
3. Split the FFN into blocks.
4. Compute each block's true contribution.
5. Rank blocks.
6. Keep only the top fraction.
7. Recompute output.
8. Measure deviation.
9. Repeat across layers, prompts, and tasks.

Retention sweep:

```text
100%
80%
70%
60%
50%
40%
30%
25%
20%
15%
10%
5%
```

---

# 13. Better Oracle Importance Score

Start with activation energy:

```text
score_e = mean(z_e^2)
```

Then use actual output contribution:

```text
score_e = ||W_down[:, I_e] z_e||
```

Possible advanced score:

```text
score_e =
output contribution
*
sensitivity estimate
```

Optional future work:

- gradient-based sensitivity,
- Hessian approximation,
- influence estimates,
- loss-aware contribution.

The oracle is allowed to be expensive.

It is a scientific upper bound, not the runtime algorithm.

---

# 14. Oracle Success Criteria

## Strong result

If <= 20-30% of FFN blocks preserve near-identical behavior across many prompts.

## Useful result

If <= 30-45% is enough.

## Weak result

If 60-70% is usually required.

## Failure signal

If removing even 30-40% consistently destroys logits or benchmark quality.

If the oracle fails, do not continue building a complicated runtime.

---

# 15. Phase 2 - Discover a Universal Core

Some blocks may be important for almost every token.

Measure:

```text
P(expert active)
```

and task-conditional activation:

```text
P(expert active | coding)
P(expert active | math)
P(expert active | general)
P(expert active | debugging)
```

Potential architecture:

```text
Always-active universal core
+
Prompt-retrieved capability blocks
+
Token-specific blocks
```

Example research target:

```text
Universal core       : 2B-3B
Prompt candidate pool: 8B-12B
Average executed     : 4B-6B
```

These are targets, not assumptions.

---

# 16. Phase 3 - Prompt-Level Parameter Retrieval

Before generation:

1. Encode the prompt.
2. Retrieve capability clusters.
3. Build a request-level candidate set.

Concept:

```text
27B total parameter universe
        |
        v
prompt retrieval
        |
        v
8B-12B candidate set
        |
        v
token routing
        |
        v
3B-6B initial active set
```

This reduces:

- router search cost,
- RAM working set,
- unnecessary prefetch,
- cold-block traffic.

---

# 17. Expert Addressing

Each micro-expert gets an address vector:

```text
expert_embedding[e] in R^d
```

A prompt/token hidden state produces:

```text
query_embedding
```

Retrieve:

```text
TopK(query_embedding dot expert_embedding)
```

Baseline implementations:

1. centroid similarity,
2. matrix dot product,
3. hierarchical clustering.

Later:

- learned low-rank router,
- approximate nearest-neighbor search,
- FAISS,
- locality-sensitive hashing,
- product-quantized routing index.

Start simple.

---

# 18. Building Expert Centroids

Calibration:

1. Run a diverse dataset.
2. For each token/layer, find oracle-important experts.
3. Save hidden states associated with each important expert.
4. Compute a centroid.

```text
centroid_e =
mean(hidden states where expert e is important)
```

Inference:

```text
score_e = cosine(x, centroid_e)
```

Compare router recall against the oracle.

---

# 19. Phase 4 - Token/Layer Router

Input:

```text
current hidden state x_l,t
```

Output:

```text
ranked expert candidates
+
confidence
+
estimated required K
```

The router must be far cheaper than the FFN computation it removes.

Measure:

```text
router latency
router FLOPs
router memory traffic
oracle recall
selected-expert precision
```

---

# 20. Major New Modification - Progressive Quality-Adaptive Activation

This should become one of the central contributions.

Do not simply choose:

```text
TopK = 16
```

Instead:

```text
Pass 1 -> small expert set
Pass 2 -> add more if needed
Pass 3 -> add more if needed
Fallback -> full layer
```

Example:

```text
easy token:
3B active

normal token:
5B active

difficult token:
8B active

very difficult token:
12B active

uncertain token:
full computation
```

The active parameter count becomes a dynamic variable.

---

# 21. Quality / Residual Estimator

The key problem is:

> How do we know whether skipped experts mattered?

Possible confidence signals:

- router entropy,
- score margin between selected/unselected experts,
- cumulative estimated expert contribution,
- hidden-state novelty,
- disagreement between multiple cheap estimators,
- predicted residual norm,
- logit margin,
- layer sensitivity,
- token type,
- prompt difficulty.

Define:

```text
confidence_t,l =
f(router scores,
  residual estimate,
  layer sensitivity,
  hidden-state novelty)
```

If confidence is low:

```text
increase K
```

If confidence remains low:

```text
activate full layer
```

---

# 22. Progressive Residual Recovery

Suppose the router selects set `S`.

Approximate:

```text
y_hat = sum_(e in S) y_e
```

The missing term is:

```text
r = sum_(e not in S) y_e
```

We cannot compute `r` exactly without losing the speed benefit.

The research problem is to cheaply estimate:

```text
||r||
```

or the risk that `r` materially changes the next-token distribution.

Potential approaches:

### Approach A - Router score mass

Estimate omitted contribution from remaining router scores.

### Approach B - Tiny residual predictor

Input:

```text
x
selected experts
selected contribution statistics
```

Output:

```text
predicted residual magnitude
```

### Approach C - Layer-specific threshold

Use calibration to learn:

```text
minimum cumulative contribution required
```

### Approach D - Dual router agreement

Two cheap independent predictors.

If they disagree strongly:

```text
activate more experts
```

This quality guard may be one of the most important additions over the original plan.

---

# 23. Why Progressive Activation Is Better Than Fixed Sparsity

Fixed Top-K assumes every token has equal difficulty.

That is unrealistic.

Example:

```text
";" in generated code
```

may need little computation.

But:

```text
"find the race condition in this lock-free cache"
```

may require much more model capacity.

Therefore the model should spend parameter compute according to token difficulty.

The paper can study:

```text
quality vs active parameter budget
```

rather than one fixed sparse operating point.

---

# 24. Phase 5 - Hierarchical Memory System

Use three principal memory tiers.

## L1 - GPU VRAM

RTX 3060 Laptop, 6 GB.

Store:

- hottest micro-experts,
- current compute blocks,
- selected common layers if profitable,
- CUDA buffers,
- KV/state as required.

## L2 - System RAM

32 GB.

Store:

- full Q4/Q5 27B model if it fits,
- candidate pool,
- warm experts,
- router metadata,
- cache,
- staging buffers.

For Qwen3.8-27B, the whole quantized model may fit in RAM depending on quantization/context overhead.

Therefore SSD should not be on the critical path for every token.

## L3 - NVMe SSD

700-800 GB.

Store:

- complete model,
- alternate precision blocks,
- cold expert blocks,
- profiling metadata,
- optional residual precision data,
- larger stretch models.

---

# 25. Important Deployment Principle

For Qwen3.8-27B:

> The SSD should ideally not be read heavily for every generated token.

The goal should be:

```text
SSD -> RAM during model load / prompt setup / rare cache misses
RAM -> VRAM during generation
```

not:

```text
SSD -> RAM -> GPU for every expert of every token
```

Continuous SSD traffic would destroy latency.

---

# 26. Predictive Prefetch

Use the current token/layer state to predict future parameter requests.

While computing token `t`:

```text
compute token t
+
predict blocks for:
    later layers of token t
    token t+1
```

Then asynchronously:

```text
SSD -> RAM
RAM -> VRAM
```

before the block is demanded.

The goal is to overlap transfer with useful computation.

---

# 27. Prefetch Horizons

Test multiple horizons.

## Same-token cross-layer

Predict experts for:

```text
layer l+1
layer l+2
...
```

## Next-token

Predict experts for:

```text
token t+1
```

## Short speculative horizon

Predict likely expert reuse for:

```text
t+1
t+2
```

Do not assume long-horizon prediction works.

Measure it.

---

# 28. Memory-Aware Expert Score

A normal router asks:

```text
Which expert is important?
```

ParamStream should also ask:

```text
Where is it?
How expensive is it to obtain?
How likely is it to be reused?
```

Possible placement score:

```text
placement_score(e) =
alpha * future_use_probability
+ beta * recent_frequency
+ gamma * task_relevance
+ delta * transfer_cost_saved
- epsilon * memory_cost
```

This score is not allowed to change the mathematical expert importance ranking unless quality-aware rules explicitly allow approximation.

The native/router importance and memory placement policy should be conceptually separated.

---

# 29. GPU Cache

VRAM is the fastest but smallest tier.

Maintain a persistent expert cache.

Good candidates:

- universal-core blocks,
- highly reused blocks,
- blocks predicted for upcoming layers,
- expensive-to-transfer blocks,
- high-frequency coding experts.

Avoid loading a block into VRAM if:

```text
transfer time > expected GPU compute advantage
```

In that case CPU execution may be faster overall.

---

# 30. CPU/GPU Heterogeneous Execution

For every selected block:

```text
if block in VRAM:
    run on GPU

elif block in RAM:
    compare:
        CPU execution cost
        vs RAM->VRAM transfer + GPU compute

    choose cheaper path

elif block only on SSD:
    prefetch asynchronously
    or use fallback scheduling
```

This is different from assuming all selected experts must reach the GPU.

The scheduler should minimize end-to-end token latency.

---

# 31. Cost Model

For block `e`:

```text
T_gpu(e) =
T_transfer_RAM_to_VRAM(e)
+
T_compute_GPU(e)
```

CPU alternative:

```text
T_cpu(e) =
T_compute_CPU(e)
```

If:

```text
T_cpu(e) < T_gpu(e)
```

execute it on CPU.

For blocks already resident in VRAM:

```text
T_transfer = 0
```

so GPU usually wins.

---

# 32. Quantization Strategy

Do not make the paper depend on sub-4-bit quantization.

Main target:

```text
Q4 or better
```

Recommended mixed-precision layout:

```text
sensitive shared tensors : Q6/Q8 or FP16 where necessary
universal core           : Q5/Q6
conditional micro-experts: Q4
cold blocks              : Q4
router metadata          : compact integer/FP16/FP32 as appropriate
```

The exact format should be chosen empirically.

---

# 33. Optional Precision-Adaptive Extension

This should be treated as an extension, not the first milestone.

Possible idea:

```text
base representation
+
precision residuals
```

Example concept:

```text
Q3/Q4 base block
+
additional correction bits
=
higher effective precision
```

Then:

```text
cold expert -> lower precision base
important expert -> base + correction
critical expert -> higher precision
```

However, this adds substantial implementation complexity.

Do not start here.

First prove conditional execution.

---

# 34. Quality Target

Do not claim "zero quality loss."

Quantization and conditional execution are approximations.

Use measurable targets.

Primary research target:

```text
>= 98-99% quality retention
```

relative to the chosen full quantized parent baseline.

Quality retention should be reported benchmark-by-benchmark.

Do not hide regressions behind one average.

---

# 35. Speed Target

Primary deployment target:

```text
>= 10 tokens/second average decode
```

on:

```text
Ryzen 7 5800H
32 GB RAM
RTX 3060 Laptop 6 GB
```

This is a research target, not a guaranteed outcome.

Secondary targets:

```text
time-to-first-token
P50 token latency
P95 token latency
worst-case fallback latency
```

A system that averages 10 tok/s but frequently stalls for seconds is not good enough.

---

# 36. Revised Parameter Budgets

Keep these separate.

## Total knowledge pool

```text
27B
```

## Request candidate pool

Research target:

```text
8B-12B
```

## Initial token active set

Research target:

```text
3B-5B
```

## Average final active set after progressive recovery

Research target:

```text
<= 5B-6B
```

## Hard-token expansion

Possible:

```text
8B
12B
full layer/model
```

The system should report the actual distribution.

---

# 37. Active Parameter Ratio

Define:

```text
APR =
executed parameters per token
/
total parent parameters
```

Example:

```text
5B / 27B = 18.5%
```

Report:

- average APR,
- median APR,
- P90 APR,
- P99 APR,
- APR by task,
- APR by layer.

---

# 38. Candidate Parameter Ratio

Define:

```text
CPR =
candidate parameters per request
/
total parent parameters
```

This measures prompt retrieval efficiency separately from token execution.

---

# 39. New Metric - Memory Traffic Ratio

Define:

```text
MTR =
bytes moved during sparse execution
/
bytes moved by full-model baseline
```

Report separately:

```text
SSD -> RAM
RAM -> VRAM
RAM read traffic
VRAM read traffic
```

This is critical because fewer FLOPs do not guarantee faster inference.

---

# 40. New Metric - Quality-Adjusted Throughput

Possible metric:

```text
QAT =
tokens/sec * quality_retention
```

This should not replace standard metrics, but can help visualize tradeoffs.

Always report raw quality and raw speed too.

---

# 41. New Metric - Recovery Rate

Measure how often progressive execution needs additional passes.

Example:

```text
initial set sufficient: 72%
one expansion        : 21%
two expansions       : 5%
full fallback        : 2%
```

This tells whether adaptive execution is actually useful.

---

# 42. New Metric - Prefetch Accuracy

Measure:

```text
prefetch precision
prefetch recall
VRAM cache hit rate
RAM cache hit rate
wasted bytes prefetched
```

A predictor can have high recall but still waste too much bandwidth.

---

# 43. New Metric - Expert Reuse Distance

Measure:

```text
number of tokens/layers between repeated uses of same block
```

This helps determine:

- cache policy,
- prefetch horizon,
- VRAM residency.

---

# 44. Phase 6 - Physical Parameter Layout

Once useful micro-experts are proven, store them contiguously.

Possible structure:

```text
layer_00/
    core/
    expert_000.bin
    expert_001.bin
    ...

layer_01/
    ...
```

Final GGUF/custom layout may contain:

```text
expert offset
expert size
quantization type
layer id
cluster id
universal-core flag
routing embedding offset
```

Do not invent a custom format before the science works.

---

# 45. Runtime Metadata

Each block may have:

```text
expert_id
layer_id
neuron_range
storage_offset
quantization_type
task statistics
activation frequency
reuse statistics
sensitivity score
router embedding
current tier
cache score
```

Keep metadata compact.

---

# 46. End-to-End Token Algorithm

Pseudo-code:

```text
INPUT:
    token t
    hidden state h
    request candidate pool C

FOR each layer l:

    1. obtain router query from h

    2. rank candidate micro-experts in C_l

    3. choose initial set S_0

    4. ask memory scheduler where blocks are:
        VRAM
        RAM
        SSD

    5. prefetch likely future blocks asynchronously

    6. execute S_0 using GPU/CPU scheduler

    7. estimate omitted contribution / confidence

    8. if confidence insufficient:
           add S_1
           execute incremental blocks

    9. repeat until:
           confidence threshold satisfied
           OR full fallback reached

    10. combine contributions

    11. continue to next layer

OUTPUT:
    next-token logits
```

---

# 47. Prompt Setup Algorithm

Before token generation:

```text
1. encode prompt
2. estimate task/capability profile
3. retrieve candidate block groups
4. include universal core
5. build per-layer candidate sets
6. prefetch hot candidate blocks into RAM
7. prefill GPU cache
8. begin generation
```

Prompt analysis cost must be amortized across the generation.

---

# 48. Calibration Dataset

Use diverse categories.

## Coding

- Python
- C/C++
- Java
- JavaScript/TypeScript
- Rust
- SQL
- shell
- debugging
- algorithms
- web development
- APIs
- concurrency
- security
- refactoring
- code explanation

## Reasoning

- arithmetic
- symbolic reasoning
- logic
- planning
- multi-step tasks

## General

- instruction following
- summarization
- factual QA
- writing
- long-context retrieval

Do not train/calibrate only on coding if the model is expected to preserve general capability.

---

# 49. Coding Evaluation

Recommended benchmark families:

```text
HumanEval+
MBPP+
LiveCodeBench
MultiPL-E
repository-level coding tasks
bug-fixing tasks
```

If resources permit:

```text
SWE-bench style evaluation
terminal/agentic coding tasks
```

Also include practical local tests:

- create a small project,
- modify multiple files,
- debug failing tests,
- write unit tests,
- refactor code,
- reason over repository context.

---

# 50. General Evaluation

Measure:

- next-token agreement,
- logit cosine similarity,
- KL divergence,
- top-k overlap,
- perplexity where useful,
- reasoning benchmarks,
- instruction-following tasks.

The sparse model should be compared against the **same quantized parent model**.

Do not compare sparse Q4 against parent BF16 and attribute all difference to routing.

---

# 51. Generation Divergence

Measure:

```text
tokens until first divergence
```

But do not rely on exact text match alone.

Two generations can differ lexically while remaining equally correct.

Therefore use:

- executable correctness,
- unit tests,
- benchmark score,
- semantic evaluation,
- task success.

---

# 52. Baselines

The paper should compare against several categories.

## Dense baseline

```text
full Qwen3.8-27B at same quantization
```

## Static sparsity

- magnitude pruning,
- structured pruning,
- fixed block removal.

## Dynamic sparsity

- activation-magnitude selection,
- fixed Top-K routing,
- token-only router.

## Prompt-only retrieval

No token router.

## Token-only routing

No prompt candidate pool.

## Memory baselines

- normal CPU/GPU offload,
- LRU cache,
- LFU cache,
- no prefetch,
- simple next-use predictor.

## Native MoE comparison

If practical:

- Qwen3-Coder-30B-A3B or similar.

This shows the cost/benefit of converting a dense model versus using a native MoE.

---

# 53. Ablation Study

A strong paper should ablate:

```text
micro-expert size
contiguous vs clustered neurons
prompt retrieval on/off
token router on/off
adaptive K on/off
quality guard on/off
universal core size
prefetch on/off
VRAM cache on/off
CPU execution on/off
memory-aware scheduling on/off
mixed precision on/off
```

Also compare:

```text
centroid router
vs
learned router
```

---

# 54. Key Research Hypotheses

## H1 - Oracle sparsity

A substantial fraction of dense FFN computation can be omitted per token with low output distortion.

## H2 - Predictability

Important blocks can be predicted from the incoming hidden state without computing all blocks first.

## H3 - Task specialization

Expert/block usage differs meaningfully across tasks and prompts.

## H4 - Adaptive compute

Dynamic expert budgets preserve quality better than a fixed sparse budget at equal average compute.

## H5 - Hierarchical retrieval

Prompt-level candidate narrowing reduces routing and memory cost without significant quality loss.

## H6 - Memory locality

Selected parameter blocks exhibit enough reuse/predictability to benefit from persistent RAM/VRAM caching.

## H7 - Wall-clock benefit

Conditional execution reduces actual latency, not merely theoretical FLOPs.

---

# 55. Falsification Criteria

Stop or pivot if:

1. Oracle experiments require >60-70% of FFN blocks most of the time.
2. Routing recall is too low even with expensive predictors.
3. Sparse kernels are slower than dense matrix multiplication.
4. Memory gathering destroys the theoretical compute savings.
5. Progressive recovery activates almost the full model on most tokens.
6. Quality falls sharply before meaningful compute savings.
7. Prompt-level candidate pools fail to narrow parameter usage.
8. Expert reuse is too weak for useful caching.
9. Router + scheduler overhead dominates token latency.

The project should be allowed to fail scientifically.

---

# 56. Continue Criteria

Continue aggressively if:

1. <=30-40% oracle FFN blocks preserve output well.
2. Router recall is close to oracle.
3. Different tasks use measurably different blocks.
4. Adaptive budgets reduce average active parameters substantially.
5. Full-generation quality remains close to baseline.
6. Sparse execution improves real tokens/sec.
7. RAM/VRAM cache hit rates are high.
8. SSD traffic stays low after prompt setup.

---

# 57. Novelty Boundary

Do not claim novelty for any of these alone:

```text
dense contextual sparsity
Top-K FFN routing
MoE inference
SSD offloading
CPU offloading
expert caching
predictive prefetch
quantization
neuron clustering
adaptive K
```

Each area has prior research.

The potential novelty is in the integrated method:

```text
post-training dense-to-micro-expert conversion
+
capability-addressable parameter retrieval
+
hierarchical prompt/token routing
+
quality-adaptive progressive activation
+
memory-tier-aware scheduling
+
predictive parameter placement
```

The final novelty claim must be written only after a systematic literature review.

---

# 58. Relation to Existing Research Categories

The literature review should explicitly investigate:

## Contextual sparsity in dense LLMs

Prior work already studies prediction of input-dependent sparse computation.

Therefore ParamStream must not claim contextual sparsity itself is new.

## Native MoE offloading

Prior work already studies:

- expert offload,
- caching,
- prefetch,
- SSD-backed MoE execution.

Therefore SSD streaming itself is not sufficient novelty.

## Dense-to-MoE conversion

Search for:

- post-training dense-to-sparse conversion,
- FFN neuron grouping,
- structural sparsification,
- dynamic sparse inference.

## Adaptive computation

Search for:

- early exit,
- dynamic depth,
- adaptive width,
- token-level conditional computation,
- uncertainty-aware compute.

## Memory-aware inference

Search for:

- heterogeneous CPU/GPU inference,
- tiered memory,
- unified cache policies,
- parameter paging.

The final paper should clearly state which mechanism is genuinely new.

---

# 59. Why This Is Better Than Only Edge-Style MoE Streaming

A native MoE already contains experts.

The research question is mostly:

```text
how do we move them efficiently?
```

ParamStream asks a deeper question:

```text
can an existing dense model
be transformed into
a retrievable conditional model
without retraining it from scratch?
```

Then it additionally solves the systems problem of executing those blocks efficiently.

That creates both:

- an ML/sparsity contribution,
- and a systems contribution.

---

# 60. Why This Is Better Than Only ParamRAG

Original ParamRAG focused strongly on:

```text
which parameters should be active?
```

The updated project also asks:

```text
where should those parameters live?
when should they be moved?
which device should execute them?
how much additional compute is needed to protect quality?
```

This makes the final system more realistic.

---

# 61. Why Quality-Adaptive Recovery Matters

A router error should not automatically become an output error.

The model should be able to spend more computation when uncertain.

This creates a spectrum:

```text
fast path
    |
    v
small active set
    |
    v
confidence check
    |
    +---- good -> continue
    |
    +---- bad -> expand
                    |
                    v
                 expand again
                    |
                    v
                 full fallback
```

The system trades compute for confidence.

---

# 62. Potential Paper Contributions

If successful, the paper may claim contributions in forms such as:

## Contribution 1

A post-training method that converts dense FFNs into exact micro-expert decompositions before sparsification.

## Contribution 2

A hierarchical prompt/token parameter retrieval mechanism for building request-specific conditional subnetworks.

## Contribution 3

A quality-adaptive progressive execution algorithm that expands the active parameter budget only when estimated omitted contribution is significant.

## Contribution 4

A memory-tier-aware scheduler that jointly considers expert importance, reuse probability, transfer cost, and CPU/GPU execution cost.

## Contribution 5

A consumer-laptop demonstration of a large coding model with a much smaller average active working set.

The final wording must depend on measured results.

---

# 63. Intended Paper Positioning

Potential positioning:

```text
Machine Learning Systems
Efficient LLM Inference
Edge AI
Conditional Computation
Sparse Neural Networks
LLM Systems
Heterogeneous Computing
```

Possible venue families to research later:

- IEEE journals/conferences related to AI systems, computers, edge computing, parallel/distributed systems.
- Springer journals/conferences related to neural computing, AI systems, machine learning, and distributed/edge systems.

Do not choose a venue before the experimental result is clear.

Different venues value:

- algorithmic novelty,
- systems implementation,
- hardware evaluation,
- or edge deployment differently.

---

# 64. Publication Standard

A strong submission should contain:

1. a clear hypothesis,
2. formal method,
3. exact baselines,
4. ablation studies,
5. multiple datasets,
6. multiple models if possible,
7. quality results,
8. speed results,
9. memory results,
10. energy results if possible,
11. failure cases,
12. statistical reporting where appropriate,
13. reproducible code.

Do not build the paper around one impressive demo video.

---

# 65. Experimental Reproducibility

Record:

```text
model commit/hash
quantization format
runtime commit
compiler
CUDA version
GPU driver
CPU thread count
RAM configuration
SSD model
context length
batch size
prompt length
generation length
temperature
seed
```

Benchmark tokens/sec using fixed settings.

---

# 66. Laptop Memory Budget

Illustrative operating budget:

```text
32 GB system RAM

Windows + background     : ~4-6 GB
Model weights            : ~18-22 GB
Runtime/cache/buffers    : remaining budget
```

This is tight.

Therefore:

- keep context length controlled during experiments,
- avoid unnecessary duplicate model copies,
- memory-map where appropriate,
- measure real peak working set.

The exact numbers depend on the selected quantization/runtime.

---

# 67. VRAM Budget

Illustrative:

```text
6 GB VRAM

CUDA/runtime             : some overhead
KV/cache                  : context-dependent
hot parameter cache      : remaining space
temporary GEMM buffers   : required
```

Do not assume all 6 GB can hold weights.

Measure actual free VRAM.

---

# 68. SSD Role

The SSD capacity is not the main limit.

700-800 GB is enough for:

- multiple quantization variants,
- calibration artifacts,
- expert files,
- alternate precision blocks,
- larger stretch models.

SSD bandwidth/latency matters only when it becomes part of the active token path.

The scheduler should minimize those reads.

---

# 69. Optional Larger-Model Extension

After the 27B dense research is proven, apply the memory scheduler to a stronger native MoE coding model.

Example architecture:

```text
large coding MoE
80B total
~3B active
```

Possible advantages:

- high total capacity,
- relatively low active compute,
- strong coding specialization.

But if the model does not fit RAM:

```text
SSD -> RAM expert streaming
```

becomes mandatory.

This extension can demonstrate that ParamStream generalizes beyond dense-to-conditional models.

---

# 70. Two-Paper Possibility

If results are strong, the work may naturally split.

## Paper A - Algorithmic

Dense-to-conditional conversion:

```text
micro-experts
hierarchical retrieval
quality-adaptive activation
```

## Paper B - Systems

Hierarchical execution:

```text
RAM/VRAM/SSD placement
predictive prefetch
heterogeneous CPU/GPU scheduling
```

Do not force everything into one paper if it becomes too broad.

For the first project, however, implement an integrated prototype.

---

# 71. Development Stack

## Research prototype

```text
Python 3.11
PyTorch
Transformers
Accelerate
Safetensors
NumPy
Pandas
scikit-learn
matplotlib
```

Optional later:

```text
FAISS
Triton
CUDA
C++
```

## Runtime

Eventually:

```text
llama.cpp
GGML
GGUF
custom sparse block kernels
custom scheduler
```

Do not start by modifying llama.cpp.

Prove the scientific assumptions first.

---

# 72. Suggested Repository Structure

```text
paramstream/
|
+-- README.md
+-- update.md
+-- requirements.txt
|
+-- configs/
|   +-- qwen4b.yaml
|   +-- qwen8b.yaml
|   +-- qwen27b.yaml
|
+-- src/
|   +-- model_utils.py
|   +-- ffns.py
|   +-- micro_experts.py
|   +-- oracle.py
|   +-- clustering.py
|   +-- prompt_router.py
|   +-- token_router.py
|   +-- quality_guard.py
|   +-- scheduler.py
|   +-- prefetch.py
|   +-- cache.py
|   +-- metrics.py
|
+-- experiments/
|   +-- 01_exact_decomposition.py
|   +-- 02_oracle_one_layer.py
|   +-- 03_oracle_all_layers.py
|   +-- 04_task_specialization.py
|   +-- 05_cluster_neurons.py
|   +-- 06_centroid_router.py
|   +-- 07_prompt_retrieval.py
|   +-- 08_adaptive_k.py
|   +-- 09_quality_guard.py
|   +-- 10_memory_trace.py
|   +-- 11_cache_simulation.py
|   +-- 12_prefetch_simulation.py
|   +-- 13_full_generation.py
|   +-- 14_wall_clock_runtime.py
|
+-- runtime/
|   +-- cpp/
|   +-- cuda/
|   +-- gguf/
|   +-- scheduler/
|
+-- evaluation/
|   +-- logits.py
|   +-- generation.py
|   +-- coding.py
|   +-- reasoning.py
|   +-- efficiency.py
|   +-- memory.py
|
+-- results/
|   +-- csv/
|   +-- plots/
|   +-- traces/
|   +-- logs/
|
+-- notebooks/
    +-- exploration.ipynb
    +-- paper_plots.ipynb
```

---

# 73. Updated Milestones

## Milestone 0 - Exactness

Prove:

```text
reordered dense FFN == original dense FFN
```

and:

```text
all micro-experts enabled == dense FFN
```

Deliverable:

```text
exactness_report.csv
```

## Milestone 1 - Oracle sparsity

Deliverables:

```text
oracle_one_layer.csv
oracle_all_layers.csv
quality_vs_retention.png
```

## Milestone 2 - Task specialization

Deliverables:

```text
expert_task_statistics.parquet
task_expert_heatmap.png
```

## Milestone 3 - Neuron clustering

Deliverables:

```text
expert_assignments.npy
cluster_statistics.csv
```

## Milestone 4 - Token router

Deliverables:

```text
router_vs_oracle.csv
router_recall.png
```

## Milestone 5 - Prompt retriever

Deliverables:

```text
candidate_pool_results.csv
```

## Milestone 6 - Progressive activation

Deliverables:

```text
adaptive_compute_results.csv
fallback_distribution.csv
quality_vs_APR.png
```

## Milestone 7 - Memory trace simulation

Deliverables:

```text
expert_trace.parquet
cache_simulation.csv
```

## Milestone 8 - Predictive prefetch

Deliverables:

```text
prefetch_accuracy.csv
traffic_reduction.csv
```

## Milestone 9 - CPU/GPU scheduler

Deliverables:

```text
placement_benchmark.csv
```

## Milestone 10 - 27B end-to-end

Deliverables:

```text
qwen38_27b_benchmarks.csv
quality_results.csv
speed_results.csv
memory_results.csv
```

## Milestone 11 - Optimized local runtime

Deliverables:

```text
custom runtime
reproducible benchmark script
```

## Milestone 12 - Paper

Deliverables:

```text
paper.pdf
artifact repository
reproduction guide
```

---

# 74. Correct Research Order

```text
1. exact decomposition
2. oracle sparsity
3. task specialization
4. predict important blocks
5. prompt candidate retrieval
6. adaptive quality recovery
7. verify full generation
8. measure theoretical memory traces
9. design cache/prefetch
10. implement CPU/GPU scheduler
11. optimize sparse kernels
12. scale to 27B
13. benchmark
14. write paper
```

Do not reverse this.

---

# 75. First Immediate Experiment

The first scientific experiment should still be simple:

> For one real Qwen FFN and one real token, how many neuron blocks can be removed by an oracle before final logits materially change?

Do this before:

- SSD work,
- CUDA work,
- custom GGUF,
- large-model download,
- trained router,
- paper writing.

If useful sparsity does not exist, the rest of the architecture does not matter.

---

# 76. First Week Updated Plan

## Day 1

- initialize repository,
- load small Qwen model,
- inspect FFN shapes,
- verify baseline.

## Day 2

- implement exact block decomposition,
- verify 100% blocks reproduce baseline.

## Day 3

- implement oracle contribution score,
- sweep retention levels.

## Day 4

- repeat across layers and prompts,
- save layer sensitivity.

## Day 5

- test coding/math/general prompts,
- calculate task-specific expert usage.

## Day 6

- test multiple block sizes,
- compare contiguous vs simple clustering.

## Day 7

Decision:

```text
continue
pivot
stop
```

based on evidence.

---

# 77. Performance Modeling Before Kernel Work

Before writing custom CUDA, estimate theoretical gains.

For each layer measure:

```text
dense bytes read
dense FLOPs
sparse bytes read
sparse FLOPs
router overhead
gather overhead
```

A sparse method can fail because:

```text
less arithmetic
but
more irregular memory traffic
```

The paper should explicitly measure this.

---

# 78. Sparse Kernel Requirement

If the selected neuron blocks are irregular, ordinary dense GEMM may no longer be efficient.

Potential approaches later:

- block-sparse GEMM,
- grouped GEMM,
- expert batching,
- fused gather + GEMM,
- reordered contiguous blocks,
- Triton kernels,
- custom CUDA kernels.

This is why physical neuron reordering is important.

---

# 79. Token-Level Batching of Experts

Instead of launching one kernel per tiny expert:

```text
collect selected experts
group contiguous blocks
execute them together
```

This reduces launch overhead.

Possible block representation:

```text
selected ranges:
[0:128]
[256:384]
[1024:1280]
...
```

Merge adjacent ranges when possible.

---

# 80. Router Overhead Budget

Set a hard rule:

```text
router + quality guard
must consume only a small fraction
of saved FFN time
```

For example, report:

```text
router_time / token_time
```

If routing takes 30% of decode time, the method may not be worthwhile.

---

# 81. Prompt Router Amortization

Prompt retrieval may be moderately expensive because it runs once per request.

Its cost can be amortized across:

```text
100
500
1000
```

generated tokens.

Report:

```text
setup latency
+
steady-state decode latency
```

separately.

---

# 82. Prefill vs Decode

Treat them separately.

## Prefill

Processes many prompt tokens.

Dense/batched computation may be more efficient.

It may be better to use less sparsity.

## Decode

One token at a time.

Memory bandwidth becomes dominant.

This is where conditional parameter execution may help most.

The runtime can use different policies for prefill and decode.

---

# 83. Context Length

Long context consumes KV/cache memory.

For the laptop, benchmark multiple context lengths:

```text
2K
4K
8K
16K
```

if the model/runtime permits.

Report speed and memory at each.

Do not quote one tok/s number without context length.

---

# 84. Coding Assistant Product Target

The eventual local assistant should support:

- code generation,
- code explanation,
- debugging,
- refactoring,
- unit-test generation,
- repository Q&A,
- patch generation,
- terminal-style instructions.

Possible final integration:

```text
VS Code extension
or
local OpenAI-compatible API
```

This product layer should come after the research runtime.

---

# 85. What Not to Do Yet

Do not spend early time on:

- GUI,
- website,
- polished extension,
- custom file format,
- 80B model,
- multiple GPUs,
- mobile deployment,
- perfect documentation,
- paper formatting,
- journal selection,
- patent filing.

First prove the hypothesis.

---

# 86. Paper Figures to Aim For

Potential figures:

## Figure 1

System architecture:

```text
Prompt -> retrieval -> micro-router -> quality guard
                         |
                         v
                   memory scheduler
                  /      |       \
                SSD     RAM     VRAM
```

## Figure 2

Quality vs active parameter ratio.

## Figure 3

Tokens/sec vs active parameter ratio.

## Figure 4

Layer-specific sparsity.

## Figure 5

Task/expert activation heatmap.

## Figure 6

RAM/VRAM cache hit rates.

## Figure 7

Memory traffic per token.

## Figure 8

Fallback distribution.

---

# 87. Main Tables

## Table 1

Model/quantization/hardware.

## Table 2

Quality benchmark comparison.

## Table 3

Throughput and memory.

## Table 4

Ablations.

## Table 5

Router accuracy.

## Table 6

Caching/prefetch metrics.

## Table 7

Task-specific active parameter counts.

---

# 88. Expected Strong Result

A strong conceptual result could look like:

```text
Parent:
Qwen3.8-27B Q4/Q5
27B nominal active
baseline quality = 100%

ParamStream:
27B full knowledge pool
8B-12B request candidate pool
~4B-6B average executed/token
>=98-99% quality retention target
>=10 tok/s local decode target
<=32 GB RAM
<=6 GB VRAM
fully offline
```

These are target outcomes, not promises.

---

# 89. Expected Moderate Result

Even if the ideal target is not reached, the paper could still be valuable if it shows:

```text
27B parent
-> 8B-10B active
with
small quality degradation
and
meaningful wall-clock improvement
```

Negative findings about hardware inefficiency can also be publishable if evaluated rigorously.

---

# 90. Possible Failure But Useful Research Outcome

Suppose:

```text
oracle sparsity is high
but
real sparse kernels are slower
```

That still provides an important result:

> Algorithmic contextual sparsity does not automatically translate into speed on commodity GPU/CPU systems because irregular parameter access dominates.

The paper can then focus on layout/scheduling.

---

# 91. Stretch Innovation - Joint Router and Memory Scheduler

A later advanced version may jointly optimize:

```text
importance
+
latency
+
memory tier
+
confidence
```

Objective:

```text
minimize token latency

subject to:

quality risk <= threshold
RAM <= limit
VRAM <= limit
```

This can be formalized as constrained optimization.

Do not begin with this complexity.

---

# 92. Stretch Innovation - Learned Cost-Aware Routing

The router could learn:

```text
expert usefulness
and
expert acquisition cost
```

But quality must remain primary.

A cheap-but-wrong block should not replace an important block solely because it is already cached.

Instead memory cost can influence:

- prefetch,
- retention,
- expansion order among near-equivalent blocks.

---

# 93. Stretch Innovation - Structured Expert Families

Micro-experts may be organized hierarchically:

```text
universal
 |
 +-- coding
 |    +-- Python
 |    +-- JavaScript
 |    +-- C++
 |
 +-- reasoning
 |
 +-- language
```

The hierarchy may remain latent rather than human-labeled.

Prompt retrieval chooses families.

Token routing chooses blocks.

---

# 94. Stretch Innovation - Cross-Layer Predictability

Investigate whether expert usage in layer `l` predicts usage in layer `l+1`.

If yes:

```text
current layer routing
->
next layer prefetch
```

This may be easier than predicting the next token.

---

# 95. Stretch Innovation - Temporal Expert Reuse

Investigate whether:

```text
expert(t)
```

predicts:

```text
expert(t+1)
```

especially during code generation where local syntax/semantics persist across tokens.

This can improve VRAM cache efficiency.

---

# 96. Research Logging

Every experiment should save machine-readable results.

Minimum columns:

```text
model
layer
token_id
task
block_size
retention
selected_blocks
oracle_blocks
router_recall
APR
quality_metric
latency_ms
bytes_RAM
bytes_VRAM
bytes_SSD
```

Do not rely on handwritten observations.

---

# 97. Statistical Practice

For benchmark comparisons:

- use enough samples,
- report confidence intervals where appropriate,
- repeat timing runs,
- discard warmup separately,
- report mean/median/P95,
- control generation settings.

A difference of 0.2 tok/s may be noise.

---

# 98. Research Ethics / Claims

Avoid claims such as:

```text
no quality loss
same intelligence
27B model running as 5B
```

unless rigorously justified.

Prefer:

```text
X% benchmark retention
Y% average active parameter ratio
Z tok/s
```

Concrete measurements are stronger than marketing language.

---

# 99. Core Deliverable Definition

The final project should include:

```text
1. post-training micro-expert converter
2. oracle analysis toolkit
3. prompt capability retriever
4. token-level router
5. adaptive quality guard
6. memory scheduler
7. RAM/VRAM expert cache
8. predictive prefetch
9. local inference runtime
10. benchmark suite
11. reproducibility scripts
12. research paper
```

---

# 100. Final Vision

The final system should behave like:

```text
                     Large dense coding model
                              |
                              v
                 exact micro-expert decomposition
                              |
                              v
                   capability parameter index
                              |
                 +------------+------------+
                 |                         |
                 v                         v
         universal shared core      prompt retrieval
                 |                         |
                 +------------+------------+
                              |
                              v
                     token/layer router
                              |
                              v
                    small initial active set
                              |
                              v
                     quality / residual guard
                              |
                  +-----------+-----------+
                  |                       |
                safe                  uncertain
                  |                       |
                  v                       v
               execute              activate more
                  |                       |
                  +-----------+-----------+
                              |
                              v
                    heterogeneous scheduler
                     /         |          \
                    /          |           \
                  SSD         RAM          VRAM
                   |           |            |
                   +-----------+------------+
                               |
                               v
                           CPU / GPU
                               |
                               v
                            output
```

The intended scientific result is not:

> "We compressed a 27B model."

It is:

> "We transformed a dense 27B coding model into a dynamically assembled, quality-adaptive conditional model whose active computation and memory footprint are determined by the current problem."

And the intended practical result is:

> "A strong coding assistant can run completely offline on a 32 GB RAM / 6 GB VRAM consumer laptop with useful interactive throughput."

---

# 101. Final Recommended Research Statement

A concise final research statement:

> **ParamStream investigates whether a pretrained dense LLM can be converted post-training into capability-addressable micro-experts and executed through hierarchical prompt/token retrieval, progressive quality-aware activation, and predictive SSD/RAM/VRAM scheduling. The objective is to preserve near-parent-model coding quality while substantially reducing active parameters, memory traffic, and latency on memory-constrained consumer hardware.**

---

# 102. Immediate Next Action

Do not start with the 27B model.

Do this first:

```text
Small Qwen model
    |
    v
one FFN layer
    |
    v
exact micro-expert decomposition
    |
    v
oracle block ranking
    |
    v
100 / 70 / 50 / 40 / 30 / 20 / 10% retention
    |
    v
compare final logits
```

The first decision gate is:

> **Does useful fine-grained oracle sparsity actually exist?**

If yes, continue to routing.

If no, pivot before investing in the memory runtime.

---

# 103. One-Sentence Summary

**ParamStream = ParamRAG's fine-grained parameter retrieval + quality-adaptive progressive execution + predictive hierarchical SSD/RAM/VRAM placement + heterogeneous CPU/GPU scheduling, with Qwen3.8-27B as the main dense research target and >=10 tok/s fully offline laptop inference as the deployment goal.**
