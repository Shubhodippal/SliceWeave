# SliceWeave (ParamStream)

Fine-Grained Retrieval-Augmented Parameter Activation for Dense Language Models —
restructured to the **updated blueprint** (`update.md`): quality-adaptive
parameter retrieval, conditional execution, and hierarchical memory streaming.

The blueprint's working titles "ParamRAG" / "ParamStream" refer to the same
method; the project codename is **SliceWeave**.

## The six research components (update.md §8)

1. **Exact dense-to-micro-expert decomposition** (`src/micro_experts.py`)
2. **Capability-aware parameter retrieval** — prompt level (`src/prompt_router.py`)
3. **Token/layer conditional execution** (`src/token_router.py`)
4. **Quality-adaptive progressive activation** (`src/quality_guard.py`)
5. **Predictive hierarchical memory placement** (`src/cache.py`, `src/prefetch.py`)
6. **Heterogeneous CPU/GPU execution** (`src/scheduler.py`)

## Research order (§74 — do not reverse)

1. exact decomposition (`experiments/00_exact_decomposition.py`)
2. oracle sparsity (`01`, `02`)
3. task specialization (`03`), clustering (`04`)
4. predict important blocks (`05`)
5. prompt candidate retrieval (`06`)
6. adaptive quality recovery (`07`, `08`)
7. verify full generation (`12`)
8. memory traces / cache (`09`), prefetch (`10`), scheduler (`11`)
9. scale to 9B/27B (`14`), wall-clock (`13`), paper

See `experiments/README.md` for the full milestone → script → deliverable table.

## Current status: Milestones 0-2 — exactness + oracle sparsity

The first scripts are scientific falsification tools, NOT acceleration.
They answer: *how much of one real FFN can be removed before the final model
output changes significantly?* The decision gate (§14): 20-30% experts keeping
quality → continue; >60-70% → weak signal.

## Setup

This machine: RTX 3060 Laptop (6 GB VRAM) + 32 GB system RAM.
Use Python 3.13 (regular build, **not** the free-threaded 3.13t).

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Validate everything (CUDA, model loading, exactness, oracle, router, guard,
cache/prefetch/scheduler) without any model download:

```powershell
python smoke_test.py
```

Run the Milestone 0 + 1 experiments (downloads a small Qwen model on first run):

```powershell
python experiments/00_exact_decomposition.py
python experiments/01_oracle_ffn.py --model Qwen/Qwen3-4B --prompts 8
```

For the primary target (Qwen3.8-27B) the laptop needs the 4-bit fallback or
Kaggle — see `notebooks/kaggle_27b.ipynb` and `experiments/14_scale_model.py`.

## Layout (§72)

- `src/` — library code: `ffns`, `micro_experts`, `hooks`, `oracle`, `clustering`,
  `token_router`, `prompt_router`, `quality_guard`, `cache`, `prefetch`,
  `scheduler`, `metrics`, `model_utils`, `datasets`
- `experiments/` — numbered drivers matching the new milestones (00-14)
- `evaluation/` — next-token agreement, generation agreement, efficiency/APR
- `runtime/` — stub until the science is proven (§85: don't start here)
- `configs/` — per-model YAML configs
- `results/csv|plots|traces|logs` — experiment outputs (§96 machine-readable)
