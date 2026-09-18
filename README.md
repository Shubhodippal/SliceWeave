# ParamRAG

Fine-Grained Retrieval-Augmented Parameter Activation for Dense Language Models.

This repository implements the research plan from `ParamRAG_Detailed_Research_Blueprint.md`.

## Research order

1. Prove sparsity exists (`experiments/01_oracle_ffn.py`, `02_oracle_all_layers.py`)
2. Prove experts can be predicted (`experiments/05_centroid_router.py`)
3. Prove routing generalizes
4. Prove quality survives generation
5. Prove computational savings
6. Optimize runtime
7. Test 27B
8. Final benchmarks
9. Write paper

## Current status: Milestone 1 — Oracle FFN Sparsity Test

The first scripts are scientific falsification tools, NOT acceleration.
They answer: *How much of one real FFN can be removed before the final model
output changes significantly?*

## Setup

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Check GPU:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Run the first experiment (downloads a small Qwen model on first run):

```powershell
python experiments/01_oracle_ffn.py --model Qwen/Qwen3-4B --prompts 8
```

## Layout

- `src/` — library code (hooks, micro-expert partitioning, oracle, metrics, clustering, routing)
- `experiments/` — numbered experiment drivers matching the blueprint milestones
- `evaluation/` — next-token agreement, generation agreement, benchmarks
- `configs/` — per-model YAML configs
- `results/csv|plots|logs` — experiment outputs
