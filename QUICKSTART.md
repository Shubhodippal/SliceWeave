# SliceWeave — Quick start on the new machine

Machine facts: RTX 3060 Laptop **6 GB VRAM**, **32 GB system RAM** (verified),
Python **3.13** installed (the default `py` launcher points to the free-threaded
`3.13t` build — always use `py -3.13`).

## 1. Set up

```powershell
py -3.13 -m venv .venv          # regular build, NOT 3.13t
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

CPU-only fallback for torch: `pip install torch` (no index-url).

## 2. Verify environment

```powershell
python smoke_test.py            # full pipeline check, no model download
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `torch 2.14.0+cu126`, `cuda=True`, device `NVIDIA GeForce RTX 3060 Laptop GPU`.
The smoke test now also checks Milestone-0 exactness, router/guard, and the
cache/prefetch/scheduler simulations.

## 3. Model stages (update.md §6)

| Stage | Model | Where | Why |
|---|---|---|---|
| 1 | Qwen3-4B bf16 (~8 GiB) | local (VRAM + RAM offload) | verify code, hooks, metrics |
| 2 | Qwen ~9B | local 4-bit or Kaggle | stress-test scaling |
| 3 | **Qwen3.8-27B** | **Kaggle** (bf16) or local 4-bit | final scientific evaluation |

The loader (`src/model_utils.py`) builds a `max_memory` plan that fills GPU
VRAM first, then spills into capped system RAM (§24) — so Stage 1 runs locally
even though 8 GiB weights exceed 6 GiB VRAM.

## 4. Run Milestone 0 + 1

```powershell
python experiments/00_exact_decomposition.py       # exactness gate (§8.1)
python experiments/01_oracle_ffn.py --model Qwen/Qwen3-4B --layers 3 --prompts 8
```

- First run downloads the model (~8 GB for 4B bf16) into HF cache
  (`HF_HOME` can point to `E:\` — plenty of space there).
- Outputs: `results/csv/exactness_report.csv`, `results/csv/oracle_ffn.csv`
  + console summary table.

## 5. Run Milestone 1 (all-layer sweep) and beyond

```powershell
python experiments/02_oracle_all_layers.py --model Qwen/Qwen3-4B --fractions 0.5 0.3 0.2 0.1
python experiments/03_task_specialization.py --layers 4    # §15 specialization
python experiments/04_cluster_neurons.py                   # §11 clustering+reorder
python experiments/05_centroid_router.py --layers 3        # §17-§19 router
python experiments/06_prompt_retrieval.py                  # §16 prompt pool
python experiments/07_adaptive_k.py                        # §20 progressive K
python experiments/08_quality_guard.py                     # quality vs APR sweep
python experiments/09_memory_trace.py                      # §24 cache sim
python experiments/10_prefetch_simulation.py               # §26-§27 prefetch
python experiments/11_placement_benchmark.py               # §28-§31 scheduler
python experiments/12_full_generation.py                   # §51 generation
python experiments/13_wall_clock_runtime.py                # §7 H7 wall clock
```

## 6. Milestone 10 — the 27B target (Qwen3.8-27B)

The 27B is the **primary research target** (update.md §5). Two supported paths:

1. **Kaggle (recommended)**: open `notebooks/kaggle_27b.ipynb` on Kaggle with a
   GPU accelerator (2×T4 or P100). It loads the model bf16 across VRAM + RAM,
   runs the same oracle sweep, and saves CSV/PNG outputs you download from the
   Output panel.
2. **Local 4-bit fallback**: on the 6 GB 3060, load the 27B with
   `load_in_4bit=True` (needs `bitsandbytes`), or run
   `python experiments/14_scale_model.py --model <id> --load-4bit`.

## 7. Decision gates (§14, §55, §56)

- 20–30% experts keeping quality → continue
- 30–50% → investigate
- >60–70% → weak signal, reconsider
- Router recall far below oracle (Milestone 4) → better predictor, not more K
- Sparse wall-clock slower than dense (Milestone 13) → attack layout/scheduling (§90)
