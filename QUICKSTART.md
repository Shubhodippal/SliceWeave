# SliceWeave — Quick start on the new machine

## 1. Clone and set up

```powershell
git clone <repo-url>
cd SliceWeave
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # NVIDIA GPU machine
pip install -r requirements.txt
```

CPU-only fallback for torch: `pip install torch` (no index-url).

## 2. Verify environment

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. Run Milestone 1 (one-layer oracle test)

```powershell
python experiments/01_oracle_ffn.py --model Qwen/Qwen3-4B --layers 3 --prompts 8
```

- First run downloads the model (~8 GB for 4B bf16) into HF cache.
- Output: `results/csv/oracle_ffn.csv` + a console summary table.

## 4. Run Milestone 2 (all-layer sweep)

```powershell
python experiments/02_oracle_all_layers.py --model Qwen/Qwen3-4B --fractions 0.5 0.3 0.2 0.1
```

- Output: `results/csv/layer_sensitivity.csv`, `results/plots/layer_sensitivity.png`

## 5. Decision gate (Blueprint §11)

- 20–30% experts keeping quality → continue
- 30–50% → investigate
- >60–70% → weak signal, reconsider
