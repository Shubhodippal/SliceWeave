"""Experiment 03 — Task Specialization (§15, Milestone 2).

Do different task categories activate different micro-experts? For each
category, measure P(expert active | task) and expert task-entropy (§31):
low-entropy experts are specialized; high-entropy ones form the universal core
(§15/§32).

Usage:
    python experiments/03_task_specialization.py --model Qwen/Qwen3-4B --layers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clustering import task_conditional_stats  # noqa: E402
from src.datasets import CATEGORIES  # noqa: E402
from src.hooks import FFNIntermediateCapture  # noqa: E402
from src.micro_experts import contiguous_partition  # noqa: E402
from src.model_utils import describe_model, load_model_and_tokenizer  # noqa: E402


def collect_z(model, tokenizer, prompt: str, layer_key: str) -> np.ndarray:
    """[n_tokens, inter] intermediate vectors for every prompt token."""
    cap = FFNIntermediateCapture()
    cap.attach(model, layer_key)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**enc)
    cap.remove()
    z = cap.captures[f"{layer_key}.down_proj"][0]  # [seq, inter]
    return z.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="activation-quantile threshold defining 'active'")
    ap.add_argument("--out-csv", default="results/csv/expert_task_statistics.csv")
    ap.add_argument("--out-json", default="results/csv/task_specialization.json")
    ap.add_argument("--out-plot", default="results/plots/task_expert_heatmap.png")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    layer_keys = [f"model.layers.{i}"
                  for i in np.linspace(2, info["layers"] - 3, args.layers).astype(int)]
    print("Probing layers:", layer_keys)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "task", "expert_id", "p_active"])
        for layer_key in layer_keys:
            groups = contiguous_partition(info["intermediate_size"], args.group_size)
            z_by_task = {}
            for task, prompts in CATEGORIES.items():
                zs = [collect_z(model, tokenizer, p, layer_key) for p in prompts[:3]]
                z_by_task[task] = np.concatenate(zs, axis=0)
                print(f"  {layer_key} {task}: {z_by_task[task].shape[0]} tokens")

            stats = task_conditional_stats(z_by_task, groups, args.quantile)
            for task, ps in stats["p_active_by_task"].items():
                for e, p in enumerate(ps):
                    w.writerow([layer_key, task, e, f"{p:.4f}"])
            summary[layer_key] = {
                "expert_task_entropy": stats["expert_task_entropy"],
                "active_per_token_mean": stats["active_per_token_mean"],
            }
            # universal core = highest-entropy quartile (§32)
            H = np.array(stats["expert_task_entropy"])
            core = np.argsort(H)[-max(1, len(H) // 4):].tolist()
            summary[layer_key]["universal_core"] = sorted(core)
            print(f"  -> mean active/token: {stats['active_per_token_mean']:.1f}/{len(groups)}")
    (out_csv.parent / "task_specialization.json").write_text(json.dumps(summary, indent=2))

    # heatmap: experts x tasks for the first probed layer
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lk = layer_keys[len(layer_keys) // 2]
        rows = list(CATEGORIES.keys())
        M = np.array([summary[lk]["expert_task_entropy"]])  # placeholder shape
        # rebuild per-task P(active) matrix from csv
        P = np.zeros((len(rows), max(len(g) for g in
                     [contiguous_partition(info["intermediate_size"], args.group_size)])))
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                if row["layer"] == lk:
                    P[rows.index(row["task"]), int(row["expert_id"])] = float(row["p_active"])
        plt.figure(figsize=(12, 4))
        plt.imshow(P, aspect="auto", cmap="magma", vmin=0, vmax=1)
        plt.colorbar(label="P(expert active | task)")
        plt.yticks(range(len(rows)), rows)
        plt.xlabel("Micro-expert id")
        plt.title(f"Task–expert activation heatmap — {lk}")
        plt.tight_layout()
        Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_plot, dpi=150)
        print(f"Saved plot: {args.out_plot}")
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
