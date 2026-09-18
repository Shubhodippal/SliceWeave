"""Experiment 02 — All-Layer Oracle Sweep (Blueprint §29, Milestone 2).

For every layer, test multiple retention fractions at the last prompt token
and produce a layer-sensitivity table + plot.

Usage:
    python experiments/02_oracle_all_layers.py --model Qwen/Qwen3-4B --fractions 0.5 0.3 0.2 0.1
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import CATEGORIES  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.model_utils import describe_model, ffn_layer_names, find_ffn_modules, load_model_and_tokenizer  # noqa: E402
from src.oracle import oracle_single_layer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.5, 0.3, 0.2, 0.1])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--score-mode", choices=["z", "output"], default="output")
    ap.add_argument("--out", default="results/csv/layer_sensitivity.csv")
    ap.add_argument("--plot", default="results/plots/layer_sensitivity.png")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)

    ffn_modules = find_ffn_modules(model)
    layer_map = ffn_layer_names(ffn_modules)
    layer_keys = sorted(layer_map.keys(), key=lambda s: int(s.rsplit(".", 1)[-1]))

    # Small balanced prompt set across categories
    prompts = [p for ps in CATEGORIES.values() for p in ps[:2]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # results[layer_idx][fraction] = list of metric dicts
    agg = {i: {f: [] for f in args.fractions} for i in range(len(layer_keys))}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_idx", "layer", "retain_fraction", "cosine",
                         "rel_l2", "top1_match", "top10_overlap", "kl"])
        for li, layer in enumerate(layer_keys):
            for prompt in prompts:
                results = oracle_single_layer(model, tokenizer, prompt, layer,
                                              args.fractions,
                                              group_size=args.group_size,
                                              score_mode=args.score_mode)
                for r in results:
                    m = compute_all(r["logits"], r["baseline_logits"])
                    agg[li][r["retain_fraction"]].append(m)
                    writer.writerow([li, layer, r["retain_fraction"],
                                     f"{m['cosine']:.4f}", f"{m['rel_l2']:.4f}",
                                     m["top1_match"], f"{m['top10_overlap']:.2f}",
                                     f"{m['kl']:.4f}"])
            f.flush()
            mean_cos = {fr: sum(m['cosine'] for m in v) / max(len(v), 1)
                        for fr, v in agg[li].items()}
            print(f"layer {li:3d}/{len(layer_keys)}  " +
                  "  ".join(f"{fr:.0%}:{c:.3f}" for fr, c in mean_cos.items()))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        for fr in args.fractions:
            xs = list(range(len(layer_keys)))
            ys = [sum(m["cosine"] for m in agg[i][fr]) / max(len(agg[i][fr]), 1)
                  for i in xs]
            plt.plot(xs, ys, marker="o", markersize=3, label=f"{fr:.0%} experts")
        plt.xlabel("Layer index")
        plt.ylabel("Mean logit cosine vs dense baseline")
        plt.title(f"FFN micro-expert oracle sparsity — {args.model}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(f"Saved plot: {plot_path}")
    except Exception as e:  # plotting is best-effort
        print(f"Plot skipped: {e}")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
