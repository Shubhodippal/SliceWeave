"""Experiment 01 — Oracle FFN Sparsity Test (Blueprint §9-12, §28, Milestone 1).

Falsification tool: for ONE layer at a time, mask low-ranked micro-experts of
the FFN intermediate at the last prompt token and measure how final logits
degrade vs. the dense baseline.

Usage:
    python experiments/01_oracle_ffn.py --model Qwen/Qwen3-4B --layers 3 --prompts 8
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

RETAIN_FRACTIONS = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.10, 0.05]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, default=3, help="number of evenly spaced middle layers")
    ap.add_argument("--prompts", type=int, default=8, help="prompts per category (capped by dataset)")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--score-mode", choices=["z", "output"], default="output")
    ap.add_argument("--out", default="results/csv/oracle_ffn.csv")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    print(f"params={info['params']/1e9:.2f}B layers={info['layers']} "
          f"hidden={info['hidden_size']} inter={info['intermediate_size']}")

    ffn_modules = find_ffn_modules(model)
    layer_map = ffn_layer_names(ffn_modules)
    layer_keys = sorted(layer_map.keys(), key=lambda s: int(s.rsplit(".", 1)[-1]))

    # Pick evenly spaced layers from the middle 80% (skip embedding-flavoured early layers)
    n = info["layers"]
    idxs = [int(round(n * f)) for f in (0.2, 0.5, 0.8)][: args.layers]
    chosen = [layer_keys[i] for i in dict.fromkeys(idxs)]
    print("Testing layers:", chosen)

    prompts = []
    for cat, ps in CATEGORIES.items():
        prompts += [(cat, p) for p in ps[: args.prompts]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "prompt", "layer", "retain_fraction", "score_mode",
                         "cosine", "rel_l2", "top1_match", "top10_overlap",
                         "top50_overlap", "kl"])

        for li, layer in enumerate(chosen):
            for cat, prompt in prompts:
                results = oracle_single_layer(
                    model, tokenizer, prompt, layer,
                    RETAIN_FRACTIONS, group_size=args.group_size,
                    score_mode=args.score_mode,
                )
                for r in results:
                    m = compute_all(r["logits"], r["baseline_logits"])
                    writer.writerow([cat, prompt[:60], layer, r["retain_fraction"],
                                     args.score_mode, f"{m['cosine']:.4f}",
                                     f"{m['rel_l2']:.4f}", m["top1_match"],
                                     f"{m['top10_overlap']:.2f}",
                                     f"{m['top50_overlap']:.2f}", f"{m['kl']:.4f}"])
                    rows.append((r["retain_fraction"], m))
                f.flush()
            print(f"[{li+1}/{len(chosen)}] layer {layer} done")

    # Summary: mean cosine / top1 per retention fraction
    print("\n=== Summary (mean over prompts) ===")
    print(f"{'retain':>8} {'cosine':>8} {'top1':>6} {'kl':>8}")
    by_frac = {}
    for frac, m in rows:
        by_frac.setdefault(frac, []).append(m)
    for frac in sorted(by_frac):
        ms = by_frac[frac]
        cos = sum(m["cosine"] for m in ms) / len(ms)
        t1 = sum(m["top1_match"] for m in ms) / len(ms)
        kl = sum(m["kl"] for m in ms) / len(ms)
        print(f"{frac:>8.2f} {cos:>8.4f} {t1:>6.2f} {kl:>8.4f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
