"""Experiment 01 — Oracle FFN Sparsity Test (§12-§14, Milestone 1).

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from src.datasets import CATEGORIES, all_prompts, generated_arithmetic  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.model_utils import describe_model, ffn_layer_names, find_ffn_modules, load_model_and_tokenizer  # noqa: E402
from src.oracle import oracle_multi_token, oracle_single_layer  # noqa: E402

RETAIN_FRACTIONS = [1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.10, 0.05]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, default=3, help="number of evenly spaced middle layers")
    ap.add_argument("--prompts", type=int, default=8, help="prompts per category (capped by dataset)")
    ap.add_argument("--positions", type=int, default=8,
                    help="analyzed tokens per prompt (multi-token oracle, §20)")
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

    prompts = all_prompts()
    if args.prompts > 0:
        prompts = [(c, p) for c, ps in CATEGORIES.items()
                   for p in ps[: args.prompts]]
        prompts += [("math_gen", p) for p in generated_arithmetic(args.prompts * 2)]
    print(f"prompt instances: {len(prompts)}  x {args.positions} positions each")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "prompt", "layer", "retain_fraction", "score_mode",
                         "cosine", "top1_match", "kl", "n_positions"])

        for li, layer in enumerate(chosen):
            for cat, prompt in prompts:
                results = oracle_multi_token(
                    model, tokenizer, prompt, layer,
                    RETAIN_FRACTIONS, group_size=args.group_size,
                    score_mode=args.score_mode, max_positions=args.positions,
                )
                for r in results:
                    n_pos = len(r["cosines"])
                    for c, t1, kl in zip(r["cosines"], r["top1"], r["kl"]):
                        writer.writerow([cat, prompt[:60], layer, r["retain_fraction"],
                                         args.score_mode, f"{c:.4f}", t1,
                                         f"{kl:.4f}", n_pos])
                    rows.append((r["retain_fraction"], r, cat))
                f.flush()
            print(f"[{li+1}/{len(chosen)}] layer {layer} done")

    # Summary: mean ± std per retention fraction (§97) + per-category means
    import statistics
    print("\n=== Summary (mean ± std over all analyzed positions) ===")
    print(f"{'retain':>8} {'n':>6} {'cosine':>16} {'top1':>6} {'kl':>8}")
    by_frac = {}
    for frac, r, _cat in rows:
        by_frac.setdefault(frac, []).extend(r["cosines"])
    gate = {}
    for frac in sorted(by_frac):
        cs = by_frac[frac]
        t1s = [t for fr, r, _ in rows if fr == frac for t in r["top1"]]
        kls = [k for fr, r, _ in rows if fr == frac for k in r["kl"]]
        gate[frac] = (sum(cs) / len(cs), sum(t1s) / len(t1s))
        print(f"{frac:>8.2f} {len(cs):>6} {sum(cs)/len(cs):>8.4f} ±{statistics.stdev(cs) if len(cs) > 1 else 0.0:<7.4f} "
              f"{sum(t1s)/len(t1s):>6.2f} {sum(kls)/len(kls):>8.4f}")

    print("\n=== Per-category mean cosine at 30% retention ===")
    by_cat = {}
    for frac, r, cat in rows:
        if frac == 0.3:
            by_cat.setdefault(cat, []).extend(r["cosines"])
    for cat, cs in sorted(by_cat.items()):
        print(f"  {cat:22s} {sum(cs)/len(cs):.4f}  (n={len(cs)})")

    # §14 decision gate hint
    if 0.3 in gate and 0.2 in gate:
        c30, t30 = gate[0.3]
        c20, t20 = gate[0.2]
        print("\n§14 gate: cosine@30%=" + f"{c30:.4f} top1@30%={t30:.2f}  "
              f"cosine@20%={c20:.4f} top1@20%={t20:.2f}")
        if c30 > 0.995 and t30 > 0.9:
            print("  -> STRONG signal: <=30% experts preserve logits. Continue to routing.")
        elif c30 > 0.98:
            print("  -> INTERESTING signal: investigate clustering/router before deciding.")
        elif c30 > 0.9:
            print("  -> WEAK signal: >30% may be required; broaden prompt set before pivoting.")
        else:
            print("  -> FAILURE zone: quality collapses well above 30%. Reconsider (§55).")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
