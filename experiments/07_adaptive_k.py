"""Experiment 07 — Adaptive-K Progressive Activation (§20, §23, Milestone 6).

Compare fixed Top-K against the quality guard's progressive expansion: for the
same AVERAGE active fraction, does adaptivity preserve quality better?

Usage:
    python experiments/07_adaptive_k.py --model Qwen/Qwen3-4B --layer 12
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
from src.micro_experts import contiguous_partition, expert_importance_output  # noqa: E402
from src.model_utils import (describe_model, get_mlp, load_model_and_tokenizer,  # noqa: E402
                             mlp_down_weight, mlp_intermediate_size)
from src.oracle import run_baseline  # noqa: E402
from src.quality_guard import GuardConfig, QualityGuard, recovery_summary  # noqa: E402
from src.token_router import CentroidRouter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--initial-frac", type=float, default=0.15)
    ap.add_argument("--target-avg-frac", type=float, default=0.30,
                    help="fixed-K baseline matched to the guard's average budget")
    ap.add_argument("--out", default="results/csv/adaptive_compute_results.csv")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    layer_keys = [f"model.layers.{i}"
                  for i in torch.linspace(2, info["layers"] - 3, args.layers).int().tolist()]
    E = len(contiguous_partition(mlp_intermediate_size(model, layer_keys[0]),
                                 args.group_size))

    # cheap calibration for router centroids
    prompts = [(c, p) for c, ps in CATEGORIES.items() for p in ps]
    routers = {lk: CentroidRouter(lk, E, info["hidden_size"]) for lk in layer_keys}
    print("calibrating router on", len(prompts), "prompts ...")
    for cat, prompt in prompts:
        base = run_baseline(model, tokenizer, prompt, layer_keys)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        pos = int(enc["input_ids"].shape[1]) - 1
        for lk in layer_keys:
            z = base["z"][f"{lk}.down_proj"][0, pos]
            groups = contiguous_partition(mlp_intermediate_size(model, lk), args.group_size)
            imp = expert_importance_output(z[None, :], mlp_down_weight(model, lk), groups)[0]
            top = torch.topk(imp, max(1, E // 4)).indices
            # router input: use z itself projected into hidden proxy (cheap demo:
            # calibrate on mean z-energy direction; production uses hidden states)
            routers[lk].add_observation(z[: info["hidden_size"]].unsqueeze(0), top.tolist())
    for r in routers.values():
        r.finalize()

    guards = {lk: QualityGuard(routers[lk], GuardConfig(initial_fraction=args.initial_frac))
              for lk in layer_keys}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = {lk: [] for lk in layer_keys}
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "mode", "prompt", "initial_k", "final_k", "expansions",
                    "fallback_full", "cosine", "top1", "kl", "active_frac"])
        for cat, prompt in prompts:
            base = run_baseline(model, tokenizer, prompt, layer_keys)
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            pos = int(enc["input_ids"].shape[1]) - 1

            # oracle ranking (upper bound "cheating" selection, §12)
            for lk in layer_keys:
                z = base["z"][f"{lk}.down_proj"][0, pos]
                groups = contiguous_partition(mlp_intermediate_size(model, lk),
                                              args.group_size)
                imp = expert_importance_output(z[None, :], mlp_down_weight(model, lk),
                                               groups)[0]

                # --- adaptive run (guard) ---
                sel, rec = guards[lk].run(z[: info["hidden_size"]])
                records[lk].append(rec)
                from src.micro_experts import keep_mask_from_sets
                from src.hooks import FFNMaskedForward
                with FFNMaskedForward(model) as masked:
                    masked.attach(lk, get_mlp(model, lk))
                    masked.set_mask(lk, keep_mask_from_sets(groups, sel, device=z.device))
                    with torch.no_grad():
                        out = model(**enc)
                    m = compute_all(out.logits[0, -1].float().cpu(), base["logits"])
                    w.writerow([lk, "adaptive", prompt[:50], rec.initial_k, rec.final_k,
                                rec.expansions, rec.fallback_full, f"{m['cosine']:.4f}",
                                m["top1_match"], f"{m['kl']:.4f}",
                                f"{rec.final_k / E:.3f}"])

                # --- matched fixed-K baseline ---
                k_fixed = max(1, int(round(E * args.target_avg_frac)))
                from src.micro_experts import keep_mask_from_topk
                with FFNMaskedForward(model) as masked:
                    masked.attach(lk, get_mlp(model, lk))
                    masked.set_mask(lk, keep_mask_from_topk(groups, imp, k_fixed / E))
                    with torch.no_grad():
                        out = model(**enc)
                    m = compute_all(out.logits[0, -1].float().cpu(), base["logits"])
                    w.writerow([lk, "fixed", prompt[:50], k_fixed, k_fixed, 0, False,
                                f"{m['cosine']:.4f}", m["top1_match"], f"{m['kl']:.4f}",
                                f"{k_fixed / E:.3f}"])
            f.flush()
            print(f"  {cat}: '{prompt[:40]}...' done")

    print("\n=== Recovery distribution (§41) ===")
    for lk in layer_keys:
        s = recovery_summary(records[lk])
        print(f"{lk}: " + "  ".join(f"{k}={v:.2f}" for k, v in s.items()))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
