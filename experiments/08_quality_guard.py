"""Experiment 08 — Quality Guard Ablation (§21-§23, Milestone 6 deliverable).

Quality vs active-parameter budget (APR, §37): sweep confidence thresholds and
initial fractions; produce quality_vs_APR data + fallback distribution.

Usage:
    python experiments/08_quality_guard.py --model Qwen/Qwen3-4B
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
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.7, 0.8, 0.9, 0.95])
    ap.add_argument("--initial-fracs", type=float, nargs="+",
                    default=[0.1, 0.15, 0.25])
    ap.add_argument("--out", default="results/csv/quality_vs_APR.csv")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    li = args.layer if args.layer is not None else info["layers"] // 2
    layer_key = f"model.layers.{li}"
    E = len(contiguous_partition(mlp_intermediate_size(model, layer_key),
                                 args.group_size))

    prompts = [(c, p) for c, ps in CATEGORIES.items() for p in ps]
    router = CentroidRouter(layer_key, E, info["hidden_size"])
    print("calibrating ...")
    for cat, prompt in prompts[: int(len(prompts) * 0.7)]:
        base = run_baseline(model, tokenizer, prompt, [layer_key])
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        pos = int(enc["input_ids"].shape[1]) - 1
        z = base["z"][f"{layer_key}.down_proj"][0, pos]
        groups = contiguous_partition(mlp_intermediate_size(model, layer_key),
                                      args.group_size)
        imp = expert_importance_output(z[None, :], mlp_down_weight(model, layer_key),
                                       groups)[0]
        top = torch.topk(imp, max(1, E // 4)).indices.tolist()
        router.add_observation(z[: info["hidden_size"]].unsqueeze(0), top)
    router.finalize()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_frac", "conf_threshold", "mean_final_k", "mean_APR",
                    "mean_cosine", "top1_rate", "fallback_full_rate"])
        for init in args.initial_fracs:
            for thr in args.thresholds:
                guard = QualityGuard(router, GuardConfig(
                    initial_fraction=init, confidence_threshold=thr))
                ks, coss, t1s = [], [], []
                recs = []
                for cat, prompt in prompts[int(len(prompts) * 0.7):]:
                    base = run_baseline(model, tokenizer, prompt, [layer_key])
                    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
                    pos = int(enc["input_ids"].shape[1]) - 1
                    z = base["z"][f"{layer_key}.down_proj"][0, pos]
                    groups = contiguous_partition(mlp_intermediate_size(model, layer_key),
                                                  args.group_size)
                    imp = expert_importance_output(z[None, :],
                                                   mlp_down_weight(model, layer_key),
                                                   groups)[0]
                    from src.micro_experts import keep_mask_from_sets
                    from src.hooks import FFNMaskedForward
                    sel, rec = guard.run(z[: info["hidden_size"]])
                    recs.append(rec)
                    ks.append(rec.final_k)
                    with FFNMaskedForward(model) as masked:
                        masked.attach(layer_key, get_mlp(model, layer_key))
                        masked.set_mask(layer_key, keep_mask_from_sets(
                            groups, sel, device=z.device))
                        with torch.no_grad():
                            out = model(**enc)
                        m = compute_all(out.logits[0, -1].float().cpu(), base["logits"])
                        coss.append(m["cosine"])
                        t1s.append(m["top1_match"])
                apr = sum(ks) / len(ks) / E
                s = recovery_summary(recs)
                w.writerow([init, thr, f"{sum(ks)/len(ks):.1f}", f"{apr:.3f}",
                            f"{sum(coss)/len(coss):.4f}",
                            f"{sum(t1s)/len(t1s):.3f}",
                            f"{s.get('fallback_full', 0):.3f}"])
                print(f"init={init:.2f} thr={thr:.2f}  APR={apr:.2f}  "
                      f"cos={sum(coss)/len(coss):.4f}  top1={sum(t1s)/len(t1s):.2f}  "
                      f"fallback={s.get('fallback_full', 0):.2f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
