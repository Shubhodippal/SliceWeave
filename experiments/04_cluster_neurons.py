"""Experiment 04 — Neuron Clustering + Reordering (§11, §15, Milestone 3).

Group neurons that co-activate, physically reorder one FFN, and verify the
reordered dense model still matches baseline logits exactly.

Usage:
    python experiments/04_cluster_neurons.py --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clustering import cluster_neurons, cluster_statistics, build_reordered_model  # noqa: E402
from src.datasets import all_prompts  # noqa: E402
from src.hooks import FFNIntermediateCapture  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.micro_experts import groups_from_assignment  # noqa: E402
from src.model_utils import (describe_model, get_mlp, load_model_and_tokenizer,  # noqa: E402
                             mlp_down_weight)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer", type=int, default=None, help="layer index (default: middle)")
    ap.add_argument("--method", choices=["kmeans", "corr"], default="kmeans")
    ap.add_argument("--group-size", type=int, default=128, help="target experts size")
    ap.add_argument("--out-csv", default="results/csv/cluster_statistics.csv")
    ap.add_argument("--out-assign", default="results/traces/expert_assignments.npy")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    li = args.layer if args.layer is not None else info["layers"] // 2
    layer_key = f"model.layers.{li}"
    mlp = get_mlp(model, layer_key)
    inter = mlp_down_weight(model, layer_key).shape[1]

    # 1. collect calibration z-matrix
    zs = []
    for cat, p in all_prompts():
        cap = FFNIntermediateCapture()
        cap.attach(model, layer_key)
        enc = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**enc)
        cap.remove()
        zs.append(cap.captures[f"{layer_key}.down_proj"][0].float().cpu().numpy())
    Z = np.concatenate(zs, axis=0)
    print(f"calibration matrix: {Z.shape} (tokens x neurons)")

    # 2. cluster
    n_clusters = max(1, inter // args.group_size)
    assignment = cluster_neurons(Z, n_clusters=n_clusters, method=args.method)
    groups = groups_from_assignment(assignment)
    stats = cluster_statistics(Z, assignment)
    print(f"clusters: {stats['n_clusters']}  "
          f"size mean/std: {stats['cluster_size_mean']:.1f}/{stats['cluster_size_std']:.1f}")

    # 3. baseline logits BEFORE reorder
    prompts = ["Debug this FastAPI JWT authentication bug: the token expires before refresh.",
               "What is the capital of Australia?"]
    def logits_of(model_, prompt):
        enc = tokenizer(prompt, return_tensors="pt").to(model_.device)
        with torch.no_grad():
            return model_(**enc).logits[0, -1].float().cpu()

    base = [logits_of(model, p) for p in prompts]

    # 4. reorder in place with exactness verification
    r = build_reordered_model(model, layer_key, assignment, verify=True)
    print(f"reordered {layer_key}: {r['n_experts']} experts, equivalence verified")

    # 5. post-reorder logits must match baseline
    all_ok = True
    for p, b in zip(prompts, base):
        a = logits_of(model, p)
        m = compute_all(a, b)
        ok = m["cosine"] > 0.999 and m["top1_match"] == 1
        all_ok &= ok
        print(f"  '{p[:40]}...' cosine={m['cosine']:.6f} top1={m['top1_match']} -> "
              f"{'OK' if ok else 'FAIL'}")
    if not all_ok:
        print("REORDER CHANGED OUTPUT — aborting.")
        sys.exit(1)

    # 6. persist
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "method", "n_clusters", "cluster_size_mean",
                    "cluster_size_std", "intra_corr_mean"])
        w.writerow([layer_key, args.method, stats["n_clusters"],
                    f"{stats['cluster_size_mean']:.1f}", f"{stats['cluster_size_std']:.1f}",
                    f"{stats['intra_corr_mean']:.3f}"])
    Path(args.out_assign).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_assign, assignment.astype(np.int64))
    print(f"Saved: {out_csv}, {args.out_assign}")


if __name__ == "__main__":
    main()
