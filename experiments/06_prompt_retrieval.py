"""Experiment 06 — Prompt-Level Candidate Pool Retrieval (§16, §47, Milestone 5).

Build per-task capability profiles, then for holdout prompts measure how much
the candidate pool narrows the expert universe (CPR, §38) and what oracle
recall survives inside the pool.

Usage:
    python experiments/06_prompt_retrieval.py --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import CATEGORIES  # noqa: E402
from src.hooks import HiddenStateCapture  # noqa: E402
from src.micro_experts import contiguous_partition, expert_importance_output  # noqa: E402
from src.model_utils import (describe_model, load_model_and_tokenizer,  # noqa: E402
                             mlp_down_weight, mlp_intermediate_size)
from src.oracle import run_baseline  # noqa: E402
from src.prompt_router import PromptRouter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--top-tasks", type=int, default=2)
    ap.add_argument("--pool-frac", type=float, default=0.25,
                    help="experts per task admitted to the pool")
    ap.add_argument("--out", default="results/csv/candidate_pool_results.csv")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    layer_keys = [f"model.layers.{i}"
                  for i in torch.linspace(2, info["layers"] - 3, args.layers).int().tolist()]

    prompts = [(c, p) for c, ps in CATEGORIES.items() for p in ps]
    n_cal = int(len(prompts) * 0.7)
    cal, hold = prompts[:n_cal], prompts[n_cal:]

    pr = PromptRouter()
    E = len(contiguous_partition(info["intermediate_size"], args.group_size))

    # ---- calibrate task directions + usage (§47 step 1-3, offline)
    print("Calibrating task profiles ...")
    for cat, prompt in cal:
        hs = HiddenStateCapture()
        for lk in layer_keys:
            hs.attach(model, lk)
        base = run_baseline(model, tokenizer, prompt, layer_keys)
        hs.remove()
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        pos = int(enc["input_ids"].shape[1]) - 1
        for lk in layer_keys:
            h = hs.captures[f"{lk}.input"][0, :pos + 1].float().cpu().numpy()
            pr.add_task_hidden(lk, cat, h.mean(axis=0))
            groups = contiguous_partition(mlp_intermediate_size(model, lk), args.group_size)
            z = base["z"][f"{lk}.down_proj"][0, pos]
            imp = expert_importance_output(z[None, :], mlp_down_weight(model, lk), groups)[0]
            top = torch.topk(imp, max(1, E // 4)).indices.cpu().numpy()
            u = np.zeros(E, dtype=np.float32)
            u[top] = 1.0
            acc = pr.task_usage.get(lk, {}).get(cat)
            pr.add_task_usage(lk, cat, u if acc is None else acc + u)
    pr.finalize()
    # usage -> frequency
    for lk in pr.task_usage:
        for cat in pr.task_usage[lk]:
            n = max(sum(1 for c, _ in cal if c == cat), 1)
            pr.task_usage[lk][cat] /= n

    # ---- holdout retrieval (§47)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recalls = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true_task", "prompt", "top1_task", "pool_size", "universe",
                    "pool_ratio", "oracle_in_pool_recall"])
        for true_cat, prompt in hold:
            hs = HiddenStateCapture()
            for lk in layer_keys:
                hs.attach(model, lk)
            base = run_baseline(model, tokenizer, prompt, layer_keys)
            hs.remove()
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            pos = int(enc["input_ids"].shape[1]) - 1

            profile = pr.profile_prompt(
                {lk: hs.captures[f"{lk}.input"][0, :pos + 1].float().cpu().numpy()
                 for lk in layer_keys})
            pool = pr.build_pool(profile, top_tasks=args.top_tasks,
                                 experts_per_task_frac=args.pool_frac)

            # oracle inside pool: how many of the true top experts does the pool cover?
            recs = []
            for lk in layer_keys:
                groups = contiguous_partition(mlp_intermediate_size(model, lk),
                                              args.group_size)
                z = base["z"][f"{lk}.down_proj"][0, pos]
                imp = expert_importance_output(z[None, :], mlp_down_weight(model, lk),
                                               groups)[0]
                top = set(torch.topk(imp, max(1, E // 4)).indices.tolist())
                cand = set(pool.candidates_for(lk))
                recs.append(len(top & cand) / max(len(top), 1))
            r = float(np.mean(recs))
            recalls.append(r)
            w.writerow([true_cat, prompt[:50], profile.top_tasks[0] if profile.top_tasks else "",
                        E, E, f"{len(pool.by_layer.get(layer_keys[0], []))/E:.3f}", f"{r:.3f}"])
            print(f"  [{true_cat}] top1='{profile.top_tasks[0] if profile.top_tasks else '?'}' "
                  f"pool={len(pool.by_layer.get(layer_keys[0], []))}/{E} oracle-recall={r:.2f}")

    print(f"\nmean oracle-in-pool recall: {sum(recalls)/len(recalls):.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
