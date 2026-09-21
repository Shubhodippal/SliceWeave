"""Experiment 13 — Wall-Clock Sparse Execution (§7 H7, §77, Milestone 10 gate).

Measures REAL decode latency of masked FFN execution vs dense. This is where
contextual sparsity can fail on hardware (§90): less arithmetic can lose to
irregular memory access. Report tok/s, P50/P95 token latency (§35, §97).

Usage:
    python experiments/13_wall_clock_runtime.py --model Qwen/Qwen3-4B --repeats 5
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import CATEGORIES  # noqa: E402
from src.hooks import FFNMaskedForward  # noqa: E402
from src.micro_experts import contiguous_partition, expert_importance_output, keep_mask_from_topk  # noqa: E402
from src.model_utils import (describe_model, get_mlp, load_model_and_tokenizer,  # noqa: E402
                             mlp_intermediate_size)
from src.oracle import run_baseline  # noqa: E402


def timed_decode(model, tokenizer, prompt: str, max_new: int, warmup: int = 2):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    times = []
    n = 0
    for i in range(warmup + 1):
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             min_new_tokens=max_new,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n = out.shape[1] - enc["input_ids"].shape[1]
        if i >= warmup:
            times.append(dt)
    return times, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--fractions", type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.2])
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--prompt", default=CATEGORIES["coding"][0])
    ap.add_argument("--out", default="results/csv/wall_clock.csv")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    li = args.layer if args.layer is not None else info["layers"] // 2
    layer_key = f"model.layers.{li}"
    mlp = get_mlp(model, layer_key)
    groups = contiguous_partition(mlp_intermediate_size(model, layer_key), args.group_size)
    E = len(groups)

    # oracle ranking for this prompt (one fixed mask per fraction)
    base = run_baseline(model, tokenizer, args.prompt, [layer_key])
    enc = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    pos = int(enc["input_ids"].shape[1]) - 1
    z = base["z"][f"{layer_key}.down_proj"][0, pos]
    imp = expert_importance_output(z[None, :], mlp.down_proj.weight, groups)[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fraction", "repeats", "tok_s_mean", "tok_s_median",
                    "latency_p50_ms", "latency_p95_ms", "gen_tokens"])
        for frac in args.fractions:
            if frac < 1.0:
                keep = keep_mask_from_topk(groups, imp, frac)
                ctx = FFNMaskedForward(model)
                ctx.attach(layer_key, mlp)
                ctx.set_mask(layer_key, keep)
            else:
                ctx = None
            try:
                times, ntok = timed_decode(model, tokenizer, args.prompt, args.max_new)
            finally:
                if ctx is not None:
                    ctx.__exit__(None, None, None)
            tok_s_list = [ntok / t for t in times]
            lat_ms = [t * 1e3 / max(ntok, 1) for t in times]
            w.writerow([frac, len(times), f"{statistics.mean(tok_s_list):.2f}",
                        f"{statistics.median(tok_s_list):.2f}",
                        f"{statistics.median(lat_ms):.2f}",
                        f"{sorted(lat_ms)[int(0.95*len(lat_ms))-1]:.2f}", ntok])
            print(f"frac={frac:.2f}: {statistics.mean(tok_s_list):.2f} tok/s "
                  f"(median {statistics.median(tok_s_list):.2f})")
    print(f"Saved: {out_path}")
    print("§90 check: if sparse tok/s < dense tok/s, sparsity lost to memory "
          "traffic — record and investigate layout/scheduling.")


if __name__ == "__main__":
    main()
