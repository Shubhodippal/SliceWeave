"""Experiment 09 — Memory Trace + Cache Simulation (§24-§25, §29, Milestone 7).

Record the expert access trace of oracle/routed execution, then replay it
against the tiered cache (VRAM/RAM/SSD) with LRU/LFU and different capacities.
Deliverables: expert trace + cache hit rates + MTR inputs (§39).

Usage:
    python experiments/09_memory_trace.py --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache import TieredCache, trace_from_lists, reuse_distance_histogram  # noqa: E402
from src.datasets import CATEGORIES  # noqa: E402
from src.micro_experts import contiguous_partition, expert_importance_output  # noqa: E402
from src.model_utils import (describe_model, load_model_and_tokenizer,  # noqa: E402
                             mlp_down_weight, mlp_intermediate_size)
from src.oracle import run_baseline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--keep-frac", type=float, default=0.25)
    ap.add_argument("--vram-mib", type=int, default=512)
    ap.add_argument("--ram-mib", type=int, default=4096)
    ap.add_argument("--out-trace", default="results/traces/expert_trace.csv")
    ap.add_argument("--out-csv", default="results/csv/cache_simulation.csv")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    layer_keys = [f"model.layers.{i}"
                  for i in torch.linspace(2, info["layers"] - 3, args.layers).int().tolist()]
    inter = mlp_intermediate_size(model, layer_keys[0])
    E = len(contiguous_partition(inter, args.group_size))
    hidden = info["hidden_size"]
    block_bytes = 2 * args.group_size * hidden * 2  # bf16 gate+up rows for one block

    prompts = [(c, p) for c, ps in CATEGORIES.items() for p in ps]
    # ---- record trace: top keep_frac experts per layer per prompt-token step
    layer_sel, token_ids = [], []
    print("recording expert trace ...")
    for cat, prompt in prompts:
        base = run_baseline(model, tokenizer, prompt, layer_keys)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        pos = int(enc["input_ids"].shape[1]) - 1
        step_experts = []
        for lk in layer_keys:
            z = base["z"][f"{lk}.down_proj"][0, pos]
            groups = contiguous_partition(mlp_intermediate_size(model, lk), args.group_size)
            imp = expert_importance_output(z[None, :], mlp_down_weight(model, lk), groups)[0]
            k = max(1, int(round(E * args.keep_frac)))
            step_experts.append(torch.topk(imp, k).indices.cpu().tolist())
        layer_sel.append(step_experts)
        token_ids.append(pos)

    trace = trace_from_lists(layer_keys, layer_sel, token_ids,
                             sizes={(lk, e): block_bytes for lk in layer_keys
                                    for e in range(E)})
    Path(args.out_trace).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_trace, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "layer", "expert_id", "token_id", "bytes"])
        for r in trace:
            w.writerow([r.step, r.layer_key, r.expert_id, r.token_id, r.bytes_size])
    print(f"trace: {len(trace)} accesses over {len(layer_keys)} layers "
          f"({E} experts/layer), block={block_bytes/1024:.0f} KiB")

    # ---- replay against cache configurations (§52 baselines)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "vram_mib", "ram_mib", "vram_hit_rate", "ram_hit_rate",
                    "ssd_miss_rate", "bytes_vram_read", "bytes_ram_read",
                    "bytes_ssd_read", "mtr_vs_all_ssd"])
        total_bytes = len(trace) * block_bytes
        for policy in ("lru", "lfu"):
            for vram_mib in (args.vram_mib // 4, args.vram_mib, args.vram_mib * 4):
                cache = TieredCache(vram_mib * 1024**2, args.ram_mib * 1024**2,
                                    policy=policy, block_bytes=block_bytes)
                for r in trace:
                    cache.access(r)
                s = cache.stats
                hr = s.hit_rates()
                w.writerow([policy, vram_mib, args.ram_mib,
                            f"{hr['vram_hit_rate']:.3f}", f"{hr['ram_hit_rate']:.3f}",
                            f"{hr['ssd_miss_rate']:.3f}", s.bytes_vram_read,
                            s.bytes_ram_read, s.bytes_ssd_read,
                            f"{(s.bytes_ssd_read + s.bytes_ram_read) / total_bytes:.3f}"])
                print(f"{policy} vram={vram_mib:5d}MiB  vramHR={hr['vram_hit_rate']:.2f} "
                      f"ramHR={hr['ram_hit_rate']:.2f}  ssdMiss={hr['ssd_miss_rate']:.2f}")

    # ---- reuse distance (§43)
    hist = reuse_distance_histogram(trace)
    gaps = np.array(list(hist.keys()))
    if len(gaps):
        print(f"reuse distance: median={int(np.median(gaps))} "
              f"p90={int(np.quantile(gaps, 0.9))} max={int(gaps.max())}")
        with open(out_path.parent / "reuse_distance.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["gap", "count"])
            for g, c in hist.items():
                w.writerow([g, c])
    print(f"Saved: {out_path}, {args.out_trace}")


if __name__ == "__main__":
    main()
