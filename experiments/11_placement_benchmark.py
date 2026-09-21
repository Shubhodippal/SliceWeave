"""Experiment 11 — CPU/GPU Placement Benchmark (§28-§31, Milestone 9).

Given calibrated block sizes and the cost model, compute the scheduler's
placement plan for a representative layer and compare theoretical dense vs
sparse latency (§77). Hardware numbers are placeholders until measured (§65).

Usage:
    python experiments/11_placement_benchmark.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scheduler import (BlockMeta, CostModel, Scheduler, Tier,  # noqa: E402
                           Device, simulate_step_latency, flops_per_expert)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts-per-layer", type=int, default=136)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=2560, help="Qwen3-4B hidden size")
    ap.add_argument("--vram-mib", type=int, default=2048)
    ap.add_argument("--out", default="results/csv/placement_benchmark.csv")
    args = ap.parse_args()

    block_bytes = 2 * args.group_size * args.hidden * 2  # bf16 gate+up rows
    sched = Scheduler(CostModel(), vram_budget_bytes=args.vram_mib * 1024**2)

    # register a synthetic-but-representative block population with reuse stats
    import numpy as np
    rng = np.random.default_rng(0)
    for l in range(args.layers):
        lk = f"model.layers.{l}"
        for e in range(args.experts_per_layer):
            freq = float(rng.beta(0.4, 6.0))  # skewed: few hot, many cold (§15)
            sched.register(BlockMeta(
                layer_key=lk, expert_id=e, size_bytes=block_bytes,
                tier=Tier.SSD if freq < 0.15 else (Tier.RAM if freq < 0.5 else Tier.VRAM),
                device=Device.GPU,
                activation_frequency=freq,
                reuse_score=float(rng.random() * freq),
                task_relevance=float(rng.random()),
                universal_core=(e < 8),  # top-8 per layer as universal core (§32)
            ))

    # fill VRAM by placement score (§29), then schedule one layer's top-K (§46)
    candidates = [(f"model.layers.{l}", e) for l in range(args.layers)
                  for e in range(args.experts_per_layer)]
    pinned = sched.fill_vram_cache(candidates)
    n_vram = sum(1 for k in pinned if not sched.meta[k].universal_core)

    ranked = list(range(24))  # router's top-24 experts for this step (18% of 136)
    plan = sched.token_plan("model.layers.2", ranked, args.group_size, args.hidden,
                            max_blocks=len(ranked))
    dense_bytes = 3 * args.hidden * args.group_size * args.experts_per_layer * 2

    res = simulate_step_latency(dense_bytes, plan)
    gpu_n = sum(1 for d in plan if d.device == Device.GPU)
    cpu_n = sum(1 for d in plan if d.device == Device.CPU)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "expert_id", "device", "transfers", "est_latency_ms",
                    "reason"])
        for d in plan:
            w.writerow([d.layer_key, d.expert_id, d.device.value,
                        "→".join(f"{a.value}->{b.value}" for a, b in d.transfers) or "-",
                        f"{d.est_latency_ms:.4f}", d.reason])
        with open(out_path.parent / "placement_summary.csv", "w", newline="",
                  encoding="utf-8") as f2:
            w2 = csv.writer(f2)
            w2.writerow(["metric", "value"])
            w2.writerow(["blocks_in_plan", len(plan)])
            w2.writerow(["gpu_blocks", gpu_n])
            w2.writerow(["cpu_blocks", cpu_n])
            w2.writerow(["vram_resident", n_vram])
            w2.writerow(["dense_ms", f"{res['dense_ms']:.4f}"])
            w2.writerow(["sparse_ms", f"{res['sparse_ms']:.4f}"])
            w2.writerow(["speedup_theoretical", f"{res['speedup_theoretical']:.2f}"])

    print(f"plan: {len(plan)} blocks  (GPU {gpu_n} / CPU {cpu_n}); "
          f"VRAM-resident {n_vram} + core")
    print(f"dense step: {res['dense_ms']:.3f} ms   sparse plan: {res['sparse_ms']:.3f} ms "
          f"-> theoretical speedup {res['speedup_theoretical']:.2f}x")
    print("NOTE: cost-model numbers are placeholders until calibrated on the 5800H/3060 "
          "(§65); treat as relative, not absolute.")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
