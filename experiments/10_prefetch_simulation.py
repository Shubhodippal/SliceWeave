"""Experiment 10 — Predictive Prefetch Simulation (§26-§27, §42, Milestone 8).

Compare prefetch horizons (§27): cross-layer, next-token, speculative (t+1,t+2)
and the oracle-hindsight ceiling. Report precision/recall/wasted bytes (§42)
and simulated traffic reduction.

Usage:
    python experiments/10_prefetch_simulation.py --trace results/traces/expert_trace.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache import AccessRecord  # noqa: E402
from src.metrics import mtr  # noqa: E402
from src.prefetch import (CrossLayerPredictor, NextTokenPredictor,  # noqa: E402
                          OracleHindsightPredictor, evaluate_prefetch, wasted_bytes)


def load_trace(path: str):
    recs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            recs.append(AccessRecord(step=int(row["step"]), layer_key=row["layer"],
                                     expert_id=int(row["expert_id"]),
                                     token_id=int(row["token_id"]),
                                     bytes_size=int(row["bytes"])))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="results/traces/expert_trace.csv",
                    help="trace from experiment 09")
    ap.add_argument("--horizon", type=int, default=8, help="prediction window (accesses)")
    ap.add_argument("--out", default="results/csv/prefetch_accuracy.csv")
    args = ap.parse_args()

    trace = load_trace(args.trace)
    print(f"trace: {len(trace)} accesses")
    sizes = {(r.layer_key, r.expert_id): r.bytes_size or 1 for r in trace}
    H = args.horizon

    # ground truth needs at each step = blocks demanded in the next H accesses
    needed = [OracleHindsightPredictor.next_needs(trace, i, H) for i in range(len(trace))]

    # ---- cross-layer predictor (§94)
    clp = CrossLayerPredictor()
    clp.observe(trace)
    # ---- next-token predictor (§95)
    ntp = NextTokenPredictor()
    ntp.observe(trace)

    pred_cross, pred_next, pred_oracle = [], [], []
    for i, r in enumerate(trace):
        cur = (r.layer_key, r.expert_id)
        pred_cross.append(clp.predict(cur, layer_key=r.layer_key, topk=H))
        window = [(x.layer_key, x.expert_id) for x in trace[max(0, i - 4):i + 1]]
        pred_next.append(ntp.predict(window, topk=H))
        pred_oracle.append(OracleHindsightPredictor.next_needs(trace, i, H))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("cross_layer", evaluate_prefetch(pred_cross, needed)),
        ("next_token", evaluate_prefetch(pred_next, needed)),
        ("oracle_hindsight", evaluate_prefetch(pred_oracle, needed)),
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["predictor", "precision_mean", "recall_mean", "precision_union",
                    "recall_union", "wasted_bytes"])
        for name, m in rows:
            wb = wasted_bytes(
                {"cross_layer": pred_cross, "next_token": pred_next,
                 "oracle_hindsight": pred_oracle}[name],
                needed, sizes)
            w.writerow([name, f"{m['precision_mean']:.3f}", f"{m['recall_mean']:.3f}",
                        f"{m['precision_union']:.3f}", f"{m['recall_union']:.3f}", wb])
            print(f"{name:16s} P={m['precision_mean']:.2f} R={m['recall_mean']:.2f} "
                  f"wasted={wb/1e6:.1f} MB")

    # traffic reduction estimate: with perfect prefetch of hits, SSD reads drop
    # to first-touch only; RAM->VRAM traffic drops for prefetched blocks (§26).
    first_touch = len({(r.layer_key, r.expert_id) for r in trace})
    all_bytes = sum(sizes.values())
    ssd_min = int(sum(sizes[(r.layer_key, r.expert_id)] for r in trace) / max(
        len(trace), 1) * first_touch)
    f2 = open(out_path.parent / "traffic_reduction.csv", "w", newline="",
              encoding="utf-8")
    w = csv.writer(f2)
    w.writerow(["scenario", "ssd_bytes", "mtr_vs_full"])
    w.writerow(["no_prefetch", all_bytes, mtr({"ssd": all_bytes}, {"ssd": all_bytes})])
    w.writerow(["oracle_prefetch", ssd_min, mtr({"ssd": ssd_min}, {"ssd": all_bytes})])
    f2.close()
    print(f"\nfirst-touch blocks: {first_touch}; oracle-prefetch SSD traffic "
          f"{ssd_min/all_bytes:.1%} of no-prefetch (lower bound)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
