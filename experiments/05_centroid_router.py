"""Experiment 05 — Centroid Router vs Oracle (§17-§19, Milestone 4).

Calibrate per-expert centroids from hidden states where experts were
oracle-important (§18), then measure router recall vs the oracle (§19) and the
router's overhead budget (§80).

Usage:
    python experiments/05_centroid_router.py --model Qwen/Qwen3-4B --layers 3
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import CATEGORIES  # noqa: E402
from src.hooks import HiddenStateCapture  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.micro_experts import contiguous_partition, expert_importance_output  # noqa: E402
from src.model_utils import (describe_model, get_mlp, load_model_and_tokenizer,  # noqa: E402
                             mlp_down_weight, mlp_intermediate_size)
from src.oracle import run_baseline  # noqa: E402
from src.token_router import (CentroidRouter, RandomRouter,  # noqa: E402
                              router_recall_vs_oracle, oracle_topk_from_importance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--k", type=int, default=0, help="experts to select (0 = 25% of E)")
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--out", default="results/csv/router_vs_oracle.csv")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    layer_keys = [f"model.layers.{i}"
                  for i in torch.linspace(2, info["layers"] - 3, args.layers).int().tolist()]
    print("Layers:", layer_keys)

    # split prompts into calibration / holdout (§52: routing must generalize)
    prompts = [(c, p) for c, ps in CATEGORIES.items() for p in ps]
    n_cal = int(len(prompts) * (1 - args.holdout_frac))
    cal, hold = prompts[:n_cal], prompts[n_cal:]
    print(f"calibration prompts: {len(cal)}   holdout: {len(hold)}")

    routers = {}
    for layer_key in layer_keys:
        mlp = get_mlp(model, layer_key)
        E = len(contiguous_partition(mlp_intermediate_size(model, layer_key),
                                     args.group_size))
        routers[layer_key] = CentroidRouter(layer_key, E, info["hidden_size"],
                                            device=next(mlp.parameters()).device.type)

    # ---- calibration (§18): hidden states where experts are oracle-important
    print("Calibrating centroids ...")
    for cat, prompt in cal:
        # one forward captures hidden states of all probed layers at once
        hs = HiddenStateCapture()
        for layer_key in layer_keys:
            hs.attach(model, layer_key)
        base = run_baseline(model, tokenizer, prompt, layer_keys)
        hs.remove()
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        pos = int(enc["input_ids"].shape[1]) - 1  # last prompt token

        for layer_key in layer_keys:
            x = hs.captures[f"{layer_key}.input"][0, pos]
            z = base["z"][f"{layer_key}.down_proj"][0, pos]
            mlp = get_mlp(model, layer_key)
            groups = contiguous_partition(mlp_intermediate_size(model, layer_key),
                                          args.group_size)
            imp = expert_importance_output(z[None, :], mlp_down_weight(model, layer_key),
                                           groups)[0]
            top = oracle_topk_from_importance(imp, max(1, len(groups) // 4))
            routers[layer_key].add_observation(x, top)
    for r in routers.values():
        r.finalize()
    print("centroids built.")

    k = args.k or 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "prompt", "router", "k", "recall", "precision",
                    "jaccard", "router_ms"])
        for layer_key in layer_keys:
            mlp = get_mlp(model, layer_key)
            groups = contiguous_partition(mlp_intermediate_size(model, layer_key),
                                          args.group_size)
            E = len(groups)
            kk = k or max(1, E // 4)
            random_router = RandomRouter(E, seed=0)

            for cat, prompt in hold:
                hs = HiddenStateCapture()
                hs.attach(model, layer_key)
                base = run_baseline(model, tokenizer, prompt, [layer_key])
                hs.remove()
                enc = tokenizer(prompt, return_tensors="pt").to(model.device)
                pos = int(enc["input_ids"].shape[1]) - 1
                x = hs.captures[f"{layer_key}.input"][0, pos]
                z = base["z"][f"{layer_key}.down_proj"][0, pos]
                imp = expert_importance_output(z[None, :], mlp_down_weight(model, layer_key),
                                               groups)[0]
                oracle_top = oracle_topk_from_importance(imp, kk)

                t0 = time.perf_counter()
                pred = routers[layer_key].predict(x, k=kk)
                dt_ms = (time.perf_counter() - t0) * 1e3
                rpred = random_router.predict(x, k=kk)

                m1 = router_recall_vs_oracle(pred.topk, oracle_top)
                m2 = router_recall_vs_oracle(rpred.topk, oracle_top)
                w.writerow([layer_key, prompt[:50], "centroid", kk,
                            f"{m1['recall']:.3f}", f"{m1['precision']:.3f}",
                            f"{m1['jaccard']:.3f}", f"{dt_ms:.3f}"])
                w.writerow([layer_key, prompt[:50], "random", kk,
                            f"{m2['recall']:.3f}", f"{m2['precision']:.3f}",
                            f"{m2['jaccard']:.3f}", ""])
                rows.append((m1, m2, dt_ms))
        f.flush()

    # summary + §80 overhead note
    cr = [r[0]["recall"] for r in rows]
    rr = [r[1]["recall"] for r in rows]
    lat = [r[2] for r in rows if r[2] > 0]
    print("\n=== Router vs Oracle (holdout) ===")
    print(f"centroid recall: {sum(cr)/len(cr):.3f}   random recall: {sum(rr)/len(rr):.3f}")
    print(f"router latency:  {sum(lat)/len(lat):.3f} ms (§80: must stay well under "
          f"the FFN time it saves)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
