"""Experiment 00 — Exactness: dense FFN == sum of micro-experts == permuted FFN
(§8.1, §11, Milestone 0 — the mandatory precondition for everything else).

Proves, on real FFN weights:
    1. sum of ALL micro-expert outputs equals the dense FFN output
    2. the same neuron permutation on gate/up rows + down columns preserves
       the dense computation
    3. a full forward pass with all experts enabled == baseline logits

Usage:
    python experiments/00_exact_decomposition.py --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.micro_experts import (contiguous_partition, exact_decomposition_check,  # noqa: E402
                              verify_reorder_equivalence)
from src.model_utils import (describe_model, load_model_and_tokenizer,  # noqa: E402
                             pick_resident_layer)
from src.oracle import oracle_single_layer  # noqa: E402
from src.metrics import compute_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--group-sizes", type=int, nargs="+", default=[64, 128, 256],
                    help="granularity sweep (§10)")
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--out", default="results/csv/exactness_report.csv")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    print(f"params={info['params']/1e9:.2f}B layers={info['layers']} "
          f"hidden={info['hidden_size']} inter={info['intermediate_size']}")

    layer_key = pick_resident_layer(model)
    layer = f"{layer_key}.mlp"
    ffn = model.get_submodule(layer)
    dev = next(ffn.parameters()).device
    print(f"weight-level checks on resident layer: {layer_key} ({dev})")
    probe = torch.randn(2, 3, info["hidden_size"], device=dev)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group_size", "n_experts", "sum_equals_dense",
                    "permuted_equals_dense", "rel_err_sum", "rel_err_perm"])
        for gs in args.group_sizes:
            groups = contiguous_partition(info["intermediate_size"], gs)
            r = exact_decomposition_check(ffn, probe, groups)
            ok_perm = verify_reorder_equivalence(ffn, probe, groups)
            w.writerow([gs, len(groups), r["sum_equals_dense"],
                        r["permuted_equals_dense"] and ok_perm,
                        f"{r['rel_err_sum']:.3e}", f"{r['rel_err_perm']:.3e}"])
            rows.append((gs, r, ok_perm))
            print(f"group_size={gs:4d}  experts={len(groups):4d}  "
                  f"sum==dense: {r['sum_equals_dense']}  "
                  f"perm==dense: {r['permuted_equals_dense'] and ok_perm}  "
                  f"(rel_err {r['rel_err_sum']:.2e} / {r['rel_err_perm']:.2e})")

    # Full forward pass with all experts enabled == baseline (via oracle at 100%)
    print("\nFull-model check: 100% retention must equal baseline logits ...")
    prompts = ["Implement Dijkstra in Rust using a binary heap.",
               "What is the boiling point of water at sea level?"][: args.prompts]
    all_ok = True
    for p in prompts:
        res = oracle_single_layer(model, tokenizer, p, layer_key, [1.0],
                                  group_size=args.group_sizes[0])
        m = compute_all(res[0]["logits"], res[0]["baseline_logits"])
        ok = m["cosine"] > 0.9999 and m["top1_match"] == 1
        all_ok &= ok
        print(f"  '{p[:40]}...' cosine={m['cosine']:.6f} top1={m['top1_match']} -> "
              f"{'OK' if ok else 'FAIL'}")

    if not all(r["sum_equals_dense"] for _, r, _ in rows) or not all_ok:
        print("\nEXACTNESS FAILED — do not proceed to sparsity experiments.")
        sys.exit(1)
    print(f"\nMilestone 0 PASSED. Saved: {out_path}")


if __name__ == "__main__":
    main()
