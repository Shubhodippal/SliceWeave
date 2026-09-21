"""Experiment 14 — Scaling driver for Stage 2/3 models (§6, Milestone 10).

Same oracle + router pipeline as experiments 01/05, parameterized for the
~9B intermediate model and the 27B primary target. On the local laptop the 27B
needs the 4-bit fallback or Kaggle (see notebooks/kaggle_27b.ipynb).

Usage:
    # Stage 2 stress test
    python experiments/14_scale_model.py --model Qwen/Qwen3-9B --layers 6

    # Stage 3 primary target (Kaggle or local 4-bit)
    python experiments/14_scale_model.py --model Qwen/Qwen3-8B-27B --load-4bit
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
from src.model_utils import describe_model, ffn_layer_names, find_ffn_modules, load_model_and_tokenizer  # noqa: E402
from src.oracle import oracle_single_layer  # noqa: E402

RETAIN_FRACTIONS = [0.5, 0.3, 0.2, 0.1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id of the stage-2/3 model")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--load-4bit", action="store_true",
                    help="bitsandbytes NF4 (laptop fallback for the 27B)")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--out", default="results/csv/scale_model.csv")
    args = ap.parse_args()

    kwargs = {}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4")

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    print(f"params={info['params']/1e9:.2f}B layers={info['layers']} "
          f"hidden={info['hidden_size']} inter={info['intermediate_size']}")

    ffn_modules = find_ffn_modules(model)
    layer_map = ffn_layer_names(ffn_modules)
    layer_keys = sorted(layer_map.keys(), key=lambda s: int(s.rsplit(".", 1)[-1]))
    idxs = [int(round(info["layers"] * f)) for f in (0.2, 0.4, 0.5, 0.6, 0.8)]
    chosen = [layer_keys[i] for i in dict.fromkeys(idxs)][: args.layers]
    print("Testing layers:", chosen)

    prompts = [(c, p) for c, ps in CATEGORIES.items() for p in ps[:2]]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "prompt", "layer", "retain_fraction",
                    "cosine", "rel_l2", "top1_match", "top10_overlap", "kl"])
        for layer in chosen:
            for cat, prompt in prompts:
                results = oracle_single_layer(model, tokenizer, prompt, layer,
                                              RETAIN_FRACTIONS,
                                              group_size=args.group_size)
                for r in results:
                    m = compute_all(r["logits"], r["baseline_logits"])
                    w.writerow([cat, prompt[:60], layer, r["retain_fraction"],
                                f"{m['cosine']:.4f}", f"{m['rel_l2']:.4f}",
                                m["top1_match"], f"{m['top10_overlap']:.2f}",
                                f"{m['kl']:.4f}"])
            f.flush()
            print(f"layer {layer} done")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
