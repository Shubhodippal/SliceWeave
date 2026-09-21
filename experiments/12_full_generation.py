"""Experiment 12 — Full Generation Agreement (§51, Milestone 6/7 gate).

Generate with the full dense model and with a routed/masked model over whole
sequences. Measures tokens-until-divergence plus final-sequence metrics, and
APR over the generation (§37).

Usage:
    python experiments/12_full_generation.py --model Qwen/Qwen3-4B --max-new 64
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import CATEGORIES  # noqa: E402
from src.hooks import FFNMaskedForward  # noqa: E402
from src.metrics import apr, compute_all, tokens_until_divergence  # noqa: E402
from src.micro_experts import (contiguous_partition,  # noqa: E402
                              expert_importance_output, keep_mask_from_topk)
from src.model_utils import (describe_model, get_mlp, load_model_and_tokenizer,  # noqa: E402
                             mlp_intermediate_size)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new: int) -> list:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return out[0, enc["input_ids"].shape[1]:].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.5, 0.3, 0.2])
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--out", default="results/csv/generation_agreement.csv")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    info = describe_model(model)
    li = args.layer if args.layer is not None else info["layers"] // 2
    layer_key = f"model.layers.{li}"
    mlp = get_mlp(model, layer_key)
    groups = contiguous_partition(mlp_intermediate_size(model, layer_key), args.group_size)
    E = len(groups)

    prompts = [(c, ps[0]) for c, ps in CATEGORIES.items()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "fraction", "divergence_token", "gen_len",
                    "cosine_first_step", "top1_first_step"])
        for cat, prompt in prompts:
            # dense generation
            dense_ids = generate(model, tokenizer, prompt, args.max_new)

            # first-step logits + oracle ranking at the last prompt token
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            base = torch.no_grad()(lambda: model(**enc))()
            base_logits = base.logits[0, -1].float().cpu()
            cap = torch.no_grad
            from src.hooks import FFNIntermediateCapture
            caph = FFNIntermediateCapture()
            caph.attach(model, layer_key)
            model(**enc)
            caph.remove()
            z = caph.captures[f"{layer_key}.down_proj"][0, -1]
            imp = expert_importance_output(z[None, :], mlp.down_proj.weight, groups)[0]

            for frac in args.fractions:
                keep = keep_mask_from_topk(groups, imp, frac)
                with FFNMaskedForward(model) as masked:
                    masked.attach(layer_key, mlp)
                    masked.set_mask(layer_key, keep)
                    sparse_ids = generate(model, tokenizer, prompt, args.max_new)
                    with torch.no_grad():
                        first = model(**enc)
                    sp_logits = first.logits[0, -1].float().cpu()
                m = compute_all(sp_logits, base_logits)
                div = tokens_until_divergence(sparse_ids, dense_ids)
                w.writerow([cat, frac, div, len(dense_ids),
                            f"{m['cosine']:.4f}", m["top1_match"]])
                a = apr(frac, 1)
                print(f"[{cat}] frac={frac:.2f} APR={a:.2f} divergence@{div}/{len(dense_ids)} "
                      f"cos={m['cosine']:.4f}")
                f.flush()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
