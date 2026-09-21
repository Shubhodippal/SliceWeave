"""Debug: batched per-row masking equivalence check (CUDA_LAUNCH_BLOCKING=1)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from src.hooks import FFNMaskedForward  # noqa: E402
from src.model_utils import load_model_and_tokenizer  # noqa: E402


def main():
    cfg = AutoConfig.for_model(
        "qwen3", hidden_size=64, intermediate_size=256, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=256)
    d = tempfile.mkdtemp()
    AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16).save_pretrained(d)
    vocab = {"<pad>": 0, "<unk>": 1}
    for i in range(32, 127):
        vocab[chr(i)] = len(vocab)
    wt = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    wt.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=wt, pad_token="<pad>",
                            unk_token="<unk>").save_pretrained(d)
    model, tok = load_model_and_tokenizer(d, "bfloat16")
    dev = next(model.parameters()).device
    print("device:", dev)

    text = "the quick brown fox jumps over the lazy dog again and again"
    enc = tok(text, return_tensors="pt")
    print("tokenizer keys:", list(enc.keys()))
    print("input_ids:", enc["input_ids"])
    print("max id:", int(enc["input_ids"].max()), "vocab:", model.config.vocab_size)
    enc = enc.to(dev)
    seq = enc["input_ids"].shape[1]
    positions = [2, 5, 8]
    B = len(positions)
    assert max(positions) < seq, f"position beyond seq {seq}"

    torch.manual_seed(0)
    keep = torch.rand(256) < 0.4

    layer_key = "model.layers.0"
    mlp = model.get_submodule(f"{layer_key}.mlp")

    print("--- step 1: plain forward, no hooks ---")
    with torch.no_grad():
        base = model(**enc)
    print("base ok, logits:", tuple(base.logits.shape))

    print("--- step 2: sequential single-row forwards ---")
    ref = []
    for i, t in enumerate(positions):
        with FFNMaskedForward(model) as m:
            m.attach(layer_key, mlp)
            m.set_mask(layer_key, keep, positions=[t])
            with torch.no_grad():
                o = model(**enc)
        ref.append(o.logits[0, t].float().cpu())
        print(f"  row {i} pos {t} ok")

    print("--- step 3: batched multi-row forward ---")
    # mirror oracle.py: expand the prompt to B rows, one analyzed position per row
    batch_enc = {
        "input_ids": enc["input_ids"].expand(B, -1).contiguous(),
        "attention_mask": enc["attention_mask"].expand(B, -1).contiguous(),
    }
    with FFNMaskedForward(model) as m:
        m.attach(layer_key, mlp)
        m.set_batch_masks(layer_key, keep.unsqueeze(0).repeat(B, 1), positions)
        with torch.no_grad():
            o = model(**batch_enc)
    bat = [o.logits[i, t].float().cpu() for i, t in enumerate(positions)]

    dmax = max((x - y).abs().max().item() for x, y in zip(ref, bat))
    cos = min(torch.nn.functional.cosine_similarity(x, y, dim=0).item()
              for x, y in zip(ref, bat))
    # bf16 batched GEMMs reduce in different order than batch-1 GEMMs, so tiny
    # divergence is expected; what matters is cosine ~1 and rel diff ~ bf16 eps
    denom = max(x.abs().max().item() for x in ref)
    print(f"max |diff|: {dmax:.2e}   max |logit|: {denom:.2f}   "
          f"min cosine: {cos:.6f}")
    verdict = "EQUIVALENT" if (cos > 0.9999 and dmax / denom < 0.01) else "MISMATCH"
    print(f"-> {verdict}")


if __name__ == "__main__":
    main()
