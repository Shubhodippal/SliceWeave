"""Oracle sparsity experiment machinery (Blueprint §9-12)."""
from __future__ import annotations

from typing import Dict, List

import torch

from .hooks import FFNIntermediateCapture, FFNMaskedForward
from .micro_experts import (contiguous_partition, expert_importance_output,
                            expert_importance_z, keep_mask_from_topk)


def run_baseline(model, tokenizer, prompt: str, layer_keys: List[str], device=None) -> Dict:
    """Run the model once, capturing z for each requested layer + final logits."""
    capture = FFNIntermediateCapture()
    for key in layer_keys:
        ffn_ref = model.get_submodule(key)
        capture.attach(model, key, ffn_ref)

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc)
    capture.remove()
    return {"logits": out.logits[0, -1].float().cpu(), "z": dict(capture.captures)}


def oracle_single_layer(model, tokenizer, prompt: str, layer_key: str,
                        retain_fractions: List[float], group_size: int = 128,
                        score_mode: str = "output") -> List[Dict]:
    """Blueprint §9: for each retention fraction, mask low-ranked micro-experts
    in ONE layer at ONE position (last prompt token) and compare final logits.

    score_mode: 'z' (mean z^2) or 'output' (||W_down z_e||, more accurate).
    """
    ffn_ref = model.get_submodule(layer_key)
    inter = ffn_ref.down_proj.weight.shape[1]
    groups = contiguous_partition(inter, group_size)

    base = run_baseline(model, tokenizer, prompt, [layer_key])
    z_all = base["z"][f"{layer_key}.down_proj"]  # [1, seq, inter]
    z = z_all[0, -1]  # last prompt token
    base_logits = base["logits"]

    if score_mode == "z":
        importance = expert_importance_z(z, groups)[0]
    else:
        importance = expert_importance_output(z[None, :], ffn_ref.down_proj.weight, groups)[0]

    results = []
    with FFNMaskedForward(model) as masked:
        masked.attach(layer_key, ffn_ref)
        masked.set_mask(layer_key, torch.ones(inter, dtype=torch.bool, device=z.device))
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)

        for frac in retain_fractions:
            keep = keep_mask_from_topk(groups, importance, frac)
            masked.set_mask(layer_key, keep)
            with torch.no_grad():
                out = model(**enc)
            logits = out.logits[0, -1].float().cpu()
            results.append({
                "layer": layer_key,
                "retain_fraction": frac,
                "logits": logits,
                "baseline_logits": base_logits,
            })
    return results
