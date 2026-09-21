"""Phase 1 — Oracle sparsity machinery (§12-§14, Milestone 1).

The oracle deliberately "cheats": it runs the full FFN to obtain the true z,
ranks micro-experts by true contribution, then masks the rest and measures how
final logits degrade. It answers only:

    If an oracle knew which micro-experts mattered, how many are necessary?
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .hooks import FFNIntermediateCapture, FFNMaskedForward
from .micro_experts import (contiguous_partition, expert_importance_output,
                            expert_importance_z, keep_mask_from_topk)


def run_baseline(model, tokenizer, prompt: str, layer_keys: List[str],
                 capture_positions: Optional[List[int]] = None) -> Dict:
    """Run the model once, capturing z for each requested layer + final logits.

    capture_positions: which sequence positions to record z for (default: all).
    """
    capture = FFNIntermediateCapture()
    for key in layer_keys:
        capture.attach(model, key)

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc)
    capture.remove()
    z_all = dict(capture.captures)
    if capture_positions is not None:
        z_all = {k: v[:, capture_positions] for k, v in z_all.items()}
    return {"logits": out.logits[0, -1].float().cpu(), "z": z_all}


def oracle_single_layer(model, tokenizer, prompt: str, layer_key: str,
                        retain_fractions: List[float], group_size: int = 128,
                        score_mode: str = "output", position: int = -1) -> List[Dict]:
    """§12: for each retention fraction, mask low-ranked micro-experts of ONE
    layer at ONE position and compare final logits to the dense baseline.

    score_mode: 'z' (mean z²) or 'output' (||W_down z_e||, more accurate, §13).
    """
    from .model_utils import mlp_down_weight, mlp_intermediate_size
    ffn_ref = model.get_submodule(f"{layer_key}.mlp")
    inter = mlp_intermediate_size(model, layer_key)
    groups = contiguous_partition(inter, group_size)

    base = run_baseline(model, tokenizer, prompt, [layer_key])
    z_all = base["z"][f"{layer_key}.down_proj"]  # [1, seq, inter]
    z = z_all[0, position]                        # selected prompt token
    base_logits = base["logits"]

    if score_mode == "z":
        importance = expert_importance_z(z, groups)[0]
    else:
        # pin weight to the activation's device: for offloaded layers the
        # materialized weight may live on CPU while z is on GPU
        w = mlp_down_weight(model, layer_key, device=z.device)
        importance = expert_importance_output(z[None, :], w, groups)[0]

    results = []
    with FFNMaskedForward(model) as masked:
        masked.attach(layer_key, ffn_ref)
        masked.set_mask(layer_key, torch.ones(inter, dtype=torch.bool, device=z.device))
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        # mask ONLY the analyzed token's slot: the ranking comes from this
        # token's z, so it must be applied to exactly that token (§9/§20)
        pos_idx = position % enc["input_ids"].shape[1]

        for frac in retain_fractions:
            keep = keep_mask_from_topk(groups, importance, frac)
            masked.set_mask(layer_key, keep, positions=[pos_idx])
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


def oracle_importance(model, tokenizer, prompt: str, layer_key: str,
                      group_size: int = 128, score_mode: str = "output",
                      position: int = -1) -> torch.Tensor:
    """Return the oracle importance vector [E] for one layer/position.

    Used by clustering (§15) and router calibration (§18): the "ground truth"
    expert ranking the router must predict.
    """
    base = run_baseline(model, tokenizer, prompt, [layer_key])
    z = base["z"][f"{layer_key}.down_proj"][0, position]
    from .model_utils import mlp_down_weight, mlp_intermediate_size
    groups = contiguous_partition(mlp_intermediate_size(model, layer_key), group_size)
    if score_mode == "z":
        return expert_importance_z(z, groups)[0]
    w = mlp_down_weight(model, layer_key, device=z.device)
    return expert_importance_output(z[None, :], w, groups)[0]


def oracle_multi_token(model, tokenizer, prompt: str, layer_key: str,
                       retain_fractions: List[float], group_size: int = 128,
                       score_mode: str = "output",
                       max_positions: int = 16) -> List[Dict]:
    """Sample several prompt tokens with BATCHED per-position masks (§97 N).

    Instead of one forward per position (max_positions × fractions forwards),
    we replicate the prompt into a batch of B=max_positions rows. Row i has
    its mask applied ONLY at its own analyzed position — mathematically
    identical to B separate single-position forwards, but the weight
    streaming cost is paid once per fraction instead of B times. On the
    VRAM+RAM offload setup that dominates runtime, this is a ~B× speedup.

    Positions are spread across the prompt: logits at t predict token t+1
    under teacher forcing, and each masked row is compared to the dense
    baseline at the same position.

    Returns per-fraction dicts with metric lists across positions.
    """
    from .model_utils import mlp_down_weight, mlp_intermediate_size
    ffn_ref = model.get_submodule(f"{layer_key}.mlp")
    inter = mlp_intermediate_size(model, layer_key)
    groups = contiguous_partition(inter, group_size)

    base = run_baseline(model, tokenizer, prompt, [layer_key])
    z_all = base["z"][f"{layer_key}.down_proj"]      # [1, seq, inter]
    seq_len = z_all.shape[1]
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    # analyze a spread of teacher-forced positions
    step = max(1, seq_len // max_positions)
    positions = list(range(1, seq_len, step))[:max_positions]
    B = len(positions)

    # dense baseline logits for every row/position
    with torch.no_grad():
        full = model(**enc)
    full_logits = full.logits[0].float().cpu()       # [seq, vocab]

    # rank experts per analyzed position once (importance is frac-independent;
    # the keep mask is rebuilt per fraction from it)
    w = mlp_down_weight(model, layer_key, device=z_all.device)
    per_pos_imp: Dict[int, torch.Tensor] = {}
    for t in positions:
        z = z_all[0, t]
        if score_mode == "z":
            imp = expert_importance_z(z, groups)[0]
        else:
            imp = expert_importance_output(z[None, :], w, groups)[0]
        per_pos_imp[t] = imp

    # build batch: row i = same prompt, analyzed position = positions[i]
    input_ids = enc["input_ids"].expand(B, -1).contiguous()
    attn = enc["attention_mask"].expand(B, -1).contiguous()
    batch_enc = {"input_ids": input_ids, "attention_mask": attn}
    # position_ids shifted so no re-computation of rotary caches is needed
    if hasattr(enc, "position_ids") and enc.get("position_ids") is not None:
        batch_enc["position_ids"] = enc["position_ids"].expand(B, -1).contiguous()

    from .metrics import compute_all
    out_by_frac: Dict[float, Dict] = {
        f: {"cosines": [], "top1": [], "kl": []} for f in retain_fractions}
    with FFNMaskedForward(model) as masked:
        masked.attach(layer_key, ffn_ref)
        for frac in retain_fractions:
            if frac >= 1.0:
                # nothing masked; dense == baseline, record perfect metrics
                for t in positions:
                    m = compute_all(full_logits[t], full_logits[t])
                    out_by_frac[frac]["cosines"].append(m["cosine"])
                    out_by_frac[frac]["top1"].append(1)
                    out_by_frac[frac]["kl"].append(0.0)
                continue
            # stack per-row keep masks: [B, inter]; hook applies row-wise
            keeps = torch.stack([keep_mask_from_topk(groups, per_pos_imp[t], frac)
                                 for t in positions]).to(z_all.device)
            masked.set_batch_masks(layer_key, keeps, positions)
            with torch.no_grad():
                out = model(**batch_enc)
            # row i's logits at its own position t_i
            for i, t in enumerate(positions):
                logits_t = out.logits[i, t].float().cpu()
                m = compute_all(logits_t, full_logits[t])
                out_by_frac[frac]["cosines"].append(m["cosine"])
                out_by_frac[frac]["top1"].append(m["top1_match"])
                out_by_frac[frac]["kl"].append(m["kl"])
    return [{"layer": layer_key, "retain_fraction": f, **v}
            for f, v in out_by_frac.items()]
