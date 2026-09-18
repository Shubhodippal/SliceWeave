"""Logit-quality metrics (Blueprint §10)."""
from __future__ import annotations

import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (b.norm() + 1e-12))


def top1_agreement(a: torch.Tensor, b: torch.Tensor) -> bool:
    return int(a.argmax()) == int(b.argmax())


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> float:
    ka = set(torch.topk(a, k).indices.tolist())
    kb = set(torch.topk(b, k).indices.tolist())
    return len(ka & kb) / k


def kl_divergence(a: torch.Tensor, b: torch.Tensor, temperature: float = 1.0) -> float:
    """KL(P_full || P_sparse); a = sparse, b = full (baseline)."""
    pa = torch.softmax(a / temperature, dim=-1)
    pb = torch.softmax(b / temperature, dim=-1)
    return float(torch.nn.functional.kl_div(pa.log(), pb, reduction="batchmean"))


def compute_all(sparse_logits: torch.Tensor, full_logits: torch.Tensor) -> dict:
    return {
        "cosine": cosine_similarity(sparse_logits, full_logits),
        "rel_l2": relative_l2(sparse_logits, full_logits),
        "top1_match": int(top1_agreement(sparse_logits, full_logits)),
        "top10_overlap": topk_overlap(sparse_logits, full_logits, 10),
        "top50_overlap": topk_overlap(sparse_logits, full_logits, 50),
        "kl": kl_divergence(sparse_logits, full_logits),
    }
