"""Quality metrics (§10) + efficiency metrics (§37-§43)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch


# ---------------------------------------------------------------- logit quality
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0))


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / (b.norm() + 1e-12))


def top1_agreement(a: torch.Tensor, b: torch.Tensor) -> bool:
    return int(a.argmax()) == int(b.argmax())


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> float:
    ka = set(torch.topk(a, k).indices.tolist())
    kb = set(torch.topk(b, k).indices.tolist())
    return len(ka & kb) / k


def kl_divergence(a: torch.Tensor, b: torch.Tensor, temperature: float = 1.0) -> float:
    """KL(P_sparse || P_full); a = sparse, b = full (baseline)."""
    pa = torch.softmax(a.float() / temperature, dim=-1)
    pb = torch.softmax(b.float() / temperature, dim=-1)
    return float(torch.nn.functional.kl_div(pa.clamp_min(1e-12).log(), pb,
                                            reduction="batchmean"))


def tokens_until_divergence(gen_sparse: Sequence[int], gen_full: Sequence[int]) -> int:
    """§51: number of tokens before the sparse generation first diverges."""
    n = min(len(gen_sparse), len(gen_full))
    for i in range(n):
        if gen_sparse[i] != gen_full[i]:
            return i
    return n


def compute_all(sparse_logits: torch.Tensor, full_logits: torch.Tensor) -> dict:
    return {
        "cosine": cosine_similarity(sparse_logits, full_logits),
        "rel_l2": relative_l2(sparse_logits, full_logits),
        "top1_match": int(top1_agreement(sparse_logits, full_logits)),
        "top10_overlap": topk_overlap(sparse_logits, full_logits, 10),
        "top50_overlap": topk_overlap(sparse_logits, full_logits, 50),
        "kl": kl_divergence(sparse_logits, full_logits),
    }


# ---------------------------------------------------------------- efficiency
def apr(executed_params: int, total_params: int) -> float:
    """§37 Active Parameter Ratio = executed/token ÷ total."""
    return executed_params / max(total_params, 1)


def apr_distribution(executed_per_token: Iterable[int], total_params: int):
    """§37: mean/median/P90/P99 APR over a generation."""
    xs = sorted(executed_per_token)
    if not xs:
        return {}
    def q(p):
        return xs[min(len(xs) - 1, int(p * len(xs)))]
    return {
        "mean": sum(xs) / len(xs) / total_params,
        "median": q(0.50) / total_params,
        "p90": q(0.90) / total_params,
        "p99": q(0.99) / total_params,
    }


def cpr(candidate_params: int, total_params: int) -> float:
    """§38 Candidate Parameter Ratio = request candidate pool ÷ total."""
    return candidate_params / max(total_params, 1)


def mtr(bytes_sparse: Dict[str, int], bytes_full: Dict[str, int]) -> float:
    """§39 Memory Traffic Ratio = bytes moved sparse / bytes moved dense.

    Each arg maps tier -> bytes, tiers like {"ssd":…, "ram":…, "vram":…}.
    """
    num = sum(bytes_sparse.values())
    den = sum(bytes_full.values())
    return num / max(den, 1)


def mtr_by_tier(bytes_sparse: Dict[str, int], bytes_full: Dict[str, int]) -> Dict[str, float]:
    """§39 per-tier MTR (SSD→RAM / RAM→VRAM / RAM read / VRAM read)."""
    return {k: bytes_sparse.get(k, 0) / max(bytes_full.get(k, 1), 1)
            for k in set(bytes_sparse) | set(bytes_full)}


def qat(tokens_per_sec: float, quality_retention: float) -> float:
    """§40 Quality-Adjusted Throughput = tok/s × quality retention ∈ [0,1]."""
    return tokens_per_sec * quality_retention


def recovery_rate(expansions_per_token: Sequence[int]) -> Dict[str, float]:
    """§41 Recovery Rate: fraction of tokens needing 0/1/2/… expansions.

    expansions_per_token[i] = number of extra passes beyond the initial set.
    """
    n = len(expansions_per_token)
    if n == 0:
        return {}
    out: Dict[str, float] = {}
    for lvl in sorted(set(expansions_per_token)):
        out[f"expansions_{lvl}"] = sum(1 for e in expansions_per_token if e == lvl) / n
    out["ever_expanded"] = sum(1 for e in expansions_per_token if e > 0) / n
    return out


def prefetch_metrics(predicted: set, actually_needed: set) -> Dict[str, float]:
    """§42 prefetch precision/recall + wasted fraction.

    predicted       : expert ids prefetched ahead of demand
    actually_needed : expert ids that were subsequently demanded
    """
    tp = len(predicted & actually_needed)
    precision = tp / len(predicted) if predicted else 1.0
    recall = tp / len(actually_needed) if actually_needed else 1.0
    wasted = 1.0 - precision
    return {"precision": precision, "recall": recall, "wasted_fraction": wasted}


def expert_reuse_distance(uses: Sequence[int]) -> List[int]:
    """§43: tokens/layers between repeated uses of the same block.

    `uses` is an ordered stream of expert ids; returns the list of gaps between
    consecutive uses of the same id (empty if nothing repeats).
    """
    last: Dict[int, int] = {}
    gaps: List[int] = []
    for t, e in enumerate(uses):
        if e in last:
            gaps.append(t - last[e])
        last[e] = t
    return gaps


def quality_retention(sparse_score: float, full_score: float) -> float:
    """§34: benchmark retention = sparse score / full score."""
    return sparse_score / max(full_score, 1e-12)


def summarize(rows: List[Dict], key: str) -> Dict[str, float]:
    """Mean/std/min/max of one numeric field over result rows."""
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not vals:
        return {}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    sv = sorted(vals)
    return {
        "mean": mean,
        "std": var ** 0.5,
        "min": sv[0],
        "max": sv[-1],
        "median": sv[len(sv) // 2],
        "n": len(vals),
    }
