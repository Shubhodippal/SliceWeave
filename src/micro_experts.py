"""Micro-expert partitioning of FFN intermediates (Blueprint §5, §6, §16)."""
from __future__ import annotations

import numpy as np
import torch


def contiguous_partition(intermediate_size: int, group_size: int = 128) -> np.ndarray:
    """Return array of shape [E] of lists: neuron index groups, contiguous.

    Blueprint §5: e.g. 17408 / 128 = 136 micro-experts per layer.
    """
    if intermediate_size % group_size != 0:
        # last group absorbs the remainder
        n_full = intermediate_size // group_size
        edges = [i * group_size for i in range(n_full)] + [intermediate_size]
        groups = [list(range(edges[i], edges[i + 1])) for i in range(len(edges) - 1)]
    else:
        groups = [list(range(i, i + group_size)) for i in range(0, intermediate_size, group_size)]
    return groups


def groups_from_assignment(assignment: np.ndarray) -> list:
    """Build neuron groups from a per-neuron cluster assignment (Blueprint §15)."""
    assignment = np.asarray(assignment)
    order = np.argsort(assignment, kind="stable")
    groups = []
    vals = assignment[order]
    start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[start]:
            groups.append(order[start:i].tolist())
            start = i
    return groups


def expert_importance_z(z: torch.Tensor, groups: list) -> torch.Tensor:
    """Score experts by mean squared activation (Blueprint §9 first score).

    z: [batch, seq, inter] or [inter]
    returns: [E] importance per group
    """
    z2 = z.float() ** 2
    if z2.dim() == 1:
        z2 = z2.unsqueeze(0)
    # [batch, seq, inter] -> mean over batch/seq? No: importance is per-token.
    # Caller should slice a single token first for per-token oracle ranking.
    scores = []
    for g in groups:
        scores.append(z2[..., g].mean(dim=-1))
    return torch.stack(scores, dim=-1)  # [..., E]


def expert_importance_output(z: torch.Tensor, down_proj_weight: torch.Tensor, groups: list) -> torch.Tensor:
    """True output-contribution score (Blueprint §9 improved score).

    score_e = || W_down[:, I_e] z[I_e] ||
    z: [..., inter]; down_proj_weight: [hidden, inter]
    returns: [..., E]
    """
    contrib = z.float() @ down_proj_weight.float().T  # [..., hidden]
    out_norm = contrib.norm(dim=-1)  # scalar per token — not per-expert
    # Per-expert contribution norm: compute each expert's output slice norm.
    scores = []
    W = down_proj_weight.float().T  # [inter, hidden]
    for g in groups:
        y_e = z[..., g].float() @ W[g, :]  # [..., hidden]
        scores.append(y_e.norm(dim=-1))
    return torch.stack(scores, dim=-1)  # [..., E]


def keep_mask_from_topk(groups: list, importance: torch.Tensor, retain_fraction: float) -> torch.Tensor:
    """Boolean keep-mask over the intermediate dim keeping top experts.

    importance: [E] (single token). Returns [inter] bool mask.
    """
    E = importance.shape[-1]
    k = max(1, int(round(E * retain_fraction)))
    top = torch.topk(importance, k, dim=-1).indices
    keep = torch.zeros(E, dtype=torch.bool, device=importance.device)
    keep[top] = True
    full = torch.zeros(len(max(groups, key=len)) * 0 + sum(len(g) for g in groups),
                       dtype=torch.bool, device=importance.device)
    for ei, g in enumerate(groups):
        if keep[ei]:
            full[torch.tensor(g, device=importance.device, dtype=torch.long)] = True
    return full


def reorder_permutation(groups: list, intermediate_size: int) -> torch.Tensor:
    """Permutation vector placing expert groups contiguously (Blueprint §16).

    Apply to: rows of gate_proj, rows of up_proj, columns of down_proj.
    """
    perm = torch.cat([torch.tensor(g, dtype=torch.long) for g in groups])
    assert perm.numel() == intermediate_size
    return perm


def verify_reorder_equivalence(ffn, perm: torch.Tensor, hidden: torch.Tensor) -> bool:
    """Check permuted FFN == original FFN exactly (Blueprint §16)."""
    gate = ffn.gate_proj.weight  # [inter, hidden]
    up = ffn.up_proj.weight
    down = ffn.down_proj.weight  # [hidden, inter]

    z = torch.nn.functional.silu(hidden @ gate.T) * (hidden @ up.T)
    y_dense = z @ down.T

    zp = z[:, perm]
    y_perm = zp @ down[:, perm].T
    return torch.allclose(y_dense, y_perm, atol=1e-4, rtol=1e-4)
