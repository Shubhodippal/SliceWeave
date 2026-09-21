"""Micro-expert partitioning and exact FFN decomposition (§9, §10, §11, Milestone 0).

The core identity (gated FFN, intermediate dim I partitioned into groups I_e):

    z = SiLU(W_gate x) * (W_up x)
    y = sum_e  W_down[:, I_e] z[I_e]   ==   W_down z      (exact in real arithmetic)

The approximation starts only when some experts are skipped (§8.1, §40 Milestone 0).
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- partitioning
def contiguous_partition(intermediate_size: int, group_size: int = 128) -> List[List[int]]:
    """Partition neuron indices 0..I-1 into contiguous groups of `group_size`.

    §10 granularity knob: 64 / 128 / 256 / 512 neurons per micro-expert.
    The last group absorbs the remainder when I % group_size != 0.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    groups: List[List[int]] = []
    start = 0
    while start < intermediate_size:
        end = min(start + group_size, intermediate_size)
        groups.append(list(range(start, end)))
        start = end
    return groups


def groups_from_assignment(assignment: np.ndarray) -> List[List[int]]:
    """Build neuron groups from a per-neuron cluster assignment (§11 clustering)."""
    assignment = np.asarray(assignment)
    order = np.argsort(assignment, kind="stable")
    groups: List[List[int]] = []
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or assignment[order[i]] != assignment[order[start]]:
            groups.append(order[start:i].tolist())
            start = i
    return groups


def groups_to_assignment(groups: List[List[int]]) -> np.ndarray:
    """Inverse of groups_from_assignment: [I] per-neuron expert id."""
    a = np.zeros(sum(len(g) for g in groups), dtype=np.int64)
    for e, g in enumerate(groups):
        a[g] = e
    return a


def expert_sizes(groups: List[List[int]]) -> np.ndarray:
    return np.array([len(g) for g in groups], dtype=np.int64)


def n_experts(intermediate_size: int, group_size: int) -> int:
    return len(contiguous_partition(intermediate_size, group_size))


# ---------------------------------------------------------------- importance
def expert_importance_z(z: torch.Tensor, groups: List[List[int]]) -> torch.Tensor:
    """§13 first score: score_e = mean(z_e^2).  z: [..., inter] -> [..., E]."""
    z2 = z.float() ** 2
    return torch.stack([z2[..., g].mean(dim=-1) for g in groups], dim=-1)


def expert_importance_output(z: torch.Tensor, down_proj_weight: torch.Tensor,
                             groups: List[List[int]]) -> torch.Tensor:
    """§13 improved oracle score: score_e = || W_down[:, I_e] z[I_e] ||₂.

    z: [..., inter]; down_proj_weight: [hidden, inter] (as stored by nn.Linear).
    Returns [..., E] — the actual norm of each expert's output contribution.
    """
    W = down_proj_weight.float().T                       # [inter, hidden]
    scores = []
    for g in groups:
        y_e = z[..., g].float() @ W[g, :]                # [..., hidden]
        scores.append(y_e.norm(dim=-1))
    return torch.stack(scores, dim=-1)


def expert_outputs(z: torch.Tensor, down_proj_weight: torch.Tensor,
                   groups: List[List[int]], stacked: bool = False):
    """Per-expert output vectors y_e = W_down[:, I_e] z[I_e].

    stacked=False returns cheap statistics ([..., E] norms, [..., E] mean|y_e|);
    stacked=True materializes [..., E, hidden] per-expert outputs (memory heavy,
    used by residual studies and quality-guard calibration).
    """
    W = down_proj_weight.float().T                       # [inter, hidden]
    if stacked:
        return torch.stack([z[..., g].float() @ W[g, :] for g in groups], dim=-2)
    norms, means = [], []
    for g in groups:
        y_e = z[..., g].float() @ W[g, :]
        norms.append(y_e.norm(dim=-1))
        means.append(y_e.abs().mean(dim=-1))
    return torch.stack(norms, dim=-1), torch.stack(means, dim=-1)


# ---------------------------------------------------------------- masking
def keep_mask_from_topk(groups: List[List[int]], importance: torch.Tensor,
                        retain_fraction: float) -> torch.Tensor:
    """Boolean keep-mask over the intermediate dim, keeping the top experts.

    importance: [E] for a single token. Returns [inter] bool (True = keep).
    """
    importance = importance.reshape(-1)
    E = importance.shape[0]
    k = max(1, min(E, int(round(E * retain_fraction))))
    keep_e = torch.zeros(E, dtype=torch.bool, device=importance.device)
    keep_e[torch.topk(importance, k).indices] = True
    keep = torch.zeros(sum(len(g) for g in groups), dtype=torch.bool,
                       device=importance.device)
    idx = torch.tensor([i for e, g in enumerate(groups) if keep_e[e] for i in g],
                       dtype=torch.long, device=importance.device)
    if idx.numel():
        keep[idx] = True
    return keep


def keep_mask_from_sets(groups: List[List[int]], active_experts: Iterable[int],
                        device=None) -> torch.Tensor:
    """Mask from an explicit set/tensor of active expert ids."""
    active = {int(e) for e in active_experts}
    keep = torch.zeros(sum(len(g) for g in groups), dtype=torch.bool,
                       device=device)
    idx = torch.tensor([i for e, g in enumerate(groups) if e in active for i in g],
                       dtype=torch.long, device=keep.device)
    if idx.numel():
        keep[idx] = True
    return keep


# ---------------------------------------------------------------- exactness
def _dense_parts(ffn):
    return (ffn.gate_proj.weight.float(), ffn.up_proj.weight.float(),
            ffn.down_proj.weight.float())


def intermediate_z(ffn, hidden: torch.Tensor) -> torch.Tensor:
    """z = SiLU(W_gate x) * (W_up x) for a Qwen-style gated FFN."""
    gate_w, up_w, _ = _dense_parts(ffn)
    hidden = hidden.to(gate_w.device)
    return F.silu(hidden @ gate_w.T) * (hidden @ up_w.T)


def exact_decomposition_check(ffn, hidden: torch.Tensor, groups: List[List[int]],
                              atol: float = 1e-4, rtol: float = 1e-4) -> dict:
    """Milestone 0 (§8.1): two exactness properties on one real FFN.

    1. sum of ALL expert outputs == dense FFN output
    2. applying the same permutation to gate/up rows + down columns preserves
       the dense computation
    hidden: [batch, seq, hidden] or [hidden].
    """
    if hidden.dim() == 1:
        hidden = hidden.unsqueeze(0)
    gate_w, up_w, down_w = _dense_parts(ffn)
    z = intermediate_z(ffn, hidden)
    y_dense = z @ down_w.T

    y_sum = torch.zeros_like(y_dense)
    for g in groups:
        idx = torch.tensor(g, dtype=torch.long, device=z.device)
        y_sum = y_sum + z[..., idx] @ down_w[:, idx].T

    perm = reorder_permutation(groups, z.shape[-1]).to(z.device)
    y_perm = z[..., perm] @ down_w[:, perm].T

    denom = y_dense.norm().item() + 1e-12
    return {
        "sum_equals_dense": bool(torch.allclose(y_sum, y_dense, atol=atol, rtol=rtol)),
        "permuted_equals_dense": bool(torch.allclose(y_perm, y_dense, atol=atol, rtol=rtol)),
        "rel_err_sum": (y_sum - y_dense).norm().item() / denom,
        "rel_err_perm": (y_perm - y_dense).norm().item() / denom,
    }


def reorder_permutation(groups: List[List[int]], intermediate_size: int) -> torch.Tensor:
    """Permutation placing expert groups contiguously (§11).

    Apply consistently to: rows of gate_proj, rows of up_proj, columns of down_proj.
    """
    perm = torch.cat([torch.tensor(g, dtype=torch.long) for g in groups])
    assert perm.numel() == intermediate_size
    return perm


def apply_reorder_inplace_(ffn, groups: List[List[int]]) -> None:
    """Physically permute FFN weights so expert `e` occupies rows/cols [e]."""
    inter = ffn.down_proj.weight.shape[1]
    perm = reorder_permutation(groups, inter).to(ffn.gate_proj.weight.device)
    with torch.no_grad():
        ffn.gate_proj.weight.copy_(ffn.gate_proj.weight[perm])
        ffn.up_proj.weight.copy_(ffn.up_proj.weight[perm])
        ffn.down_proj.weight.copy_(ffn.down_proj.weight[:, perm])


def verify_reorder_equivalence(ffn, hidden: torch.Tensor, groups: List[List[int]],
                               atol: float = 1e-4, rtol: float = 1e-4) -> bool:
    """Physically permuted weights produce the original dense output (§11)."""
    dev = ffn.down_proj.weight.device
    perm = reorder_permutation(groups, ffn.down_proj.weight.shape[1]).to(dev)
    gate = ffn.gate_proj.weight.float()[perm]
    up = ffn.up_proj.weight.float()[perm]
    down = ffn.down_proj.weight.float()[:, perm]
    hidden = hidden.to(dev)
    z = F.silu(hidden @ gate.T) * (hidden @ up.T)
    y_perm = z @ down.T
    y0 = intermediate_z(ffn, hidden) @ ffn.down_proj.weight.float().T
    return bool(torch.allclose(y0, y_perm, atol=atol, rtol=rtol))
