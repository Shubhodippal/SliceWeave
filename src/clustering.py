"""Phase 3 — activation-based neuron clustering + physical reordering (§11, §15,
Milestones 2-3).

Group neurons that behave alike (co-activate over calibration tokens), so that
micro-experts become semantically meaningful instead of arbitrary contiguous
slices. The permutation preserves dense computation exactly, so clustering is
loss-free by construction (verify before sparsifying, Milestone 0 rule).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from .micro_experts import (apply_reorder_inplace_, expert_importance_output,
                            expert_sizes, groups_from_assignment,
                            groups_to_assignment, verify_reorder_equivalence)


def neuron_signatures(z_matrix: np.ndarray) -> np.ndarray:
    """Per-neuron activation signatures (§15): [N_tokens, I] -> [I, N_tokens].

    z_matrix rows are calibration tokens' intermediate vectors z.
    """
    return np.asarray(z_matrix, dtype=np.float32).T


def cluster_neurons(z_matrix: np.ndarray, n_clusters: Optional[int] = None,
                    method: str = "kmeans", random_state: int = 0) -> np.ndarray:
    """Cluster neurons by activation behavior. Returns [I] assignment vector.

    method:
        "kmeans" - torch minibatch k-means over z-signatures (no sklearn dep)
        "corr"   - scipy average-link hierarchical grouping (§15 alternative)
    n_clusters None -> auto: ~128 neurons per cluster (§10 default granularity).
    """
    sig = neuron_signatures(z_matrix)
    I = sig.shape[0]
    if n_clusters is None:
        n_clusters = max(1, I // 128)
    n_clusters = max(1, min(n_clusters, I))

    if method == "kmeans":
        return _torch_kmeans(sig, n_clusters, random_state)

    if method == "corr":
        from scipy.cluster.hierarchy import average, fcluster  # optional dep
        from scipy.spatial.distance import pdist
        C = np.corrcoef(sig)
        D = 1.0 - C[np.triu_indices(I, k=1)]
        D = np.nan_to_num(D, nan=1.0)
        Z = average(D)
        return fcluster(Z, t=n_clusters, criterion="maxclust").astype(np.int64) - 1

    raise ValueError(f"unknown clustering method: {method}")


def _torch_kmeans(sig: np.ndarray, n_clusters: int, random_state: int = 0,
                  n_iters: int = 50, batch: int = 4096) -> np.ndarray:
    """Minimal minibatch k-means in torch — enough for I×T calibration
    matrices, avoids sklearn's blocked Cython extensions on locked-down
    Windows machines."""
    rng = np.random.default_rng(random_state)
    X = torch.from_numpy(sig)
    init_idx = rng.choice(sig.shape[0], size=n_clusters, replace=False)
    centers = X[init_idx].clone()
    for _ in range(n_iters):
        idx = rng.choice(sig.shape[0], size=min(batch, sig.shape[0]), replace=False)
        B = X[idx]
        assign = torch.cdist(B, centers).argmin(dim=1)
        for c in range(n_clusters):
            m = assign == c
            if m.any():
                centers[c] = B[m].mean(dim=0)
    assign_all = torch.cdist(X, centers).argmin(dim=1)
    return assign_all.numpy().astype(np.int64)


def cluster_statistics(z_matrix: np.ndarray, assignment: np.ndarray) -> Dict:
    """Milestone 3 deliverable stats: cluster sizes + intra-cluster similarity.

    assignment: [I] per-neuron cluster id (neurons = columns of z_matrix).
    intra_corr_mean: mean Pearson correlation between each sampled neuron's
    activation signature (row of sig) and the mean signature of its cluster.
    """
    sig = neuron_signatures(z_matrix)                    # [I, T]
    assignment = np.asarray(assignment).astype(np.int64)
    sizes_all = np.bincount(assignment, minlength=int(assignment.max()) + 1)
    sizes = sizes_all[sizes_all > 0]
    # mean signature per cluster: [n_clusters, T]
    n_c = sizes_all.shape[0]
    sums = np.zeros((n_c, sig.shape[1]), dtype=np.float64)
    np.add.at(sums, assignment, sig)
    means = sums / np.maximum(sizes_all[:, None], 1)

    intra: List[float] = []
    rng = np.random.default_rng(0)
    idx = rng.permutation(sig.shape[0])[:512]            # cap cost
    for i in idx:
        c = assignment[i]
        if sizes_all[c] > 1:
            a, b = sig[i], means[c]
            va, vb = a - a.mean(), b - b.mean()
            denom = (np.linalg.norm(va) * np.linalg.norm(vb))
            if denom > 0:
                intra.append(float(va @ vb / denom))
    return {
        "n_clusters": int(len(sizes)),
        "cluster_size_mean": float(sizes.mean()) if len(sizes) else 0.0,
        "cluster_size_std": float(sizes.std()) if len(sizes) else 0.0,
        "intra_corr_mean": float(np.mean(intra)) if intra else float("nan"),
    }


def expert_activation_stats(z_matrix: np.ndarray, groups: List[List[int]],
                            threshold_quantile: float = 0.5) -> Dict:
    """P(expert active) and per-token active-expert counts (§15, §32, Milestone 2).

    An expert counts as "active" at a token when its mean z-energy is above the
    expert's own quantile threshold over the calibration set.
    """
    energy = (np.asarray(z_matrix, dtype=np.float32) ** 2)
    per_expert = np.stack([energy[:, g].mean(axis=1) for g in groups], axis=1)  # [T, E]
    thresh = np.quantile(per_expert, threshold_quantile, axis=0, keepdims=True)
    active = per_expert > thresh                              # [T, E]
    return {
        "p_active": active.mean(axis=0),                      # [E]
        "active_per_token_mean": float(active.sum(axis=1).mean()),
        "active_per_token_p90": float(np.quantile(active.sum(axis=1), 0.9)),
    }


def task_conditional_stats(z_by_task: Dict[str, np.ndarray],
                           groups: List[List[int]],
                           threshold_quantile: float = 0.5) -> Dict:
    """§15 task-conditional activation P(expert active | task) + expert entropy
    (§31): low-entropy experts are specialized; high-entropy form the core."""
    per_task: Dict[str, np.ndarray] = {}
    for task, zmat in z_by_task.items():
        energy = np.asarray(zmat, dtype=np.float32) ** 2
        per_expert = np.stack([energy[:, g].mean(axis=1) for g in groups], axis=1)
        thresh = np.quantile(per_expert, threshold_quantile, axis=0, keepdims=True)
        per_task[task] = (per_expert > thresh).mean(axis=0)   # [E] P(active|task)

    tasks = sorted(per_task)
    P = np.stack([per_task[t] for t in tasks], axis=0)        # [T_tasks, E]
    eps = 1e-12
    H = -(P * np.log2(P + eps)).sum(axis=0) / np.log2(len(tasks))  # normalized [0,1]
    return {
        "tasks": tasks,
        "p_active_by_task": {t: per_task[t].tolist() for t in tasks},
        "expert_task_entropy": H.tolist(),                    # [E] 1.0 = universal
    }


def build_reordered_model(model, layer_key: str, assignment: np.ndarray,
                          verify: bool = True) -> Dict:
    """Cluster → permute one layer's FFN in-place (§11, Milestone 3).

    Returns verification info. Dense computation is preserved exactly.
    """
    from .model_utils import get_mlp
    ffn = get_mlp(model, layer_key)
    groups = groups_from_assignment(assignment)
    if verify:
        w = ffn.down_proj.weight
        probe = torch.randn(1, 1, w.shape[0], device=w.device)
        ok = verify_reorder_equivalence(ffn, probe, groups)
        if not ok:
            raise RuntimeError("reorder equivalence check failed — aborting")
    apply_reorder_inplace_(ffn, groups)
    return {"layer": layer_key, "n_experts": len(groups),
            "sizes": expert_sizes(groups).tolist()}


def save_assignments(assignment: np.ndarray, centroids: Optional[np.ndarray],
                     out_dir: str) -> Dict[str, str]:
    """Milestone 3 deliverables: expert_assignments.npy (+ centroids)."""
    from pathlib import Path
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    p1 = d / "expert_assignments.npy"
    np.save(p1, assignment.astype(np.int64))
    paths = {"assignments": str(p1)}
    if centroids is not None:
        p2 = d / "expert_centroids.npy"
        np.save(p2, centroids.astype(np.float32))
        paths["centroids"] = str(p2)
    return paths