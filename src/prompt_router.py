"""Phase 5 — prompt-level capability retrieval (§16, Milestone 5).

Before generation:
    encode prompt -> retrieve capability clusters -> request candidate pool
    (plus the always-active universal core, §15/§32)

This narrows the universe (27B) to a candidate set (target 8-12B) BEFORE token
routing, cutting router search cost, RAM working set, and cold traffic (§16).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class CandidatePool:
    """Request-level candidate set (§16, §47)."""
    prompt: str
    # layer_key -> list of candidate expert ids (in addition to universal core)
    by_layer: Dict[str, List[int]] = field(default_factory=dict)
    # expert id -> task affinity score (mean router score under this prompt family)
    affinity: Dict[Tuple[str, int], float] = field(default_factory=dict)
    universal_core: Dict[str, List[int]] = field(default_factory=dict)

    def candidates_for(self, layer_key: str) -> List[int]:
        core = self.universal_core.get(layer_key, [])
        extra = self.by_layer.get(layer_key, [])
        return sorted(set(core) | set(extra))

    def candidate_params(self, ffns, total_params: int) -> Tuple[int, float]:
        """CPR (§38): candidate parameters per request and ratio."""
        from .ffns import FFNInfo
        n = 0
        for key, info in ffns.items():
            k = len(self.candidates_for(key))
            n += info.expert_params(k)
        return n, n / max(total_params, 1)


@dataclass
class PromptProfile:
    """Latent task/capability profile of a prompt (§8.2). No human labels."""
    task_scores: Dict[str, float]        # calibration task name -> affinity
    top_tasks: List[str]
    hidden_mean: np.ndarray              # mean prompt hidden state per probed layer


class PromptRouter:
    """Prompt-level retriever built on top of calibration task profiles.

    Calibration (offline):
        for each task category, record per-layer router scores / expert usage.
        A task's "capability direction" per layer = mean hidden state of its
        prompt tokens (§18 generalized to the prompt level).

    Retrieval (per request, §47):
        1. encode prompt, probe hidden states at a few layers
        2. score each calibration task by cosine to its capability direction
        3. take experts frequently used by the top-matching tasks
        4. union with the universal core (experts active for nearly all tasks)
    """

    def __init__(self):
        # layer_key -> {task: (sum of hidden vectors, count)}
        self.task_directions: Dict[str, Dict[str, tuple]] = {}
        # layer_key -> {task: usage frequency per expert}  ([E] arrays)
        self.task_usage: Dict[str, Dict[str, np.ndarray]] = {}
        # layer_key -> universal-core expert ids (high task entropy, §32)
        self.universal_core: Dict[str, List[int]] = {}

    # ------------------------------------------------ calibration
    def add_task_hidden(self, layer_key: str, task: str, hidden: np.ndarray) -> None:
        d = self.task_directions.setdefault(layer_key, {})
        acc, n = d.get(task, (None, 0))
        h = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if acc is None:
            d[task] = (h, 1)
        else:
            d[task] = (acc + h, n + 1)  # type: ignore[assignment]

    def add_task_usage(self, layer_key: str, task: str, p_active: np.ndarray) -> None:
        self.task_usage.setdefault(layer_key, {})[task] = np.asarray(p_active, np.float32)

    def set_universal_core(self, layer_key: str, expert_ids: Sequence[int]) -> None:
        self.universal_core[layer_key] = sorted(int(e) for e in expert_ids)

    def finalize(self) -> None:
        for layer_key, tasks in self.task_directions.items():
            self.task_directions[layer_key] = {
                t: v[0] / max(v[1], 1) for t, v in tasks.items()}

    # ------------------------------------------------ retrieval (§47)
    def profile_prompt(self, layer_hiddens: Dict[str, np.ndarray]) -> PromptProfile:
        """layer_hiddens: {layer_key: [tokens, d] hidden states from a probe run}."""
        scores: Dict[str, float] = {}
        means: Dict[str, np.ndarray] = {}
        for layer_key, h in layer_hiddens.items():
            dirs = self.task_directions.get(layer_key, {})
            if not dirs:
                continue
            hm = np.asarray(h, dtype=np.float32).mean(axis=0)
            hm = hm / max(float(np.linalg.norm(hm)), 1e-8)
            means[layer_key] = hm
            for task, vec in dirs.items():
                vec = np.asarray(vec[0] if isinstance(vec, tuple) else vec,
                                 dtype=np.float32)
                v = vec / max(float(np.linalg.norm(vec)), 1e-8)
                scores[task] = scores.get(task, 0.0) + float(v @ hm)
        # average across probed layers, rank
        n_layers = max(len(means), 1)
        ranked = sorted(((s / n_layers, t) for t, s in scores.items()), reverse=True)
        return PromptProfile(task_scores={t: s for s, t in ranked},
                             top_tasks=[t for _, t in ranked],
                             hidden_mean=np.stack(list(means.values())) if means
                             else np.zeros(0, dtype=np.float32))

    def build_pool(self, profile: PromptProfile, top_tasks: int = 2,
                   experts_per_task_frac: float = 0.25,
                   core_override: Optional[Dict[str, List[int]]] = None) -> CandidatePool:
        """Assemble the candidate pool from the top matching tasks (§16)."""
        pool = CandidatePool(prompt="")
        pool.universal_core = dict(self.universal_core)
        if core_override:
            pool.universal_core = dict(core_override)
        tasks = profile.top_tasks[:top_tasks]
        for layer_key, usage in self.task_usage.items():
            cand: set = set()
            for t in tasks:
                u = usage.get(t)
                if u is None:
                    continue
                k = max(1, int(round(len(u) * experts_per_task_frac)))
                cand |= set(np.argsort(u)[-k:].tolist())
            if cand:
                pool.by_layer[layer_key] = sorted(int(e) for e in cand)
        return pool
