"""Phase 4 — token/layer-level micro-expert router (§17-§19, Milestones 4-5).

Baseline router (start simple, §20): cosine similarity between the incoming
hidden state x_l,t and per-expert centroids built during calibration (§18).

Router overhead rule (§80): this module stays pure-torch matrix math so its
latency can be measured against the FFN compute it saves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class RouterPrediction:
    scores: torch.Tensor            # [E] raw router scores per expert
    ranking: List[int]              # expert ids, best first
    topk: List[int]                 # selected experts
    entropy: float = 0.0            # router uncertainty signal (§21)
    margin: float = 0.0             # gap between selected/unselected mass (§21)


class CentroidRouter:
    """Per-layer centroid router: score_e = cos(x, centroid_e) (§17-§18)."""

    def __init__(self, layer_key: str, n_experts: int, d_model: int,
                 device: str = "cpu"):
        self.layer_key = layer_key
        self.n_experts = n_experts
        self.d_model = d_model
        self.device = device
        self.centroids = torch.zeros(n_experts, d_model, device=device)
        self.counts = torch.zeros(n_experts, device=device)

    # ------------------------------------------------ calibration (§18)
    def add_observation(self, hidden: torch.Tensor, important_experts: Sequence[int],
                        weights: Optional[torch.Tensor] = None) -> None:
        """Accumulate hidden states where each expert was oracle-important."""
        hidden = hidden.detach().float().reshape(-1, self.d_model).to(self.device)
        for i, e in enumerate(important_experts):
            w = 1.0 if weights is None else float(weights[i])
            self.centroids[e] += w * hidden.reshape(-1)
            self.counts[e] += 1.0

    def finalize(self) -> None:
        """Normalize accumulated sums into mean centroids (§18 formula)."""
        c = self.counts.clamp_min(1.0).unsqueeze(1)
        self.centroids = self.centroids / c
        empty = self.counts == 0
        if empty.any():
            # give unseen experts a random-ish direction on the unit sphere
            g = torch.Generator().manual_seed(0)
            rnd = torch.randn(int(empty.sum()), self.d_model, generator=g)
            self.centroids[empty] = rnd / rnd.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.centroids = torch.nn.functional.normalize(self.centroids, dim=1)

    # ------------------------------------------------ inference (§19)
    def predict(self, hidden: torch.Tensor, k: Optional[int] = None) -> RouterPrediction:
        x = hidden.detach().float().reshape(-1, self.d_model).to(self.device)
        x = torch.nn.functional.normalize(x, dim=-1)
        scores = self.centroids @ x.reshape(-1)              # [E] cosine scores
        order = torch.argsort(scores, descending=True)
        k = k or max(1, self.n_experts // 4)
        top = order[:k].tolist()

        # uncertainty signals for the quality guard (§21)
        p = torch.softmax(scores, dim=0)
        entropy = float(-(p * p.clamp_min(1e-12).log()).sum())
        sel = float(p[order[:k]].sum())
        margin = float(scores[order[k - 1]] - scores[order[min(k, self.n_experts - 1)]])

        return RouterPrediction(scores=scores.cpu(), ranking=order.tolist(),
                                topk=top, entropy=entropy, margin=margin)

    def predict_batch(self, hidden: torch.Tensor, k: Optional[int] = None) -> List[RouterPrediction]:
        """hidden: [batch, d] or [batch, seq, d] -> per-row predictions."""
        x = hidden.detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        x = torch.nn.functional.normalize(x, dim=-1).to(self.device)
        scores = x @ self.centroids.T                        # [B, E]
        k = k or max(1, self.n_experts // 4)
        order = torch.argsort(scores, dim=1, descending=True)
        p = torch.softmax(scores, dim=1)
        ent = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
        sel = torch.gather(p, 1, order[:, :k]).sum(dim=1)
        out = []
        for b in range(scores.shape[0]):
            top = order[b, :k].tolist()
            kk = min(k, self.n_experts - 1)
            margin = float(scores[b, order[b, k - 1]] - scores[b, order[b, kk]])
            out.append(RouterPrediction(scores=scores[b].cpu(), ranking=order[b].tolist(),
                                        topk=top, entropy=float(ent[b]), margin=margin))
        return out

    # ------------------------------------------------ persistence
    def state_dict(self) -> Dict:
        return {"layer": self.layer_key, "n_experts": self.n_experts,
                "centroids": self.centroids.cpu().numpy(),
                "counts": self.counts.cpu().numpy()}

    @classmethod
    def from_state(cls, state: Dict, device: str = "cpu") -> "CentroidRouter":
        r = cls(state["layer"], state["n_experts"], state["centroids"].shape[1],
                device=device)
        r.centroids = torch.from_numpy(state["centroids"]).to(device)
        r.counts = torch.from_numpy(state["counts"]).to(device)
        return r


class RandomRouter:
    """Ablation baseline (§52/§53): random expert selection."""

    def __init__(self, n_experts: int, seed: int = 0):
        self.n_experts = n_experts
        self.rng = np.random.default_rng(seed)

    def predict(self, hidden: torch.Tensor, k: Optional[int] = None) -> RouterPrediction:
        k = k or max(1, self.n_experts // 4)
        top = self.rng.choice(self.n_experts, size=k, replace=False).tolist()
        return RouterPrediction(scores=torch.zeros(self.n_experts), ranking=top, topk=top)


class MagnitudeRouter:
    """Ablation baseline (§52): predict via activation magnitude of z — requires
    computing gate/up first (upper bound of what token-level cheap routers do)."""

    def __init__(self, n_experts: int):
        self.n_experts = n_experts

    def predict_from_z(self, z_energy: torch.Tensor, k: Optional[int] = None) -> RouterPrediction:
        k = k or max(1, self.n_experts // 4)
        order = torch.argsort(z_energy, descending=True).tolist()
        return RouterPrediction(scores=z_energy.cpu(), ranking=order, topk=order[:k])


def router_recall_vs_oracle(router_topk: Sequence[int], oracle_topk: Sequence[int]) -> Dict[str, float]:
    """Milestone 4 deliverable metric: fraction of oracle-important experts the
    router recovered (recall) and precision of the router's selection."""
    A, O = set(router_topk), set(oracle_topk)
    return {
        "recall": len(A & O) / max(len(O), 1),
        "precision": len(A & O) / max(len(A), 1),
        "jaccard": len(A & O) / max(len(A | O), 1),
    }


def oracle_topk_from_importance(importance: torch.Tensor, k: int) -> List[int]:
    return torch.topk(importance, k).indices.tolist()
