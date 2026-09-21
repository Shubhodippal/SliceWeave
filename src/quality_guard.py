"""Phase 6 — quality-adaptive progressive activation (§20-§23, §61, Milestone 6).

Central contribution: do NOT commit to one fixed Top-K. Start small, estimate
the omitted contribution / confidence (§22), and expand the active expert set
only when the estimate says the skipped experts matter. Fallback chain:

    S0 -> +S1 -> +S2 -> ... -> full layer (§61)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .token_router import CentroidRouter, RouterPrediction


@dataclass
class ExpansionRecord:
    """Per-token/layer trace for the recovery-rate metric (§41)."""
    layer_key: str
    initial_k: int
    final_k: int
    expansions: int          # 0 = initial set was sufficient
    fallback_full: bool
    confidences: List[float] = field(default_factory=list)
    estimated_residual: float = 0.0


@dataclass
class GuardConfig:
    """§20/§22 knobs."""
    initial_fraction: float = 0.15      # starting expert fraction (fast path)
    max_fraction: float = 1.0           # expansion ceiling (1.0 = full layer)
    expansion_factor: float = 2.0       # K *= factor on each expansion
    max_expansions: int = 3
    confidence_threshold: float = 0.90  # proceed when confidence >= this
    # residual estimator weights (§22 Approach A: router score mass)
    score_mass_threshold: float = 0.85  # fraction of total score mass required
    use_margin: bool = True             # include router margin in confidence
    margin_floor: float = 0.02


class ResidualEstimator:
    """§22 cheap estimates of the omitted contribution r = Σ_{e∉S} y_e.

    Approach A (default): router score mass — the unselected experts' softmax
    score mass approximates the fraction of contribution left behind. This is
    the only signal that costs zero extra compute.
    """

    def __init__(self, config: GuardConfig):
        self.cfg = config

    def score_mass(self, prediction: RouterPrediction, selected: Sequence[int]) -> float:
        """Fraction of total router score mass covered by `selected`."""
        s = prediction.scores
        total = float(s.sum())
        if total <= 0:
            return 1.0
        covered = float(s[list(selected)].sum())
        return covered / total

    def confidence(self, prediction: RouterPrediction,
                   selected: Sequence[int]) -> Tuple[float, float]:
        """Return (confidence ∈ [0,1], estimated residual mass ∈ [0,1]).

        Combines §21 signals available at zero cost: score mass + entropy +
        margin. confidence = score_mass, discounted by router entropy.
        """
        mass = self.score_mass(prediction, selected)
        residual = 1.0 - mass
        # entropy discount: a flat router score distribution is less trustworthy
        p = prediction.scores.clamp_min(1e-12)
        pn = p / p.sum()
        H = float(-(pn * pn.log()).sum()) / float(torch.log(pn.new_tensor(pn.numel())))
        conf = mass * (1.0 - 0.15 * H)
        if self.cfg.use_margin and prediction.margin < self.cfg.margin_floor:
            conf *= 0.95  # near-tie at the selection boundary -> slightly riskier
        return min(conf, 1.0), residual


class QualityGuard:
    """Progressive activation driver (§20, §46 steps 7-9, §61).

    Works over a CentroidRouter: given a hidden state, produce an initial
    expert set, then a schedule of expansions with confidences, stopping when
    the guard is satisfied or the ceiling/full fallback is reached.
    """

    def __init__(self, router: CentroidRouter, config: Optional[GuardConfig] = None,
                 residual_estimator: Optional[ResidualEstimator] = None):
        self.router = router
        self.cfg = config or GuardConfig()
        self.estimator = residual_estimator or ResidualEstimator(self.cfg)

    def _k_for(self, fraction: float) -> int:
        return max(1, int(round(self.router.n_experts * fraction)))

    def run(self, hidden: torch.Tensor) -> Tuple[List[int], ExpansionRecord]:
        """One progressive routing decision for one token/layer (§46 1-9)."""
        E = self.router.n_experts
        frac = self.cfg.initial_fraction
        pred = self.router.predict(hidden, k=self._k_for(frac))
        selected = list(pred.topk)
        conf, residual = self.estimator.confidence(pred, selected)
        rec = ExpansionRecord(layer_key=self.router.layer_key,
                              initial_k=len(selected), final_k=len(selected),
                              expansions=0, fallback_full=False,
                              confidences=[conf], estimated_residual=residual)

        while conf < self.cfg.confidence_threshold and \
                len(selected) / E < self.cfg.max_fraction and \
                rec.expansions < self.cfg.max_expansions:
            frac = min(frac * self.cfg.expansion_factor, self.cfg.max_fraction)
            k = self._k_for(frac)
            if k <= len(selected):
                k = min(len(selected) + 1, E)
            # incremental expansion: keep the current set, add next-ranked (§46 8)
            new_top = [e for e in pred.ranking if e not in selected][: max(0, k - len(selected))]
            selected = selected + new_top
            rec.expansions += 1
            rec.final_k = len(selected)
            conf, residual = self.estimator.confidence(pred, selected)
            rec.confidences.append(conf)
            rec.estimated_residual = residual

        if conf < self.cfg.confidence_threshold and len(selected) >= E:
            rec.fallback_full = True
        return selected, rec

    def fixed(self, hidden: torch.Tensor, fraction: float) -> List[int]:
        """Ablation baseline (§52/§53): fixed Top-K, no adaptation."""
        return self.router.predict(hidden, k=self._k_for(fraction)).topk


def recovery_summary(records: Sequence[ExpansionRecord]) -> Dict[str, float]:
    """Milestone 6 deliverable: fallback/recovery distribution (§41)."""
    if not records:
        return {}
    n = len(records)
    max_exp = max(r.expansions for r in records)
    out = {f"expansions_{i}": sum(1 for r in records if r.expansions == i) / n
           for i in range(max_exp + 1)}
    out["fallback_full"] = sum(1 for r in records if r.fallback_full) / n
    out["mean_final_fraction"] = sum(r.final_k for r in records) / n / \
        max(records[0].initial_k and records[0].final_k / records[0].final_k, 1) \
        if False else sum(r.final_k for r in records) / n
    return out
