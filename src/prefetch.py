"""Phase 8 — predictive prefetch simulation (§26-§27, §42, Milestone 8).

Predict which expert blocks will be demanded next and move them before they
are requested (§26). Horizons under test (§27):

    cross-layer : experts of layer l+1 predicted while layer l routes
    next-token  : experts of token t+1
    speculative : t+1 and t+2

Do not assume long-horizon prediction works — measure precision/recall (§42).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .cache import AccessRecord
from .metrics import prefetch_metrics

Key = Tuple[str, int]  # (layer_key, expert_id)


class CrossLayerPredictor:
    """§94: expert usage in layer l predicts usage in layer l+1.

    Calibrated from a trace: P(layer l+1 needs e | layer l used e).
    """

    def __init__(self):
        self.cooccurrence: Dict[Key, Counter] = defaultdict(Counter)
        self.totals: Counter = Counter()

    def observe(self, records: Sequence[AccessRecord]) -> None:
        by_step: Dict[int, List[AccessRecord]] = defaultdict(list)
        for r in records:
            by_step[r.step].append(r)
        for step, recs in by_step.items():
            recs.sort(key=lambda r: r.step)  # stable layer order within step
            for i in range(len(recs)):
                for j in range(i + 1, len(recs)):
                    if recs[j].layer_key == recs[i].layer_key:
                        continue
                    self.cooccurrence[(recs[i].layer_key, recs[i].expert_id)][
                        (recs[j].layer_key, recs[j].expert_id)] += 1
                self.totals[(recs[i].layer_key, recs[i].expert_id)] += 1

    def predict(self, current: Key, layer_key: str, topk: int = 8) -> Set[Key]:
        """Given the current block, predict blocks for `layer_key` (next layer)."""
        c = self.cooccurrence.get(current)
        if not c:
            return set()
        out: Set[Key] = set()
        for (lk, e), n in c.most_common():
            if lk == layer_key and n >= 1:
                out.add((lk, e))
                if len(out) >= topk:
                    break
        return out


class NextTokenPredictor:
    """§95 temporal reuse: expert(t) predicts expert(t+1), strong in code
    generation where local syntax persists across tokens."""

    def __init__(self):
        self.transitions: Dict[Key, Counter] = defaultdict(Counter)

    def observe(self, records: Sequence[AccessRecord]) -> None:
        by_token: Dict[int, Set[Key]] = defaultdict(set)
        for r in records:
            by_token[r.token_id].add((r.layer_key, r.expert_id))
        tokens = sorted(by_token)
        for a, b in zip(tokens, tokens[1:]):
            for src in by_token[a]:
                for dst in by_token[b]:
                    self.transitions[src][dst] += 1

    def predict(self, current_used: Iterable[Key], topk: int = 16) -> Set[Key]:
        agg: Counter = Counter()
        for src in current_used:
            agg.update(self.transitions.get(src, Counter()))
        return {k for k, _ in agg.most_common(topk)}


class OracleHindsightPredictor:
    """Upper bound: prefetch exactly what the trace will need next (§42 ceiling).
    Not deployable — used to measure how much waste a perfect predictor avoids.
    """

    @staticmethod
    def next_needs(records: Sequence[AccessRecord], index: int,
                   horizon_steps: int = 1) -> Set[Key]:
        lo = index + 1
        hi = min(len(records), index + 1 + horizon_steps)
        return {(r.layer_key, r.expert_id) for r in records[lo:hi]}


def evaluate_prefetch(predicted: Sequence[Set[Key]],
                      needed: Sequence[Set[Key]]) -> Dict[str, float]:
    """§42 aggregate precision/recall/waste over per-step predictions."""
    P: Set[Key] = set()
    N: Set[Key] = set()
    prec, rec = [], []
    for p, n in zip(predicted, needed):
        m = prefetch_metrics(set(p), set(n))
        prec.append(m["precision"])
        rec.append(m["recall"])
        P |= set(p)
        N |= set(n)
    return {
        "precision_mean": sum(prec) / max(len(prec), 1),
        "recall_mean": sum(rec) / max(len(rec), 1),
        "precision_union": prefetch_metrics(P, N)["precision"],
        "recall_union": prefetch_metrics(P, N)["recall"],
    }


def wasted_bytes(predicted: Sequence[Set[Key]], needed: Sequence[Set[Key]],
                 sizes: Dict[Key, int]) -> int:
    """Bytes prefetched that were never needed in their horizon (§42)."""
    wasted = 0
    for p, n in zip(predicted, needed):
        wasted += sum(sizes.get(k, 0) for k in set(p) - set(n))
    return wasted
