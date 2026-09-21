"""M4 Steps 5-9 (headless twin of notebooks/token_router.ipynb): split, train,
evaluate routers vs the M1 oracle; writes results/m4_token_router/ artifacts.

Logic is IDENTICAL to the notebook cells and must stay in sync.

- Split: BY PROMPT at 70/30 (never token-random — neighboring tokens leak).
- Routers: always-on top-frequency baseline, random baseline, centroid
  (cosine to per-block centroids from train tokens where block was oracle
  top-19), linear probe (76x2560, no bias, L2-init, full-batch GD regressing
  log1p(importance)).
- Metrics: recall@K / precision@K / jaccard@K vs oracle top-K at K in {8,19,30}.
- Artifacts per ARCHIVE_POLICY: router_vs_oracle.csv, router_recall.png,
  router_summary.json.

Usage:
    python scripts/run_m4_router.py            # full run (~2-3 min)
    python scripts/run_m4_router.py --epochs 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", r"E:\hf_cache")

MODEL_ID = "Qwen/Qwen3-4B"
LAYERS = ["model.layers.7", "model.layers.18", "model.layers.29"]
GROUP_SIZE = 128
K_LIST = [8, 19, 30]
K_GATE = 19
K_BUILD = 19          # centroid membership: block in token's oracle top-19
CAPTURE_PROMPTS = 8
MATH_PROMPTS = 20
SPLIT_FRAC = 0.70
SPLIT_SEED = 0
PROBE_EPOCHS = 400
PROBE_LR = 3e-2
PROBE_L2 = 1e-4

CACHE_DIR = ROOT / "results" / "traces" / "m4_capture"
OUT_DIR = ROOT / "results" / "m4_token_router"


def slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)[:40].strip("_") or "p"


def prompt_list():
    from src.datasets import CATEGORIES, generated_arithmetic
    items = [("cat", c, p) for c, ps in CATEGORIES.items() for p in ps[:CAPTURE_PROMPTS]]
    items += [("math", "math_gen", p) for p in generated_arithmetic(MATH_PROMPTS, seed=7)]
    # dedupe on the cache key (slug collisions in generated math collapsed
    # to one cache file; dedupe keeps dataset and cache consistent)
    seen, out = set(), []
    for kind, cat, p in items:
        key = (kind, cat, slug(p))
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, cat, p))
    return out


def cache_path(kind: str, cat: str, prompt: str, lk: str) -> Path:
    return CACHE_DIR / f"{lk.replace('.', '_')}__{kind}_{slug(cat)}__{slug(prompt)}.npz"


def load_layer_data(lk: str):
    """Returns X [T, H] float32, IMP [T, E] float32, prompt_ids [T] int."""
    items = prompt_list()
    Xs, Is, pid = [], [], []
    for i, (kind, cat, p) in enumerate(items):
        f = cache_path(kind, cat, p, lk)
        if not f.exists():
            continue
        d = np.load(f)
        Xs.append(d["x"].astype(np.float32))
        Is.append(d["imp"].astype(np.float32))
        pid.append(np.full(d["x"].shape[0], i, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(Is), np.concatenate(pid)


def oracle_stats(IMP: np.ndarray, name: str) -> dict:
    """Distribution of oracle top-K (holdout ground truth quality check)."""
    K = K_GATE
    top = np.argsort(-IMP, axis=1)[:, :K]
    share = np.take_along_axis(IMP, top, axis=1).sum(1) / np.maximum(IMP.sum(1), 1e-9)
    cnt = Counter(top.ravel().tolist())
    head = cnt.most_common(3)
    head_share = sum(c for _, c in head) / top.size
    stats = {
        "tokens": int(IMP.shape[0]),
        f"top{K}_importance_share_mean": float(share.mean()),
        f"top{K}_importance_share_std": float(share.std()),
        "gini": float(gini(IMP.mean(0))),
        "head3_blocks": [int(b) for b, _ in head],
        "head3_share_of_picks": float(head_share),
    }
    print(f"  [{name}] top{K} share of importance: {stats[f'top{K}_importance_share_mean']:.3f}"
          f" ± {stats[f'top{K}_importance_share_std']:.3f} | head-3 blocks {head[0][0]},{head[1][0]},{head[2][0]}"
          f" cover {head_share:.1%} of top-{K} picks | gini(mean imp) {stats['gini']:.3f}")
    return stats


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, np.float64))
    n = x.size
    if x.sum() <= 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


# ------------------------------------------------------------------ routers
class FrequencyRouter:
    """Always-on baseline (§52): pick the globally most frequent oracle blocks."""

    def __init__(self, counts: np.ndarray):
        self.order = np.argsort(-counts)          # [E]

    def topk(self, X, k: int) -> np.ndarray:
        n = X.shape[0]
        return np.tile(self.order[:k], (n, 1))


class RandomRouter:
    def __init__(self, n_experts: int, seed: int = 0):
        self.n_experts = n_experts
        self.rng = np.random.default_rng(seed)

    def topk(self, X, k: int) -> np.ndarray:
        out = np.empty((X.shape[0], k), dtype=np.int64)
        for i in range(X.shape[0]):
            out[i] = self.rng.choice(self.n_experts, size=k, replace=False)
        return out


class KNNetRouter:
    """Nonparametric predictability test: score_e = mean importance of the m
    nearest train tokens (cosine). If this fails too, x carries no decodable
    signal about block importance beyond global frequency."""

    def __init__(self, Xtr: np.ndarray, Itr: np.ndarray, m: int = 16):
        self.Xtr = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-9)
        self.Itr = Itr
        self.m = m

    def scores(self, X: np.ndarray) -> np.ndarray:
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        sim = Xn @ self.Xtr.T                                   # [T, Ttr]
        idx = np.argpartition(-sim, self.m, axis=1)[:, :self.m]
        return self.Itr[idx].mean(axis=1)                       # [T, E]

    def topk(self, X: np.ndarray, k: int) -> np.ndarray:
        return np.argsort(-self.scores(X), axis=1)[:, :k]


class CentroidRouter:
    """§18: score_e = cos(x, centroid_e); centroid = mean x where e in top-19."""

    def __init__(self, Xtr: np.ndarray, top_tr: np.ndarray, n_experts: int):
        E, H = n_experts, Xtr.shape[1]
        Xn = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-9)
        centroids = np.zeros((E, H), np.float32)
        counts = np.zeros(E, np.float64)
        for e in range(E):
            m = (top_tr == e).any(axis=1)
            counts[e] = m.sum()
            if counts[e]:
                centroids[e] = Xn[m].mean(0)
        empty = counts == 0
        if empty.any():
            rng = np.random.default_rng(0)
            r = rng.standard_normal((int(empty.sum()), H)).astype(np.float32)
            centroids[empty] = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
        norms = np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
        self.centroids = centroids / norms
        self.counts = counts
        self.n_experts = E

    def scores(self, X: np.ndarray) -> np.ndarray:
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        return Xn @ self.centroids.T            # [T, E]

    def topk(self, X: np.ndarray, k: int) -> np.ndarray:
        return np.argsort(-self.scores(X), axis=1)[:, :k]


class LinearProbeRouter:
    """76xH linear map, no bias, L2 init; full-batch GD on log1p(importance)."""

    def __init__(self, Xtr: np.ndarray, Ytr: np.ndarray, epochs: int = PROBE_EPOCHS,
                 lr: float = PROBE_LR, l2: float = PROBE_L2, seed: int = 0):
        import torch
        torch.manual_seed(seed)
        X = torch.from_numpy(Xtr).float()
        Y = torch.from_numpy(np.log1p(Ytr)).float()
        # standardize features with train stats
        self.mu = X.mean(0, keepdim=True)
        self.sd = X.std(0, keepdim=True).clamp_min(1e-6)
        Xn = (X - self.mu) / self.sd
        E = Y.shape[1]
        W = torch.nn.Parameter(torch.empty(E, X.shape[1]).normal_(0, 0.01))
        opt = torch.optim.Adam([W], lr=lr)
        loss_fn = torch.nn.SmoothL1Loss()
        for ep in range(epochs):
            opt.zero_grad()
            pred = Xn @ W.T
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()
        self.W = W.detach()
        self._t = torch

    def scores(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            Xn = (torch.from_numpy(X).float() - self.mu) / self.sd
            return (Xn @ self.W.T).numpy().astype(np.float32)

    def topk(self, X: np.ndarray, k: int) -> np.ndarray:
        return np.argsort(-self.scores(X), axis=1)[:, :k]


# ------------------------------------------------------------------ metrics
def pr_at_k(router_top: np.ndarray, oracle_top: np.ndarray) -> tuple:
    hits = np.array([len(set(a) & set(o)) for a, o in zip(router_top, oracle_top)],
                    dtype=np.float64)
    return hits / router_top.shape[1], hits / oracle_top.shape[1]   # precision, recall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=PROBE_EPOCHS)
    args = ap.parse_args()

    import torch
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "csv").mkdir(exist_ok=True)
    (OUT_DIR / "plots").mkdir(exist_ok=True)
    (OUT_DIR / "json").mkdir(exist_ok=True)

    items = prompt_list()
    n_prompts = len(items)
    print(f"prompts (deduped cache keys): {n_prompts}")
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n_prompts)
    n_train = int(round(SPLIT_FRAC * n_prompts))
    train_pids = set(perm[:n_train].tolist())
    print(f"split by prompt: {n_train} train / {n_prompts - n_train} holdout "
          f"(seed {SPLIT_SEED})")

    import csv as _csv
    csv_path = OUT_DIR / "csv" / "router_vs_oracle.csv"
    f_csv = open(csv_path, "w", newline="", encoding="utf-8")
    w = _csv.writer(f_csv)
    w.writerow(["layer", "router", "K", "recall", "precision", "jaccard"])

    summary = {"model": MODEL_ID, "group_size": GROUP_SIZE, "n_experts": 9728 // GROUP_SIZE,
               "split": {"frac": SPLIT_FRAC, "seed": SPLIT_SEED,
                         "n_train_prompts": n_train, "n_holdout_prompts": n_prompts - n_train,
                         "discipline": "by-prompt (no token-level random split)"},
               "oracle": {}, "routers": {}, "probe": {"epochs": args.epochs, "lr": PROBE_LR,
                                                       "l2": PROBE_L2},
               "latency": {}, "train_time_s": {}}

    for lk in LAYERS:
        t0 = time.time()
        X, IMP, pid = load_layer_data(lk)
        E = IMP.shape[1]
        print(f"\n=== {lk}: {X.shape[0]} tokens | {E} blocks | {X.shape[1]} hidden ===")

        tr_m = np.isin(pid, list(train_pids))
        ho_m = ~tr_m
        Xtr, Itop_tr, pid_tr = X[tr_m], IMP[tr_m], pid[tr_m]
        Xho, Iho, pid_ho = X[ho_m], IMP[ho_m], pid[ho_m]
        print(f"  train {Xtr.shape[0]} tok ({len(set(pid_tr.tolist()))} prompts) | "
              f"holdout {Xho.shape[0]} tok ({len(set(pid_ho.tolist()))} prompts)")

        summary["oracle"][lk] = oracle_stats(Iho, "holdout oracle")

        # static-share diagnostic: how much of per-token top-K does the STATIC
        # train-frequency top-K already cover (upper bound of always-on)?
        top_ho_all = np.argsort(-Iho, axis=1)[:, :K_GATE]
        static_top = np.argsort(-np.array([Counter(top_ho_all.ravel().tolist()).get(e, 0)
                                           for e in range(E)]))[:K_GATE]
        static_hit = np.mean([len(set(static_top) & set(t)) for t in top_ho_all])
        print(f"  static-share: fixed train-freq top-{K_GATE} covers "
              f"{static_hit:.3f} of holdout top-{K_GATE} picks")
        summary["oracle"][lk]["static_top19_coverage"] = round(float(static_hit), 4)

        top_tr = np.argsort(-Itop_tr, axis=1)[:, :K_BUILD]      # oracle top-19 on train
        top_ho = np.argsort(-Iho, axis=1)[:, :K_GATE]           # oracle top-19 on holdout
        freq = Counter(top_tr.ravel().tolist())

        routers = {
            "always_freq": FrequencyRouter(np.array([freq.get(e, 0) for e in range(E)])),
            "random": RandomRouter(E, seed=0),
            "centroid": CentroidRouter(Xtr, top_tr, E),
            "knn": KNNetRouter(Xtr, Itop_tr, m=16),
            "probe": LinearProbeRouter(Xtr, Itop_tr, epochs=args.epochs),
        }
        # centroid coverage diagnostics
        empty = int((routers["centroid"].counts == 0).sum())
        print(f"  centroid: {empty}/{E} blocks never in train top-{K_BUILD} "
              f"(random directions)")
        print(f"  centroid build: {time.time()-t0:.1f}s | probe training...", flush=True)

        tp = time.time()
        routers["probe"] = routers["probe"]  # trained in constructor
        summary["train_time_s"][lk] = round(time.time() - t0, 1)

        for rname, router in routers.items():
            t_r = time.time()
            tr_scores = router.scores(Xtr) if hasattr(router, "scores") else None
            rtr = None
            for K in K_LIST:
                orc_tr = np.argsort(-Itop_tr, axis=1)[:, :K]
                if tr_scores is not None:
                    top_tr_pred = np.argsort(-tr_scores, axis=1)[:, :K]
                    _ptr, rtr = pr_at_k(top_tr_pred, orc_tr)
                ho_top = router.topk(Xho, K)
                orc_ho = np.argsort(-Iho, axis=1)[:, :K]
                p_ho, r_ho = pr_at_k(ho_top, orc_ho)
                inter = np.array([len(set(a) & set(o)) for a, o in zip(ho_top, orc_ho)])
                jac = inter / (2 * K - inter)
                w.writerow([lk, rname, K, f"{r_ho.mean():.4f}", f"{p_ho.mean():.4f}",
                            f"{jac.mean():.4f}"])
                if K == K_GATE:
                    tr_str = f"train {rtr.mean():.3f} | " if rtr is not None else ""
                    print(f"    {rname:>12} recall@{K}: {tr_str}"
                          f"holdout {r_ho.mean():.3f} (prec {p_ho.mean():.3f})")
            summary["routers"].setdefault(lk, {})[rname] = {
                "holdout_recall": {str(K): round(float(pr_at_k(
                    router.topk(Xho, K), np.argsort(-Iho, axis=1)[:, :K])[1].mean()), 4)
                    for K in K_LIST},
                "eval_time_ms_per_1k_tokens": round(
                    (time.time() - t_r) / max(Xho.shape[0], 1), 3),
            }
        f_csv.flush()

    f_csv.close()
    with open(OUT_DIR / "json" / "router_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved: {csv_path}")
    print(f"saved: {OUT_DIR / 'json' / 'router_summary.json'}")


if __name__ == "__main__":
    main()
