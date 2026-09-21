"""M4b — Router × Granularity Sweep (headless twin of
notebooks/token_router_granularity.ipynb; logic IDENTICAL, kept in sync).

Question: does the M4 null (token-importance predictable from x only at the
level of a static frequency prior) survive coarser micro-expert granularity?

Key changes vs M4:
1. CORRECTED ROUTER INPUT: M4 hooked the decoder-layer input (pre-attention
   residual). The true inference-time router input is the MLP's own input
   (post-attention hidden state). We now hook model.layers.N.mlp inputs.
2. Capture z (down_proj input) alongside; big-block importance is EXACT via
   ||W[:,I] z||^2 = sum over 128-neuron children of ||W[:,Ic] z||^2, so one
   capture serves every granularity (128/256/512/1024).
3. z-energy diagnostic: ranking by mean z^2 per block (post-hoc, needs the
   full gate/up first) — the upper bound of any cheap router.

Usage: python scripts/run_m4b_granularity.py [--epochs 1000]
"""
from __future__ import annotations

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
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_ID = "Qwen/Qwen3-4B"
LAYERS = ["model.layers.7", "model.layers.18", "model.layers.29"]
GRANULARITIES = [128, 256, 512, 1024]
K_FRAC = 0.25
CAPTURE_PROMPTS = 8
MATH_PROMPTS = 20
SPLIT_FRAC = 0.70
SPLIT_SEED = 0
PROBE_EPOCHS = 1000
PROBE_LR = 3e-2

CACHE_DIR = ROOT / "results" / "traces" / "m4b_capture"
OUT_DIR = ROOT / "results" / "m4b_granularity_router"


def slug(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)[:40].strip("_") or "p"


def prompt_list():
    from src.datasets import CATEGORIES, generated_arithmetic
    items = [("cat", c, p) for c, ps in CATEGORIES.items() for p in ps[:CAPTURE_PROMPTS]]
    items += [("math", "math_gen", p) for p in generated_arithmetic(MATH_PROMPTS, seed=7)]
    seen, out = set(), []
    for kind, cat, p in items:
        key = (kind, cat, slug(p))
        if key not in seen:
            seen.add(key)
            out.append((kind, cat, p))
    return out


def cache_path(kind, cat, prompt, lk):
    return CACHE_DIR / f"{lk.replace('.', '_')}__{kind}_{slug(cat)}__{slug(prompt)}.npz"


def capture_all(model, tokenizer, layers):
    """Capture per (prompt, layer): x_mlp [T,H] f16 (MLP input, token 0 dropped)
    and z [T,I] f16. Big-block importances derived later in numpy (exact)."""
    import torch
    from tqdm.auto import tqdm
    from src.hooks import FFNIntermediateCapture

    items = prompt_list()

    def capture_prompt(prompt: str):
        x_store, z_caps, handles = {}, {}, []
        for lk in layers:
            mlp = model.get_submodule(f"{lk}.mlp")

            def make_hook(key):
                def _hook(mod, args, kwargs=None):
                    x_store[key] = (args[0] if args else kwargs.get("hidden_states")).detach()
                    return None
                return _hook

            handles.append(mlp.register_forward_pre_hook(make_hook(lk), with_kwargs=True))
            zc = FFNIntermediateCapture()
            zc.attach(model, lk)
            z_caps[lk] = zc
        try:
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**enc)
        finally:
            for h in handles:
                h.remove()
            for lk in layers:
                z_caps[lk].remove()
        out = {}
        for lk in layers:
            x = x_store[lk][0, 1:].float()
            z = z_caps[lk].captures[f"{lk}.down_proj"][0, 1:].float()
            out[lk] = {"x": x.cpu().numpy().astype(np.float16),
                       "z": z.cpu().numpy().astype(np.float16)}
        return out

    done0 = sum(1 for kind, c, p in items for lk in layers
                if cache_path(kind, c, p, lk).exists())
    print(f"capture plan: {len(items)} prompts x {len(layers)} layers; "
          f"{done0} already cached")
    t0 = time.time()
    pbar = tqdm(items, unit="prompt")
    n_new = 0
    for kind, cat, p in pbar:
        per_layer = None
        for lk in layers:
            if cache_path(kind, cat, p, lk).exists():
                continue
            if per_layer is None:
                per_layer = capture_prompt(p)
            np.savez_compressed(cache_path(kind, cat, p, lk), **per_layer[lk])
            n_new += 1
    pbar.close()
    files = sorted(CACHE_DIR.glob("*.npz"))
    print(f"capture: {len(files)} files ({n_new} new), {time.time()-t0:.0f}s this session")


def load_layer(lk):
    Xs, Zs, pid = [], [], []
    for i, (kind, cat, p) in enumerate(prompt_list()):
        f = cache_path(kind, cat, p, lk)
        if not f.exists():
            continue
        d = np.load(f)
        Xs.append(d["x"].astype(np.float32))
        Zs.append(d["z"].astype(np.float32))
        pid.append(np.full(d["x"].shape[0], i, dtype=np.int64))
    return np.concatenate(Xs), np.concatenate(Zs), np.concatenate(pid)


def child_block_norms(Z, W_inter_hidden, child=128):
    """[T, I] z @ [I, H] W^T per 128-neuron child block -> [T, I/child] norms.
    W_inter_hidden: [hidden, inter] as stored (down_proj.weight)."""
    import torch
    H = W_inter_hidden.shape[0]
    Wf = torch.from_numpy(W_inter_hidden).float().T          # [I, H]
    out = []
    B = 512
    for s in range(0, Z.shape[0], B):
        zb = torch.from_numpy(Z[s:s+B]).float()              # [b, I]
        y = zb @ Wf                                          # [b, H] full output
        # per-child contribution norm: || W[:,Ic] z[Ic] || needs per-block
        # projection; compute blockwise to stay exact
        norms = torch.empty(zb.shape[0], zb.shape[1] // child)
        for bi in range(zb.shape[1] // child):
            sl = slice(bi * child, (bi + 1) * child)
            yb = zb[:, sl] @ Wf[sl, :]                       # [b, H]
            norms[:, bi] = yb.norm(dim=1)
        out.append(norms.numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def merge_children(imp_child, g):
    """[T, I/128] child norms -> [T, I/g] big-block norms (exact: sqrt of sum
    of squared child norms within each big block)."""
    k = g // 128
    T, C = imp_child.shape
    big = np.sqrt((imp_child.reshape(T, C // k, k) ** 2).sum(axis=2))
    return big.astype(np.float32)


# ------------------------------------------------------------------ routers
class FrequencyRouter:
    def __init__(self, counts):
        self.order = np.argsort(-counts)
    def scores(self, X):
        return np.tile(-np.arange(len(self.order)), (X.shape[0], 1))
    def topk(self, X, k):
        return np.tile(self.order[:k], (X.shape[0], 1))


class RandomRouter:
    def __init__(self, n_experts, seed=0):
        self.n_experts, self.rng = n_experts, np.random.default_rng(seed)
    def topk(self, X, k):
        return np.stack([self.rng.permutation(self.n_experts)[:k]
                         for _ in range(X.shape[0])])


class CentroidRouter:
    def __init__(self, Xtr, top_tr, E):
        Xn = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-9)
        C = np.zeros((E, Xtr.shape[1]), np.float32)
        counts = np.zeros(E)
        for e in range(E):
            m = (top_tr == e).any(axis=1)
            counts[e] = m.sum()
            if counts[e]:
                C[e] = Xn[m].mean(0)
        empty = counts == 0
        if empty.any():
            r = np.random.default_rng(0).standard_normal(
                (int(empty.sum()), Xtr.shape[1])).astype(np.float32)
            C[empty] = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)
        self.centroids = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        self.counts = counts
    def scores(self, X):
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        return Xn @ self.centroids.T
    def topk(self, X, k):
        return np.argsort(-self.scores(X), axis=1)[:, :k]


class KNNetRouter:
    def __init__(self, Xtr, Itr, m=16):
        self.Xtr = Xtr / np.maximum(np.linalg.norm(Xtr, axis=1, keepdims=True), 1e-9)
        self.Itr, self.m = Itr, m
    def scores(self, X):
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        sim = Xn @ self.Xtr.T
        idx = np.argpartition(-sim, self.m, axis=1)[:, :self.m]
        return self.Itr[idx].mean(axis=1)
    def topk(self, X, k):
        return np.argsort(-self.scores(X), axis=1)[:, :k]


class LinearProbeRouter:
    def __init__(self, Xtr, Itr, epochs=PROBE_EPOCHS, lr=PROBE_LR, seed=0):
        import torch
        torch.manual_seed(seed)
        X = torch.from_numpy(Xtr).float()
        Y = torch.from_numpy(np.log1p(Itr)).float()
        self.mu = X.mean(0, keepdim=True)
        self.sd = X.std(0, keepdim=True).clamp_min(1e-6)
        Xn = (X - self.mu) / self.sd
        W = torch.nn.Parameter(torch.empty(Y.shape[1], X.shape[1]).normal_(0, 0.01))
        opt = torch.optim.Adam([W], lr=lr)
        loss_fn = torch.nn.SmoothL1Loss()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(Xn @ W.T, Y)
            loss.backward()
            opt.step()
        self.W = W.detach()
    def scores(self, X):
        import torch
        with torch.no_grad():
            Xn = (torch.from_numpy(X).float() - self.mu) / self.sd
            return (Xn @ self.W.T).numpy().astype(np.float32)
    def topk(self, X, k):
        return np.argsort(-self.scores(X), axis=1)[:, :k]


class ZEnergyRouter:
    """Post-hoc diagnostic: rank blocks by mean z^2 (needs full gate/up first).
    NOT a deployable router — the ceiling of cheap post-hoc selection."""
    def __init__(self, Ztr):
        self.energy_tr = (Ztr.astype(np.float32) ** 2).mean(axis=0)  # [I]
    def scores_from_z(self, Z):
        return (Z.astype(np.float32) ** 2).mean(axis=-1)  # placeholder
    def topk_from_childnorm(self, imp_big):
        return np.argsort(-imp_big, axis=1)[:, :]


def pr_at_k(router_top, oracle_top):
    hits = np.array([len(set(a) & set(o)) for a, o in zip(router_top, oracle_top)], float)
    return hits / router_top.shape[1], hits / oracle_top.shape[1]


def main():
    import torch
    from src.model_utils import load_model_and_tokenizer, mlp_down_weight

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("csv", "plots", "json"):
        (OUT_DIR / sub).mkdir(exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, "bfloat16")
    from src.model_utils import describe_model
    info = describe_model(model)
    print(f"params={info['params']/1e9:.2f}B hidden={info['hidden_size']} "
          f"inter={info['intermediate_size']}")

    capture_all(model, tokenizer, LAYERS)

    items = prompt_list()
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(items))
    n_train = int(round(SPLIT_FRAC * len(items)))
    train_pids = set(perm[:n_train].tolist())
    print(f"split by prompt: {n_train} train / {len(items)-n_train} holdout (seed {SPLIT_SEED})")

    import csv as _csv
    rows_csv = []
    summary = {"granularities": GRANULARITIES, "K_frac": K_FRAC,
               "input": "mlp input (post-attention hidden state) — CORRECTED vs M4",
               "split": {"frac": SPLIT_FRAC, "seed": SPLIT_SEED},
               "per_granularity": {}}

    for lk in LAYERS:
        X, Z, pid = load_layer(lk)
        tr_m = np.isin(pid, list(train_pids))
        Xtr, Xho = X[tr_m], X[~tr_m]
        Ztr, Zho = Z[tr_m], Z[~tr_m]
        w = mlp_down_weight(model, lk, device="cpu").float().numpy()  # [hidden, inter]
        t0 = time.time()
        imp_child_tr = child_block_norms(Ztr, w, child=128)
        imp_child_ho = child_block_norms(Zho, w, child=128)
        print(f"{lk}: child norms computed in {time.time()-t0:.0f}s "
              f"(train {imp_child_tr.shape}, holdout {imp_child_ho.shape})")

        for g in GRANULARITIES:
            E = 9728 // g
            K = max(1, int(round(K_FRAC * E)))
            Itr = merge_children(imp_child_tr, g)
            Iho = merge_children(imp_child_ho, g)
            top_tr19 = np.argsort(-Itr, axis=1)[:, :K]
            freq = Counter(top_tr19.ravel().tolist())

            routers = {
                "always_freq": FrequencyRouter(np.array([freq.get(e, 0) for e in range(E)])),
                "random": RandomRouter(E, seed=0),
                "centroid": CentroidRouter(Xtr, top_tr19, E),
                "knn": KNNetRouter(Xtr, Itr, m=16),
                "probe": LinearProbeRouter(Xtr, Itr, epochs=PROBE_EPOCHS),
            }
            # z-energy diagnostic (post-hoc; uses z, not x). z energy per big
            # block = mean z^2 over the block's neurons (z shape known here).
            zenergy_ho = np.argsort(
                -(Zho.astype(np.float32) ** 2)[:, :E * g].reshape(Zho.shape[0], E, g).mean(axis=2),
                axis=1)[:, :K]

            orc_ho = np.argsort(-Iho, axis=1)[:, :K]
            orc_tr = np.argsort(-Itr, axis=1)[:, :K]
            gsum = {}
            for rname, router in routers.items():
                s_tr = router.scores(Xtr) if hasattr(router, "scores") else None
                if s_tr is not None:
                    _p, r_tr = pr_at_k(np.argsort(-s_tr, axis=1)[:, :K], orc_tr)
                else:
                    r_tr = None
                ho_top = router.topk(Xho, K)
                p_ho, r_ho = pr_at_k(ho_top, orc_ho)
                rows_csv.append([lk, g, E, K, rname, r_ho.mean(), p_ho.mean(),
                                 r_tr.mean() if r_tr is not None else -1])
                gsum[rname] = {"holdout_recall": round(float(r_ho.mean()), 4),
                               "holdout_precision": round(float(p_ho.mean()), 4),
                               "train_recall": round(float(r_tr.mean()), 4)
                               if r_tr is not None else None}
                print(f"  {lk} g={g:4d} E={E:3d} K={K:2d} {rname:>12}: "
                      f"holdout {r_ho.mean():.3f}"
                      + (f" (train {r_tr.mean():.3f})" if r_tr is not None else ""))
            p_ze, r_ze = pr_at_k(zenergy_ho, orc_ho)
            rows_csv.append([lk, g, E, K, "z_energy_posthoc", r_ze.mean(), p_ze.mean(), -1])
            gsum["z_energy_posthoc"] = {"holdout_recall": round(float(r_ze.mean()), 4),
                                        "holdout_precision": round(float(p_ze.mean()), 4)}
            print(f"  {lk} g={g:4d} E={E:3d} K={K:2d} {'z_energy':>12}: "
                  f"holdout {r_ze.mean():.3f}  (post-hoc ceiling)")
            summary["per_granularity"].setdefault(lk, {})[str(g)] = gsum

    csv_path = OUT_DIR / "csv" / "router_granularity.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["layer", "granularity", "n_experts", "K", "router",
                    "recall", "precision", "train_recall"])
        w.writerows(rows_csv)

    # ---- verdict per granularity (aggregated over layers) ----
    verdict = {}
    for g in GRANULARITIES:
        E = 9728 // g
        K = max(1, int(round(K_FRAC * E)))
        knn = np.mean([r[5] for r in rows_csv if r[1] == g and r[4] == "knn"])
        stat = np.mean([r[5] for r in rows_csv if r[1] == g and r[4] == "always_freq"])
        ze = np.mean([r[5] for r in rows_csv if r[1] == g and r[4] == "z_energy_posthoc"])
        verdict[str(g)] = {
            "K": K, "n_experts": E,
            "knn_holdout_recall": round(float(knn), 4),
            "static_holdout_recall": round(float(stat), 4),
            "learned_margin_over_static": round(float(knn - stat), 4),
            "z_energy_posthoc_recall": round(float(ze), 4),
            "gate_recall_ge_070": bool(knn >= 0.70),
            "gate_margin_ge_005": bool(knn - stat >= 0.05),
        }
    summary["verdict"] = verdict
    with open(OUT_DIR / "json" / "granularity_router_verdict.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== VERDICT (aggregated over 3 layers) ===")
    for g, v in verdict.items():
        gates = v["gate_recall_ge_070"] and v["gate_margin_ge_005"]
        print(f"  g={g:>4} ({v['n_experts']} experts, K={v['K']}): "
              f"knn {v['knn_holdout_recall']:.3f} vs static {v['static_holdout_recall']:.3f} "
              f"(margin {v['learned_margin_over_static']:+.3f}) | "
              f"z-energy ceiling {v['z_energy_posthoc_recall']:.3f} | "
              f"GATES {'PASS' if gates else 'FAIL'}")
    print(f"\nsaved: {csv_path}")
    print(f"saved: {OUT_DIR / 'json' / 'granularity_router_verdict.json'}")


if __name__ == "__main__":
    main()
