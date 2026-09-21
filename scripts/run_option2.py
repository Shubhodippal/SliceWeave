"""Option 2 — last granularity attempt (arm 2a: finer k; arm 2b: token-space MI).

Headless twin of notebooks/option2_granularity.ipynb. Saves ALL artifacts to
results/m3b_granularity_option2/ and appends to results/RUN_MANIFEST.md.

Usage:
    python scripts/run_option2.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", r"E:\hf_cache")

import torch  # noqa: E402

from src.clustering import cluster_neurons  # noqa: E402
from src.datasets import CATEGORIES  # noqa: E402
from src.micro_experts import (contiguous_partition,  # noqa: E402
                               expert_importance_output)
from src.model_utils import (load_model_and_tokenizer,  # noqa: E402
                             mlp_down_weight)

M3B = ROOT / "results" / "m3b_granularity_option2"
for sub in ("csv", "plots", "json"):
    (M3B / sub).mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / "results" / "traces" / "m3_capture"
RUN_STAMP = datetime.now().strftime("%Y-%m-%d %H:%M")

LAYERS = ["model.layers.7", "model.layers.18", "model.layers.29"]
FINE_KS = [608, 1216, 2432]
BASE_K = 304
QUANTILE = 0.5
TOKEN_CLUSTER_KS = [15, 40, 76]
KMEANS_SEED = 0
MI_PERMUTATIONS = 200


def pooled_entropy(z_by_task, groups, quantile=0.5):
    energies = []
    for Z in z_by_task.values():
        e = np.asarray(Z, np.float32) ** 2
        energies.append(np.stack([e[:, g].mean(1) for g in groups], axis=1))
    allE = np.concatenate(energies, axis=0)
    thr = np.quantile(allE, quantile, axis=0, keepdims=True)
    P = np.stack([(E > thr).mean(0) for E in energies], axis=0)
    p = np.clip(P, 1e-12, 1)
    H = -(p * np.log2(p)).sum(0) / np.log2(P.shape[0])
    return H, P


def mutual_info(cluster_ids, task_labels):
    ci = {c: i for i, c in enumerate(sorted(set(cluster_ids)))}
    ti = {t: i for i, t in enumerate(sorted(set(task_labels)))}
    J = np.zeros((len(ci), len(ti)))
    for c, t in zip(cluster_ids, task_labels):
        J[ci[c], ti[t]] += 1
    J /= J.sum()
    pc, pt = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
    indep = pc @ pt
    mask = J > 0
    return float((J[mask] * np.log2(J[mask] / indep[mask])).sum())


def main() -> None:
    # ---- load cached captures + labels -------------------------------
    z_store, labels = {}, {}
    for lk in LAYERS:
        mats, labs = [], []
        for task in CATEGORIES:
            f = CACHE_DIR / f"{lk.replace('.', '_')}__{task}.npy"
            if not f.exists():
                raise FileNotFoundError(f"missing capture {f} (run M3 notebook step 4)")
            Z = np.load(f)
            mats.append(Z)
            labs += [task] * Z.shape[0]
        z_store[lk] = np.concatenate(mats, axis=0)
        labels[lk] = np.array(labs)
        print(f"loaded {lk}: {z_store[lk].shape[0]} tokens")

    # ---- arm 2a: finer k ---------------------------------------------
    print("\n=== arm 2a: finer-k clustering ===")
    arm2a = []
    for lk in LAYERS:
        Z = z_store[lk]
        z_by_task = {t: Z[labels[lk] == t] for t in CATEGORIES}
        for k in [BASE_K] + FINE_KS:
            a = cluster_neurons(Z, n_clusters=k, method="kmeans",
                                random_state=KMEANS_SEED)
            H, _ = pooled_entropy(z_by_task,
                                  [list(np.where(a == c)[0]) for c in range(k)],
                                  QUANTILE)
            arm2a.append({"layer": lk, "k": k, "meanH": float(H.mean()),
                          "frac_specialized": float((H < 0.75).mean()),
                          "neurons_per_expert": int(Z.shape[1] // k),
                          "tokens_per_expert": round(Z.shape[0] / k, 2)})
            print(f"{lk} k={k:4d} ({Z.shape[1]//k:3d} neu/exp, "
                  f"{Z.shape[0]/k:6.1f} tok/exp): meanH={H.mean():.3f} "
                  f"spec={(H < 0.75).mean():.0%}")

    # ---- arm 2b: token-space MI ---------------------------------------
    print("\n=== arm 2b: token-routing-fingerprint MI ===")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen3-4B", "bfloat16")
    contig76 = contiguous_partition(9728, 128)
    contrib = {}
    for lk in LAYERS:
        w = mlp_down_weight(model, lk, device=next(model.parameters()).device)
        Zt = torch.from_numpy(z_store[lk]).to(w.device)
        C = expert_importance_output(Zt, w, contig76)
        contrib[lk] = C.float().cpu().numpy().astype(np.float32)
        print(f"{lk}: contribution matrix {contrib[lk].shape}")

    rng = np.random.default_rng(KMEANS_SEED)
    arm2b = []
    for lk in LAYERS:
        X = contrib[lk]
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
        labs = labels[lk]
        H_task = mutual_info(labs.tolist(), labs.tolist())
        for k in TOKEN_CLUSTER_KS:
            a = cluster_neurons(Xn, n_clusters=k, method="kmeans",
                                random_state=KMEANS_SEED)
            mi = mutual_info(a.tolist(), labs.tolist())
            null = []
            for _ in range(MI_PERMUTATIONS):
                perm = rng.permutation(len(labs))
                null.append(mutual_info(a.tolist(), labs[perm].tolist()))
            pval = float((np.array(null) >= mi).mean())
            arm2b.append({"layer": lk, "k": k, "MI_bits": mi,
                          "task_entropy_bits": H_task,
                          "frac_MI_explained": mi / max(H_task, 1e-9),
                          "p_value": pval})
            print(f"{lk} k={k:2d}: MI={mi:.3f}/{H_task:.3f} bits "
                  f"({mi / max(H_task, 1e-9):.0%}), perm-p={pval:.3f}")

    # ---- plots ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for lk in LAYERS:
        pts = sorted([(r["k"], r["meanH"]) for r in arm2a if r["layer"] == lk])
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                     label=lk.split(".")[-1])
        pts = sorted([(r["k"], r["frac_MI_explained"]) for r in arm2b
                      if r["layer"] == lk])
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], marker="s",
                     label=lk.split(".")[-1])
    axes[0].axhline(1.85, ls="--", c="gray", lw=1, label="M3 null level")
    axes[0].set_xscale("log"); axes[0].set_xlabel("cluster count k (log)")
    axes[0].set_ylabel("mean task entropy"); axes[0].set_title("arm 2a: finer clusters")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("token-cluster count k")
    axes[1].set_ylabel("MI / task entropy")
    axes[1].set_title("arm 2b: routing-fingerprint task information")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(M3B / "plots" / "option2_summary.png", dpi=150)
    plt.close(fig)

    # ---- save everything -----------------------------------------------
    with open(M3B / "csv" / "arm2a_fine_k.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(arm2a[0].keys()))
        w.writeheader(); w.writerows(arm2a)
    with open(M3B / "csv" / "arm2b_token_mi.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(arm2b[0].keys()))
        w.writeheader(); w.writerows(arm2b)

    best_spec = max(r["frac_specialized"] for r in arm2a)
    best_mi = max(r["frac_MI_explained"] for r in arm2b)
    best_p = min(r["p_value"] for r in arm2b)
    with open(M3B / "json" / "option2_verdict.json", "w") as f:
        json.dump({"run_stamp": RUN_STAMP,
                   "arm2a_best_frac_specialized": best_spec,
                   "arm2b_best_frac_MI_explained": best_mi,
                   "arm2b_best_p_value": best_p,
                   "arm2a": arm2a, "arm2b": arm2b}, f, indent=2)

    with open(ROOT / "results" / "RUN_MANIFEST.md", "a", encoding="utf-8") as f:
        f.write(f"| {RUN_STAMP} | M3b option-2 (headless) | finer-k + token-MI | "
                f"2a spec={best_spec:.0%}, 2b MI={best_mi:.0%} (p={best_p:.3f}) | "
                f"scripts/run_option2.py | m3b_granularity_option2/ |\n")

    print(f"\n=== VERDICT ===\narm 2a best specialized fraction: {best_spec:.0%}")
    print(f"arm 2b best MI explained: {best_mi:.0%} of task entropy (p={best_p:.3f})")
    if best_mi > 0.30 and best_p < 0.05:
        print("-> TASK INFORMATION EXISTS in routing fingerprints. "
              "Revisit prompt-level routing.")
    elif best_mi > 0.10 and best_p < 0.05:
        print("-> WEAK task information: significant but tiny. Token-level router "
              "(M4) remains the plan; note as caveat.")
    else:
        print("-> CONFIRMED NULL: no task specialization in routing space either. "
              "Pivot to token-level router (M4) per §55.7.")
    print("artifacts:", M3B)


if __name__ == "__main__":
    main()
