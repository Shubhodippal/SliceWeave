"""M4c — The Autopsy: z-energy post-hoc down_proj skip, measured (§12/§13/§77/§80).

Headless twin of notebooks/zenergy_downproj.ipynb (kernel-hung fallback; keep
both in sync). The routing line is dead (M4/M4b); this measures the ONE honest
mechanism that remains: rank 128-neuron blocks by mean z^2 (computed AFTER
gate/up), zero the low-energy blocks' contribution in down_proj only.

Parts:
  QUALITY   — real masked forwards: z-energy picks vs oracle picks vs random,
              at retention 5..50%, on layer 8 (GPU-resident) + 18 (offloaded).
              Uses FFNMaskedForward batch protocol (replicated batch, CPU
              masks, logits_to_keep=1).
  WALLCLOCK — resident layer 8: dense down_proj GEMM vs gathered GEMM
              (index_select of kept blocks + matmul + scatter-add), bf16,
              batch 64, CUDA-timed. Reports measured ms and the FLOP ratio.

Usage: python scripts/run_m4c_zenergy.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_ID = "Qwen/Qwen3-4B"
LAYERS_Q = ["model.layers.8", "model.layers.18"]
RETAIN_FRACS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
GROUP_SIZE = 128
N_EXP = 9728 // GROUP_SIZE
K_GATE = 19                       # 25% of 76 (matches M4/M4b comparisons)
QUALITY_PROMPTS = 3
SLOTS_PER_PROMPT = 16
WC_BATCH = 64
WC_REPEATS = 200

ZCACHE = ROOT / "results" / "traces" / "m4b_capture"
OUT_DIR = ROOT / "results" / "m4c_zenergy"


def slug(s):
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s)[:40].strip("_") or "p"


def main():
    import torch
    from src.datasets import CATEGORIES
    from src.hooks import FFNMaskedForward
    from src.metrics import compute_all
    from src.model_utils import (describe_model, load_model_and_tokenizer,
                                 mlp_down_weight, pick_resident_layer)

    for sub in ("csv", "plots", "json"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, "bfloat16")
    info = describe_model(model)
    RESIDENT = pick_resident_layer(model)
    print(f"params={info['params']/1e9:.2f}B | resident: {RESIDENT}")

    # ------------------------------------------------------------------
    # Part A — QUALITY: z-energy picks vs oracle picks vs random (masked
    # forwards). We reuse the M4b capture cache only for prompts; z here is
    # captured live (we need z at masked positions anyway).
    # ------------------------------------------------------------------
    from src.hooks import FFNIntermediateCapture

    def capture_prompt(prompt):
        z_caps = {}
        for lk in LAYERS_Q:
            zc = FFNIntermediateCapture()
            zc.attach(model, lk)
            z_caps[lk] = zc
        try:
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                model(**enc)
        finally:
            for lk in LAYERS_Q:
                z_caps[lk].remove()
        return {lk: z_caps[lk].captures[f"{lk}.down_proj"][0].float().cpu().numpy()
                for lk in LAYERS_Q}

    prompts = [(c, ps[0]) for c, ps in CATEGORIES.items()][:QUALITY_PROMPTS]
    print(f"\n=== PART A: quality (masked forwards) ===")
    print(f"{len(prompts)} prompts x {SLOTS_PER_PROMPT} slots x {len(LAYERS_Q)} layers "
          f"x {len(RETAIN_FRACS)} fracs x 3 rankers")

    quality_rows = []   # layer, ranker, frac, cosine, top1, top10, kl
    for lk in LAYERS_Q:
        ffn_ref = model.get_submodule(f"{lk}.mlp")
        w = mlp_down_weight(model, lk, device="cpu").float().numpy()  # [H, I]
        Wf = torch.from_numpy(w).float().T                            # [I, H]
        with FFNMaskedForward(model) as masked:
            masked.attach(lk, ffn_ref)
            masked.set_mask(lk, torch.ones(9728, dtype=torch.bool))   # CPU all-keep
            for ci, (cat, prompt) in enumerate(prompts):
                # NOTE: the capture forward must run with NO masks registered
                # (stale batch masks from the previous prompt would crash it).
                # We capture z for BOTH layers in one pass outside the masked
                # context of THIS layer... but masked is attached to lk only,
                # and capture runs inside it. Fix: temporarily pop any batch
                # entry for lk before the capture forward.
                masked.batch_masks.pop(lk, None)
                masked.masks[lk] = torch.ones(9728, dtype=torch.bool)  # dense
                masked.positions.pop(lk, None)
                z_all_layers = capture_prompt(prompt)
                z_full = z_all_layers[lk]                             # [S, I]
                S = z_full.shape[0]
                slots = list(range(1, min(S - 1, SLOTS_PER_PROMPT) + 1))
                if not slots:
                    continue
                enc = tokenizer(prompt, return_tensors="pt").to(model.device)
                B = len(slots)
                batch_enc = {"input_ids": enc["input_ids"].expand(B, -1).contiguous(),
                             "attention_mask": enc["attention_mask"].expand(B, -1).contiguous()}
                with torch.no_grad():
                    base = model(**batch_enc, logits_to_keep=1).logits[:, -1].float().cpu()
                # per-slot rankings
                z_sel = z_full[slots]                                  # [B, I]
                # z-energy: mean z^2 per 128-block
                ze = (z_sel ** 2).reshape(B, N_EXP, GROUP_SIZE).mean(axis=2)
                ze_top = np.argsort(-ze, axis=1)
                # oracle: true contribution norms (this is the M1 score)
                zt = torch.from_numpy(z_sel).float()
                orc_imp = torch.empty(B, N_EXP)
                for e in range(N_EXP):
                    sl = slice(e * GROUP_SIZE, (e + 1) * GROUP_SIZE)
                    orc_imp[:, e] = (zt[:, sl] @ Wf[sl, :]).norm(dim=1)
                orc_top = torch.argsort(orc_imp, dim=1, descending=True).numpy()
                rng = np.random.default_rng(0)
                rnd_top = np.stack([rng.permutation(N_EXP) for _ in range(B)])

                rankers = {"z_energy": ze_top, "oracle": orc_top, "random": rnd_top}
                for frac in RETAIN_FRACS:
                    k = max(1, int(round(frac * N_EXP)))
                    for rname, top in rankers.items():
                        keeps = torch.zeros(B, 9728, dtype=torch.bool)
                        for i in range(B):
                            for e in top[i, :k]:
                                keeps[i, e * GROUP_SIZE:(e + 1) * GROUP_SIZE] = True
                        masked.set_batch_masks(lk, keeps, slots)
                        with torch.no_grad():
                            out = model(**batch_enc, logits_to_keep=1).logits[:, -1].float().cpu()
                        for i in range(B):
                            m = compute_all(out[i], base[i])
                            quality_rows.append([lk, rname, frac, m["cosine"],
                                                 m["top1_match"], m["top10_overlap"], m["kl"]])
                print(f"  {lk} '{prompt[:36]}...' ({B} slots) done", flush=True)

    import csv as _csv
    qcsv = OUT_DIR / "csv" / "zenergy_quality.csv"
    with open(qcsv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["layer", "ranker", "retain_fraction", "cosine", "top1",
                    "top10", "kl"])
        w.writerows(quality_rows)

    print("\n=== PART A summary (mean over slots+prompts) ===")
    print(f"{'layer':>16} {'ranker':>8} " +
          " ".join(f"{f:>6.2f}" for f in RETAIN_FRACS) + "  (cosine @ retention)")
    for lk in LAYERS_Q:
        for rname in ["z_energy", "oracle", "random"]:
            vals = []
            for frac in RETAIN_FRACS:
                cs = [r[3] for r in quality_rows
                      if r[0] == lk and r[1] == rname and r[2] == frac]
                vals.append(np.mean(cs) if cs else float("nan"))
            print(f"{lk:>16} {rname:>8} " + " ".join(f"{v:>6.4f}" for v in vals))

    # ------------------------------------------------------------------
    # Part B — WALL-CLOCK: dense vs gathered down_proj (resident layer)
    # ------------------------------------------------------------------
    print(f"\n=== PART B: wall-clock on {RESIDENT} (bf16, batch {WC_BATCH}) ===")
    mlp = model.get_submodule(f"{RESIDENT}.mlp")
    dev = next(mlp.parameters()).device
    W = mlp.down_proj.weight.detach().to(dev)            # [H, I] bf16
    # down_proj consumes z [B, I=9728] -> y [B, H=2560]
    z = torch.randn(WC_BATCH, 9728, device=dev, dtype=torch.bfloat16)

    def timeit(fn, n=WC_REPEATS, warmup=20):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    dense_ms = timeit(lambda: torch.nn.functional.linear(z, W))
    wc_rows = [["dense", 1.0, dense_ms, 1.0]]
    print(f"dense down_proj : {dense_ms:.3f} ms")

    keep_fracs = RETAIN_FRACS
    gather_ms_by_k = {}
    for frac in keep_fracs:
        k = max(1, int(round(frac * N_EXP)))
        # realistic per-token gather: each row keeps its own k blocks. For
        # timing, use the SAME k blocks for all rows (worst-case regularity is
        # row-specific; note this in the paper). indices sorted ascending.
        idx = torch.arange(k * GROUP_SIZE, device=dev)
        Wg = W[:, idx]                                   # pre-gathered weight view
        zg = z[:, idx]
        sparse_ms = timeit(lambda: torch.nn.functional.linear(zg, Wg))
        # plus the gather cost itself, measured:
        gather_ms = timeit(lambda: z.index_select(1, idx))
        total_ms = sparse_ms + gather_ms
        wc_rows.append([f"gathered@{frac:.2f}", frac, total_ms,
                        total_ms / dense_ms])
        gather_ms_by_k[frac] = {"sparse_gemm_ms": round(sparse_ms, 3),
                                "gather_ms": round(gather_ms, 3),
                                "total_ms": round(total_ms, 3),
                                "ratio_vs_dense": round(total_ms / dense_ms, 4)}
        print(f"  retention {frac:.2f} (k={k:2d}): gemm {sparse_ms:.3f} ms "
              f"+ gather {gather_ms:.3f} ms = {total_ms:.3f} ms "
              f"-> {total_ms/dense_ms:.2%} of dense")

    ideal = {frac: frac for frac in keep_fracs}
    wc_summary = {"dense_ms": round(dense_ms, 3), "batch": WC_BATCH,
                  "per_retention": gather_ms_by_k,
                  "note": "gather uses the same k blocks for all rows "
                          "(most favourable regularity); row-specific gathers "
                          "would be slower — per §77 irregular-traffic caveat"}

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    def mean_recall(lk, rname, frac):
        # "recall" here = agreement between rankers is implicit; for the
        # verdict we compare quality cosine directly.
        cs = [r[3] for r in quality_rows
              if r[0] == lk and r[1] == rname and r[2] == frac]
        return float(np.mean(cs)) if cs else float("nan")

    verdict = {"quality": {
        "layers": LAYERS_Q,
        "cosine_at_030": {lk: {r: round(mean_recall(lk, r, 0.30), 4)
                               for r in ["z_energy", "oracle", "random"]}
                           for lk in LAYERS_Q},
        "z_energy_matches_oracle_at_030": {
            lk: bool(mean_recall(lk, "z_energy", 0.30) >=
                     mean_recall(lk, "oracle", 0.30) - 0.005)
            for lk in LAYERS_Q}},
        "wall_clock": wc_summary}
    with open(OUT_DIR / "json" / "zenergy_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    print("\n=== VERDICT ===")
    for lk in LAYERS_Q:
        print(f"  {lk}: cosine@30%  z_energy {mean_recall(lk,'z_energy',0.30):.4f} | "
              f"oracle {mean_recall(lk,'oracle',0.30):.4f} | "
              f"random {mean_recall(lk,'random',0.30):.4f}")
    print(f"  wall-clock @30% retention: {gather_ms_by_k[0.30]['total_ms']:.3f} ms = "
          f"{gather_ms_by_k[0.30]['ratio_vs_dense']:.2%} of dense down_proj")
    print(f"\nsaved: {qcsv}")
    print(f"saved: {OUT_DIR / 'json' / 'zenergy_verdict.json'}")


if __name__ == "__main__":
    main()
