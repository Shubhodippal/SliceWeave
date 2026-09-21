"""M4 Step 1-3 (headless twin of notebooks/token_router.ipynb steps 1-4).

The VS Code notebook kernel hangs on this machine (known issue), so capture
runs headless. Logic is IDENTICAL to the notebook cells and must stay in sync.

Captures PAIRED data per (prompt, layer): for every analyzed token,
  x   = incoming hidden state (decoder-layer input, BEFORE the FFN)
  imp = oracle block-importance ||W_down z_e|| over 76 contiguous 128-neuron
        blocks (M1's output-mode score, computed on the fly from z)
Both captured in ONE forward pass per prompt (input hook + down_proj hook on
all 3 layers simultaneously). Disk-cached per (prompt, layer) under
results/traces/m4_capture/ and resumable: existing files are skipped.

Usage:
    python scripts/run_m4_capture.py            # full capture (~25-40 min)
    python scripts/run_m4_capture.py --max 4    # pilot: first 4 prompts only
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HOME", r"E:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=0, help="pilot: cap number of prompts")
    args = ap.parse_args()

    import numpy as np
    import torch
    from tqdm.auto import tqdm

    from src.datasets import CATEGORIES, generated_arithmetic
    from src.hooks import FFNIntermediateCapture
    from src.micro_experts import contiguous_partition, expert_importance_output
    from src.model_utils import (describe_model, load_model_and_tokenizer,
                                 mlp_down_weight, pick_resident_layer)

    MODEL_ID = "Qwen/Qwen3-4B"
    LAYERS = ["model.layers.7", "model.layers.18", "model.layers.29"]
    GROUP_SIZE = 128
    CAPTURE_PROMPTS = 8      # first-N per category (M2/M3 protocol)
    MATH_PROMPTS = 20        # generated arithmetic for cheap scale

    CACHE_DIR = ROOT / "results" / "traces" / "m4_capture"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("repo root :", ROOT)
    print("HF cache  :", os.environ["HF_HOME"])
    print("cache dir :", CACHE_DIR)

    print(f"Loading {MODEL_ID} ...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, "bfloat16")
    info = describe_model(model)
    INTER, HIDDEN = info["intermediate_size"], info["hidden_size"]
    assert INTER % GROUP_SIZE == 0
    N_EXP = INTER // GROUP_SIZE
    GROUPS = contiguous_partition(INTER, GROUP_SIZE)
    print(f"loaded in {time.time()-t0:.0f}s | params={info['params']/1e9:.2f}B "
          f"hidden={HIDDEN} inter={INTER} -> {N_EXP} blocks")
    print("resident layer:", pick_resident_layer(model))

    def slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", s)[:40].strip("_") or "p"

    def capture_prompt(prompt: str) -> dict:
        """One forward pass -> per-layer {x:[T-1,H] f16, imp:[T-1,E] f32}.

        Token 0 (BOS, no FFN context) is dropped; under teacher forcing logits
        at t predict token t+1, so x at t is exactly the inference-time router
        input for that token.
        """
        x_store: dict = {}
        handles = []

        def make_x_hook(key):
            def _hook(mod, args, kwargs=None):
                # torch>=2.9 passes (mod, args); older passes (mod, args, kwargs)
                hs = args[0] if args else kwargs.get("hidden_states")
                x_store[key] = hs.detach()
                return None  # do not alter the forward
            return _hook

        z_caps = {}
        for lk in LAYERS:
            layer = model.get_submodule(lk)
            handles.append(layer.register_forward_pre_hook(
                make_x_hook(lk), with_kwargs=True))
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
            for lk in LAYERS:
                z_caps[lk].remove()
        out = {}
        for lk in LAYERS:
            x = x_store[lk][0, 1:].float()                             # [T, H]
            z = z_caps[lk].captures[f"{lk}.down_proj"][0, 1:].float()  # [T, I]
            w = mlp_down_weight(model, lk, device=z.device)            # real tensor
            imp = expert_importance_output(z, w, GROUPS)               # [T, E]
            out[lk] = {"x": x.cpu().numpy().astype(np.float16),
                       "imp": imp.cpu().numpy().astype(np.float32)}
        return out

    def cache_path(kind: str, cat: str, prompt: str, lk: str) -> Path:
        return CACHE_DIR / f"{lk.replace('.', '_')}__{kind}_{slug(cat)}__{slug(prompt)}.npz"

    items = [("cat", c, p) for c, ps in CATEGORIES.items() for p in ps[:CAPTURE_PROMPTS]]
    items += [("math", "math_gen", p) for p in generated_arithmetic(MATH_PROMPTS, seed=7)]
    if args.max:
        items = items[: args.max]
    total_files = len(items) * len(LAYERS)
    done0 = sum(1 for kind, c, p in items for lk in LAYERS
                if cache_path(kind, c, p, lk).exists())
    print(f"capture plan: {len(items)} prompts x {len(LAYERS)} layers = "
          f"{total_files} files; {done0} already cached -> resuming")

    t_cap = time.time()
    pbar = tqdm(items, unit="prompt")
    n_new = 0
    for kind, cat, p in pbar:
        per_layer = None
        for lk in LAYERS:
            if cache_path(kind, cat, p, lk).exists():
                continue
            if per_layer is None:
                per_layer = capture_prompt(p)          # ONE forward for all layers
            np.savez_compressed(cache_path(kind, cat, p, lk), **per_layer[lk])
            n_new += 1
        if per_layer is not None:
            pbar.set_postfix_str(
                f"{cat}: {sum(v['x'].shape[0] for v in per_layer.values())} tok/layer")
    pbar.close()

    files = sorted(CACHE_DIR.glob("*.npz"))
    n_tok = sum(np.load(f)["x"].shape[0] for f in files)
    mins = (time.time() - t_cap) / 60
    print(f"capture complete: {len(files)} files ({n_new} new), {n_tok:,} rows, "
          f"{mins:.1f} min this session (cached under m4_capture/)")


if __name__ == "__main__":
    main()
