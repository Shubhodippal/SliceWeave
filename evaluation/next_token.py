"""Next-token agreement evaluation (§50).

Compare the sparse/conditioned model against the SAME quantized parent baseline
(never compare Q4-sparse against BF16-dense and attribute the gap to routing).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from src.hooks import FFNMaskedForward
from src.metrics import compute_all
from src.model_utils import load_model_and_tokenizer


def evaluate_next_token(model, tokenizer, prompts: List[str],
                        mask_factory: Optional[Callable] = None,
                        out_csv: Optional[str] = None) -> List[Dict]:
    """mask_factory(prompt, model) -> list of (layer_key, keep_mask) or None.

    None = dense baseline evaluation (useful as a sanity anchor).
    """
    rows: List[Dict] = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            dense = model(**enc).logits[0, -1].float().cpu()

        if mask_factory is None:
            m = compute_all(dense, dense)
            m["mode"] = "dense"
        else:
            pairs = mask_factory(prompt, model)
            with FFNMaskedForward(model) as masked:
                for layer_key, keep in pairs:
                    masked.attach(layer_key, model.get_submodule(f"{layer_key}.mlp"))
                    masked.set_mask(layer_key, keep)
                with torch.no_grad():
                    sparse = model(**enc).logits[0, -1].float().cpu()
            m = compute_all(sparse, dense)
            m["mode"] = "sparse"
        m["prompt"] = prompt[:80]
        rows.append(m)

    if out_csv:
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows
