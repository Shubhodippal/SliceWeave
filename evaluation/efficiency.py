"""Generation agreement + efficiency reporting (§35-§43, §51, Milestone 10).

Reports the paper's core efficiency table: APR (§37), throughput, latency
percentiles, memory traffic — always next to quality, never alone.
"""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional


def generation_agreement(model, tokenizer, prompts: List[str], max_new: int,
                         sparse_generate: Optional[Callable] = None,
                         out_csv: Optional[str] = None) -> List[Dict]:
    """Compare full generations (§51). sparse_generate(model, prompt) -> ids.

    Lexical divergence is reported but NOT treated as quality by itself — pair
    with executable/benchmark scores (§51).
    """
    rows = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        dense = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        dense_ids = dense[0, enc["input_ids"].shape[1]:].tolist()
        row = {"prompt": prompt[:80], "gen_len": len(dense_ids)}

        if sparse_generate is not None:
            sparse_ids = sparse_generate(model, prompt)
            n = min(len(dense_ids), len(sparse_ids))
            row["divergence_token"] = next(
                (i for i in range(n) if sparse_ids[i] != dense_ids[i]), n)
            row["exact_prefix_agreement"] = row["divergence_token"] / max(len(dense_ids), 1)
        rows.append(row)

    if out_csv:
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def throughput_report(latencies_s: List[float], tokens: List[int],
                      out_csv: Optional[str] = None) -> Dict[str, float]:
    """§35/§97: mean/median/P95 with warmup discarded by the caller."""
    tok_s = [n / t for n, t in zip(tokens, latencies_s) if t > 0 and n > 0]
    lat_ms = [t * 1e3 / max(n, 1) for n, t in zip(tokens, latencies_s)]
    out = {
        "tok_s_mean": statistics.mean(tok_s),
        "tok_s_median": statistics.median(tok_s),
        "latency_p50_ms": statistics.median(lat_ms),
        "latency_p95_ms": sorted(lat_ms)[min(len(lat_ms) - 1, int(0.95 * len(lat_ms)))],
        "n_runs": len(tok_s),
    }
    if out_csv:
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in out.items():
                w.writerow([k, f"{v:.4f}"])
    return out


def quality_adjusted_throughput(tok_s: float, retention: float) -> float:
    """§40 QAT — supplementary visualization metric, never a replacement."""
    from src.metrics import qat
    return qat(tok_s, retention)
