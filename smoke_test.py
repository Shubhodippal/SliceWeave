"""Environment + pipeline smoke test (run before real experiments).

Validates, with a tiny locally-created Qwen3 model (no download):
  1. CUDA availability and a real GPU computation
  2. Model load with VRAM+RAM budgeting via device_map="auto"
  3. FFN discovery (src/ffns.py)
  4. Milestone 0 exactness: all-experts sum == dense; permutation == dense
  5. Full oracle path: capture z, rank micro-experts, mask, compare logits
  6. 100% retention == baseline (oracle sanity check)
  7. Centroid router + quality guard + recovery record pipeline
  8. Cache/prefetch/scheduler simulations run end-to-end

Usage:
    python smoke_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cache import TieredCache, trace_from_lists  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.micro_experts import (contiguous_partition, exact_decomposition_check,  # noqa: E402
                               expert_importance_output, verify_reorder_equivalence)
from src.model_utils import (_auto_max_memory, describe_model,  # noqa: E402
                             find_ffn_modules, load_model_and_tokenizer)
from src.oracle import oracle_single_layer  # noqa: E402
from src.prefetch import CrossLayerPredictor, NextTokenPredictor  # noqa: E402
from src.quality_guard import GuardConfig, QualityGuard, recovery_summary  # noqa: E402
from src.scheduler import BlockMeta, CostModel, Scheduler, Tier  # noqa: E402
from src.token_router import CentroidRouter  # noqa: E402
from src.ffns import discover_ffns  # noqa: E402


def main():
    print("=== SliceWeave smoke test (ParamStream update) ===")
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1024**3:.1f} GiB")
    print("auto memory budget:", _auto_max_memory())

    # --- build a tiny Qwen3 model locally ---
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(
        "qwen3", hidden_size=64, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=256,
    )
    with tempfile.TemporaryDirectory() as d:
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        model.save_pretrained(d)
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        vocab = {"<pad>": 0, "<unk>": 1}
        for i in range(32, 127):
            vocab[chr(i)] = len(vocab)
        wt = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        wt.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        PreTrainedTokenizerFast(tokenizer_object=wt, pad_token="<pad>",
                                unk_token="<unk>").save_pretrained(d)

        model, tokenizer = load_model_and_tokenizer(d, "bfloat16")
        info = describe_model(model)
        print(f"\ntiny model: params={info['params']} layers={info['layers']} "
              f"hidden={info['hidden_size']} inter={info['intermediate_size']}")

        # --- FFN discovery (new ffns.py + classic path) ---
        ffn_infos = discover_ffns(model, group_size=64)
        assert len(ffn_infos) == info["layers"]
        ffn_modules = find_ffn_modules(model)
        assert len(ffn_modules) == info["layers"] * 3
        print(f"FFN discovery: {len(ffn_infos)} layers x {ffn_infos['model.layers.0'].n_experts} experts  OK")

        # --- Milestone 0: exactness ---
        ffn = model.get_submodule("model.layers.1.mlp")
        groups = contiguous_partition(info["intermediate_size"], 64)
        probe = torch.randn(2, 3, info["hidden_size"])
        ex = exact_decomposition_check(ffn, probe, groups)
        assert ex["sum_equals_dense"] and ex["permuted_equals_dense"], ex
        assert verify_reorder_equivalence(ffn, probe, groups)
        print(f"exactness: sum==dense & perm==dense (rel_err {ex['rel_err_sum']:.2e})  OK")

        # --- oracle pipeline ---
        layer_key = "model.layers.1"
        results = oracle_single_layer(model, tokenizer, "test prompt for smoke test",
                                      layer_key, [1.0, 0.5, 0.1], group_size=64,
                                      score_mode="output")
        by_frac = {r["retain_fraction"]: r for r in results}
        m = compute_all(by_frac[1.0]["logits"], by_frac[1.0]["baseline_logits"])
        assert m["cosine"] > 0.9999 and m["rel_l2"] < 1e-3 and m["top1_match"] == 1, \
            f"100% retention diverged from baseline: {m}"
        print(f"oracle: 100% retention == baseline (cosine={m['cosine']:.6f})  OK")
        for frac in (0.5, 0.1):
            mm = compute_all(by_frac[frac]["logits"], by_frac[frac]["baseline_logits"])
            print(f"  {frac:.0%}: cosine={mm['cosine']:.4f} top1={mm['top1_match']} kl={mm['kl']:.4f}")

        # --- router + quality guard ---
        E = len(groups)
        router = CentroidRouter(layer_key, E, info["hidden_size"])
        # calibrate on the captured z-energy directions (tiny-model stand-in)
        from src.hooks import FFNIntermediateCapture, HiddenStateCapture
        enc = tokenizer("test prompt for smoke test", return_tensors="pt").to(model.device)

        hs = HiddenStateCapture()
        hs.attach(model, layer_key)
        cz = FFNIntermediateCapture()
        cz.attach(model, layer_key)
        with torch.no_grad():
            model(**enc)
        hs.remove()
        cz.remove()
        x = hs.captures[f"{layer_key}.input"][0, -1]
        zt = cz.captures[f"{layer_key}.down_proj"][0, -1]
        imp = expert_importance_output(zt[None, :], ffn.down_proj.weight, groups)[0]
        top = torch.topk(imp, E // 4).indices.tolist()
        for _ in range(3):
            router.add_observation(x, top)
        router.finalize()
        guard = QualityGuard(router, GuardConfig(initial_fraction=0.25,
                                                 confidence_threshold=0.6))
        sel, rec = guard.run(x)
        assert len(sel) >= rec.initial_k and rec.final_k <= E
        s = recovery_summary([rec])
        print(f"guard: initial={rec.initial_k} final={rec.final_k} "
              f"expansions={rec.expansions} fallback={rec.fallback_full}  OK")

        # --- cache / prefetch / scheduler simulations ---
        trace = trace_from_lists([layer_key], [[list(range(E // 4))]], [1],
                                 sizes={(layer_key, e): 4096 for e in range(E)})
        cache = TieredCache(vram_capacity=16 * 4096, ram_capacity=64 * 4096,
                            policy="lru", block_bytes=4096)
        tier = cache.access(trace[0])
        assert tier in ("vram", "ram", "ssd")
        hr = cache.stats.hit_rates()

        clp = CrossLayerPredictor(); clp.observe(trace)
        ntp = NextTokenPredictor(); ntp.observe(trace)
        assert isinstance(clp.predict((layer_key, 0), layer_key), set)
        assert isinstance(ntp.predict({(layer_key, 0)}), set)

        sched = Scheduler(CostModel(), vram_budget_bytes=1 << 20)
        for e in range(E):
            sched.register(BlockMeta(layer_key=layer_key, expert_id=e,
                                     size_bytes=4096, tier=Tier.RAM,
                                     activation_frequency=1.0 / (e + 1)))
        plan = sched.token_plan(layer_key, list(range(E // 4)), 64, info["hidden_size"],
                                max_blocks=E // 4)
        assert len(plan) == E // 4 and all(d.est_latency_ms >= 0 for d in plan)
        print(f"cache tier={tier} hits={hr}  prefetchers OK  "
              f"plan={len(plan)} blocks (GPU={sum(1 for d in plan if d.device.value=='gpu')})  OK")

        print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
