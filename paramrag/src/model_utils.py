"""Model loading and FFN module discovery (Blueprint §7, §28)."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(hf_id: str, dtype: str = "bfloat16", device: str = "auto"):
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    kwargs = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    if device == "auto":
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()
    return model, tokenizer


def find_ffn_modules(model):
    """Return {module_name: module} for all gate/up/down FFN projections.

    Works with Qwen-family models (Qwen2/Qwen3 blocks use gate_proj/up_proj/down_proj)
    and most other gated-FFN architectures.
    """
    ffn = {}
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in ("gate_proj", "up_proj", "down_proj"):
            ffn[name] = module
    return ffn


def ffn_layer_names(ffn_modules: dict) -> list:
    """Group projection names per transformer layer: {'model.layers.5': {...}}."""
    layers = {}
    for name in ffn_modules:
        parts = name.split(".")
        # e.g. model.layers.5.mlp.gate_proj -> layer = model.layers.5
        layer_key = ".".join(parts[:-2])
        proj = parts[-1]
        layers.setdefault(layer_key, {})[proj] = name
    return layers


def get_module(model, name: str):
    obj = model
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def describe_model(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    info = {
        "params": total,
        "layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
    }
    return info
