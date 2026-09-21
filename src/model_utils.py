"""Model loading and FFN module discovery (§7, §28, §71 dev stack)."""
from __future__ import annotations

import torch

from ._sklearn_shim import SHIM_INSTALLED  # noqa: F401  (must precede transformers)
from transformers import AutoModelForCausalLM, AutoTokenizer


def _auto_max_memory() -> dict:
    """Blueprint §24 memory organization on the local machine:

    GPU VRAM: hot/shared blocks + current KV/state
    System RAM: everything else (cold parameters, offloaded layers)

    device_map='auto' with an explicit budget keeps ~5 GiB of weights on the
    6 GiB RTX 3060 (leaving room for activations/KV) and spills the rest into
    the 32 GB system RAM, so a 4B bf16 model (~8 GiB) runs without OOM.
    """
    max_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            # leave headroom for activations, KV cache, CUDA context (~1.2 GiB)
            max_memory[i] = f"{max(1, int(total_gib - 1.2))}GiB"
    max_memory["cpu"] = "24GiB"  # cap CPU offload well below physical RAM
    return max_memory


def load_model_and_tokenizer(hf_id: str, dtype: str = "bfloat16", device: str = "auto",
                             max_memory: bool = True, revision: str | None = None,
                             cache_dir: str | None = None, local_files_only: bool = False):
    """Load a causal LM and tokenizer with reproducibility/cache controls.

    ``revision`` accepts a commit hash so a notebook can pin the exact model
    snapshot it records in its run manifest.  ``cache_dir`` lets each model
    study keep downloaded weights inside its own result bundle rather than a
    shared Hugging Face cache.  Existing callers retain the original defaults.
    """
    torch_dtype = getattr(torch, dtype)
    source_kwargs = {
        k: v for k, v in {
            "revision": revision,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }.items() if v is not None and v is not False
    }
    tokenizer = AutoTokenizer.from_pretrained(hf_id, **source_kwargs)
    kwargs = dict(low_cpu_mem_usage=True, **source_kwargs)
    # transformers 5.x renamed `torch_dtype` -> `dtype`; accept both.
    try:
        model = _load(hf_id, torch_dtype, dtype_kw="dtype", device=device,
                      max_memory=max_memory, extra=kwargs)
    except (TypeError, ValueError):
        model = _load(hf_id, torch_dtype, dtype_kw="torch_dtype", device=device,
                      max_memory=max_memory, extra=kwargs)
    model.eval()
    return model, tokenizer


def _load(hf_id, torch_dtype, dtype_kw, device, max_memory, extra):
    kwargs = dict(extra)
    kwargs[dtype_kw] = torch_dtype
    if device == "auto":
        kwargs["device_map"] = "auto"
        if max_memory:
            kwargs["max_memory"] = _auto_max_memory()
    else:
        kwargs["device_map"] = device
    return AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)


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


def ffn_layer_names(ffn_modules: dict) -> dict:
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


def get_mlp(model, layer_key: str):
    """Typed accessor for a layer's gated-FFN module (gate_proj/up_proj/down_proj).

    get_submodule() returns an untyped nn.Module, which trips static checks;
    attribute access via getattr keeps type checkers happy and matches Qwen
    MLP structure.
    """
    mlp = get_module(model, f"{layer_key}.mlp")
    assert hasattr(mlp, "down_proj"), f"{layer_key}.mlp is not a gated FFN"
    return mlp


def mlp_intermediate_size(model, layer_key: str) -> int:
    return int(getattr(getattr(get_mlp(model, layer_key), "down_proj"), "weight").shape[1])


def mlp_down_weight(model, layer_key: str, device=None) -> torch.Tensor:
    """down_proj weight as a REAL tensor (never meta).

    With device_map='auto', accelerate offloads whole layers to CPU/disk: their
    parameters sit on the meta device between forwards and are materialized
    only inside forward hooks. For weight-level analysis we resolve the true
    data through the module's AlignDevicesHook.weights_map (accelerate's own
    offload container) instead of failing on meta tensors.
    """
    p = getattr(getattr(get_mlp(model, layer_key), "down_proj"), "weight")
    w = _materialize_param(model, p)
    return w if device is None else w.to(device)


def _materialize_param(model, p: torch.Tensor) -> torch.Tensor:
    """Return real parameter data; resolve meta params via the offload hook."""
    if p.device.type != "meta":
        return p.detach()
    # locate the module in the tree to know its name path
    target = None
    for name, mod in model.named_modules():
        if any(getattr(mod, a, None) is p or getattr(getattr(mod, a, None), "weight", None) is p
               for a in ()) or mod is p:
            target = name
    # p is a Parameter: find its owning module + attr name
    owner_name = owner_attr = None
    for name, mod in model.named_modules():
        for attr, val in list(mod._parameters.items()) + list(getattr(mod, "_buffers", {}).items()):
            if val is p:
                owner_name, owner_attr = name, attr
                break
        if owner_name is not None:
            break
    if owner_name is None:
        raise RuntimeError("could not locate offloaded parameter in the module tree")
    parts = owner_name.split(".")
    # try each ancestor's hook, widest-first (weights_map keys are relative)
    for depth in range(len(parts), -1, -1):
        anc_name = ".".join(parts[:depth])
        anc = model.get_submodule(anc_name) if anc_name else model
        hook = getattr(anc, "_hf_hook", None)
        wm = getattr(hook, "weights_map", None)
        if wm is None:
            continue
        rel = ".".join(parts[depth:] + [owner_attr]) if depth <= len(parts) else owner_attr
        for key in (rel, owner_attr):
            try:
                if key in wm:
                    return wm[key]
            except Exception:
                try:
                    return wm[key]
                except Exception:
                    continue
    raise RuntimeError(
        f"parameter '{owner_name}.{owner_attr}' is on the meta device and no "
        f"offload weights_map holds its data — was the model loaded with "
        f"device_map offloading? Use a resident layer or load without offload.")


def pick_resident_layer(model) -> str:
    """Return a decoder layer whose weights live fully in GPU memory.

    Weight-math experiments (exactness, in-place reordering) must run on real
    resident parameters, not on offloaded (meta-between-forwards) ones.
    """
    dev_map = getattr(model, "hf_device_map", {})
    import re
    idxs = sorted(int(m.group(1)) for k, v in dev_map.items()
                  if (m := re.fullmatch(r"model\.layers\.(\d+)", k))
                  and isinstance(v, int))
    if idxs:
        return f"model.layers.{idxs[len(idxs) // 2]}"
    # fallback: first layer whose params are actually on a real device
    n = model.config.num_hidden_layers
    for i in range(n):
        key = f"model.layers.{i}"
        params = list(get_mlp(model, key).parameters())
        if params and all(p.device.type != "meta" for p in params):
            return key
    raise RuntimeError("no resident layer found — model fully offloaded?")


def describe_model(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    info = {
        "params": total,
        "layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "intermediate_size": model.config.intermediate_size,
    }
    return info
