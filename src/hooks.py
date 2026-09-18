"""Forward hooks for capturing FFN intermediates (Blueprint §28)."""
from __future__ import annotations

from typing import Dict, List

import torch


class FFNIntermediateCapture:
    """Capture the post-activation intermediate vector z for selected FFN layers.

    For a gated FFN: z = SiLU(gate(x)) * up(x).
    We register a forward hook on `down_proj`; its input is exactly z.
    """

    def __init__(self):
        self.handles = []
        self.captures: Dict[str, torch.Tensor] = {}

    def attach(self, model, layer_key: str, module):
        proj_name = f"{layer_key}.down_proj"
        obj = module
        for part in proj_name.split("."):
            obj = getattr(obj, part)

        def hook(mod, inputs, output, key=proj_name):
            # inputs[0] shape: [batch, seq, intermediate]
            self.captures[key] = inputs[0].detach()

        self.handles.append(obj.register_forward_hook(hook))

    def clear(self):
        self.captures = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class FFNMaskedForward:
    """Context manager that masks micro-expert groups inside down_proj inputs.

    Works generically via a pre-forward hook on down_proj that zeroes the
    activations of the non-selected intermediate neurons before the matmul.

    This is the oracle mechanism: it runs the FULL gate/up computation first
    (so we know true activations), then masks before down_proj. Compute is NOT
    saved — this is only for measuring output-quality degradation (Blueprint §12).
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        # layer_key -> boolean keep mask over intermediate dim (True = keep)
        self.masks: Dict[str, torch.Tensor] = {}

    def set_mask(self, layer_key: str, keep_mask: torch.Tensor):
        self.masks[layer_key] = keep_mask

    def attach(self, layer_key: str, ffn_ref):
        obj = ffn_ref  # the module holding gate/up/down, e.g. model.layers.5.mlp

        def pre_hook(mod, args, kwargs):
            z = args[0]
            mask = self.masks.get(layer_key)
            if mask is not None:
                z = z * mask.to(z.dtype).to(z.device)
                return (z,) + args[1:], kwargs
            return args, kwargs

        self.handles.append(obj.down_proj.register_forward_pre_hook(pre_hook))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


class FFNRecomputeMasked:
    """True sparse execution for small models: replaces down_proj input.

    Instead of masking, this fully replaces the intermediate activation with
    (z * mask). Mathematically identical to FFNMaskedForward; kept separate for
    clarity when benchmarking later.
    """
