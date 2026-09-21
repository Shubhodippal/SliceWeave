"""Forward hooks (§28): capture FFN intermediates, mask micro-experts, grab
router inputs (pre-MLP hidden states) and quality-guard inputs."""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


class FFNIntermediateCapture:
    """Capture the post-activation intermediate z for selected FFN layers.

    For a gated FFN: z = SiLU(gate(x)) * up(x). We hook `down_proj` and keep
    its input, which is exactly z.
    """

    def __init__(self):
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.captures: Dict[str, torch.Tensor] = {}

    def attach(self, model, layer_key: str, module=None):
        obj = model.get_submodule(f"{layer_key}.mlp.down_proj")
        proj_name = f"{layer_key}.down_proj"  # capture key convention (see repo notes)

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
    """Mask micro-expert groups inside down_proj's input (oracle mechanism, §12).

    Registers a pre-forward hook on down_proj that zeroes non-selected neurons
    of z. The full gate/up computation still runs first — so we know the true
    activations — this tool measures output-quality degradation, it saves no
    compute by design.
    """

    def __init__(self, model):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        # layer_key -> boolean keep mask over intermediate dim (True = keep)
        self.masks: Dict[str, torch.Tensor] = {}
        # layer_key -> sequence positions where the mask applies (None = all)
        self.positions: Dict[str, Optional[List[int]]] = {}
        # layer_key -> ([B, inter] masks, positions) for batched oracle runs
        self.batch_masks: Dict[str, tuple] = {}

    def set_mask(self, layer_key: str, keep_mask: torch.Tensor,
                 positions: Optional[List[int]] = None):
        """positions=None masks every token; a list masks only those slots.

        Per-token oracle measurement (§9/§20) sets positions=[token_index] so
        the ranking derived from one token is applied to exactly that token.
        """
        self.masks[layer_key] = keep_mask
        self.positions[layer_key] = positions
        self.batch_masks.pop(layer_key, None)

    def set_batch_masks(self, layer_key: str, keep_masks: torch.Tensor,
                        positions: List[int]):
        """Batched per-row masking for the batched oracle (6x+ speedup).

        keep_masks: [B, inter] bool — row i's mask applies to row i's z at
        sequence position positions[i] only; other rows/positions untouched.
        """
        self.batch_masks[layer_key] = (keep_masks, list(positions))
        self.masks.pop(layer_key, None)

    def attach(self, layer_key: str, ffn_ref):
        obj = ffn_ref  # the module holding gate/up/down, e.g. model.layers.5.mlp

        def pre_hook(mod, args, kwargs):
            z = args[0] if args else kwargs.get("hidden_states")
            if z is None:
                return args, kwargs
            batch = self.batch_masks.get(layer_key)
            if batch is not None:
                keeps, positions = batch
                zm = z.clone()
                for i, t in enumerate(positions):
                    zm[i, t] = zm[i, t] * keeps[i].to(z.dtype).to(z.device)
                z = zm
            elif layer_key in self.masks:
                mask = self.masks[layer_key]
                pos = self.positions.get(layer_key)
                if pos is None:
                    z = z * mask.to(z.dtype).to(z.device)
                else:
                    idx = torch.tensor(pos, dtype=torch.long, device=z.device)
                    zm = z.clone()
                    zm[:, idx] = zm[:, idx] * mask.to(z.dtype).to(z.device)
                    z = zm
            if args:
                return (z,) + args[1:], kwargs
            return args, {**kwargs, "hidden_states": z}

        self.handles.append(
            obj.down_proj.register_forward_pre_hook(pre_hook, with_kwargs=True))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


class FFNRecomputeMasked:
    """True sparse execution path (§30 heterogeneous execution, Phase 6 kernels).

    Replaces down_proj's input with (z * mask). Mathematically identical to
    FFNMaskedForward; kept separate so later real sparse kernels (grouped GEMM
    over contiguous expert blocks, §79 expert batching) can drop in behind the
    same interface.
    """

    def __init__(self, model):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.masks: Dict[str, torch.Tensor] = {}

    def set_mask(self, layer_key: str, keep_mask: torch.Tensor):
        self.masks[layer_key] = keep_mask

    def attach(self, layer_key: str, ffn_ref):
        obj = ffn_ref

        def pre_hook(mod, args, kwargs):
            z = args[0] if args else kwargs.get("hidden_states")
            mask = self.masks.get(layer_key)
            if mask is not None and z is not None:
                z = z * mask.to(z.dtype).to(z.device)
                if args:
                    return (z,) + args[1:], kwargs
                return args, {**kwargs, "hidden_states": z}
            return args, kwargs

        self.handles.append(
            obj.down_proj.register_forward_pre_hook(pre_hook, with_kwargs=True))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


class HiddenStateCapture:
    """Capture pre-MLP hidden states (router input x_l,t, §17/§19) and, for the
    quality guard (§21), optionally the same layer's output hidden state."""

    def __init__(self, positions: Optional[List[int]] = None):
        """
        positions: sequence positions to keep ([batch, pos, hidden]); None = all.
        """
        self.positions = positions
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.captures: Dict[str, torch.Tensor] = {}

    def attach(self, model, layer_key: str):
        """Hooks the decoder layer input — the hidden state entering layer `l`."""
        obj = model.get_submodule(layer_key)

        def pre_hook(mod, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            if h is None:
                return
            h = h.detach()
            if self.positions is not None:
                h = h[:, self.positions]
            self.captures[f"{layer_key}.input"] = h

        self.handles.append(obj.register_forward_pre_hook(pre_hook, with_kwargs=True))

    def clear(self):
        self.captures = {}

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
