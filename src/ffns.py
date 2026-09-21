"""FFN abstraction: locate, enumerate and describe gated FFN blocks (§72 ffns.py).

`FFNInfo` is the unit of work for every downstream component (router, guard,
cache, scheduler): it identifies one transformer layer's MLP and its
micro-expert partitioning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

from .micro_experts import contiguous_partition


@dataclass
class FFNInfo:
    """One transformer layer's FFN + its micro-expert partitioning."""
    layer_key: str                  # e.g. "model.layers.5"
    mlp_key: str                    # e.g. "model.layers.5.mlp"
    hidden_size: int
    intermediate_size: int
    group_size: int
    groups: List[List[int]] = field(default_factory=list)

    @property
    def n_experts(self) -> int:
        return len(self.groups)

    @property
    def experts_params(self) -> int:
        """Parameters in the three projections of this FFN."""
        h, i = self.hidden_size, self.intermediate_size
        return 3 * h * i  # gate, up, down (bias-free for Qwen-style MLPs)

    def expert_params(self, n_active: int) -> int:
        """Parameters touched when `n_active` experts run."""
        h, i = self.hidden_size, self.intermediate_size
        per_expert = 2 * (len(self.groups[0]) if self.groups else 0) * h  # gate+up rows
        down_cols = len(self.groups[0]) if self.groups else 0
        per_expert += down_cols * h
        return n_active * per_expert


def discover_ffns(model, group_size: int = 128) -> Dict[str, FFNInfo]:
    """Enumerate all gated FFNs of a Qwen-family model into FFNInfo records."""
    infos: Dict[str, FFNInfo] = {}
    for name, module in model.named_modules():
        if name.split(".")[-1] == "mlp" and hasattr(module, "down_proj"):
            layer_key = ".".join(name.split(".")[:-1])
            inter = module.down_proj.weight.shape[1]
            hidden = module.down_proj.weight.shape[0]
            infos[layer_key] = FFNInfo(
                layer_key=layer_key,
                mlp_key=name,
                hidden_size=hidden,
                intermediate_size=inter,
                group_size=group_size,
                groups=contiguous_partition(inter, group_size),
            )
    return dict(sorted(infos.items(), key=lambda kv: int(kv[0].rsplit(".", 1)[-1])))


def ffn_module(model, info: FFNInfo):
    return model.get_submodule(info.mlp_key)


def total_ffn_params(ffns: Dict[str, FFNInfo]) -> int:
    return sum(f.experts_params for f in ffns.values())
