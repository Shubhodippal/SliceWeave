"""Phase 9 — memory-tier-aware scheduling + CPU/GPU cost model (§28-§31,
Milestone 9).

Separation of concerns (§28): the router decides *importance*; the scheduler
decides *where blocks live and on which device they execute*. The placement
score must never reorder mathematical importance unless the quality guard
explicitly allows approximation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class Tier(str, Enum):
    VRAM = "vram"   # L1 hot (§24)
    RAM = "ram"     # L2 warm
    SSD = "ssd"     # L3 cold


class Device(str, Enum):
    GPU = "gpu"
    CPU = "cpu"


@dataclass
class BlockMeta:
    """Runtime metadata per micro-expert block (§45, kept compact)."""
    layer_key: str
    expert_id: int
    size_bytes: int
    tier: Tier = Tier.SSD
    device: Device = Device.CPU
    activation_frequency: float = 0.0     # P(expert active) from calibration
    reuse_score: float = 0.0              # recent reuse statistic (§43)
    task_relevance: float = 0.0           # prompt-pool affinity (§28 gamma)
    universal_core: bool = False          # always-resident flag (§29)


@dataclass
class CostModel:
    """Hardware cost parameters (§31). Calibrate by measurement on the target
    machine — the defaults below are order-of-magnitude laptop values.

    bandwidth: RAM->VRAM ~8-16 GB/s (PCIe4 x8 laptop), SSD->RAM ~3.5 GB/s NVMe,
    VRAM read ~360 GB/s (3060 laptop), DDR4 dual-channel ~40 GB/s.
    """
    vram_read_gbps: float = 300.0
    ram_read_gbps: float = 30.0
    ssd_read_gbps: float = 3.0
    ram_to_vram_gbps: float = 10.0
    ssd_to_ram_gbps: float = 3.0
    gpu_tflops: float = 10.0              # bf16 fp32-acc realistic on 3060 laptop
    cpu_tflops: float = 0.6               # 5800H 8C/16T avx2 realistic bf16/fp32
    fixed_launch_overhead_ms: float = 0.05

    def transfer_ms(self, size_bytes: int, gbps: float) -> float:
        return size_bytes / (gbps * 1e9) * 1e3

    def compute_ms(self, flops: float, tflops: float) -> float:
        return flops / (tflops * 1e12) * 1e3 + self.fixed_launch_overhead_ms


@dataclass
class PlacementDecision:
    layer_key: str
    expert_id: int
    device: Device
    transfers: List[Tuple[Tier, Tier]] = field(default_factory=list)
    est_latency_ms: float = 0.0
    reason: str = ""


def flops_per_expert(n_neurons: int, hidden: int) -> int:
    """FLOPs of one micro-expert: gate + up rows and down columns (2 flops/MAC,
    three projections of shape [n, hidden])."""
    return 6 * n_neurons * hidden


class Scheduler:
    """Chooses execution device + async prefetch plan per selected block (§30)."""

    def __init__(self, cost: Optional[CostModel] = None,
                 vram_budget_bytes: int = 2 * 1024**3):
        self.cost = cost or CostModel()
        self.vram_budget = vram_budget_bytes
        self.vram_used = 0
        self.meta: Dict[Tuple[str, int], BlockMeta] = {}

    def register(self, meta: BlockMeta) -> None:
        self.meta[(meta.layer_key, meta.expert_id)] = meta

    def placement_score(self, m: BlockMeta) -> float:
        """§28 memory-aware score — used for RESIDENCY/prefetch priority only.

        placement_score = α·future_use + β·recent_freq + γ·task_relevance
                          + δ·transfer_saved − ε·memory_cost
        """
        alpha, beta, gamma, delta, epsilon = 1.0, 0.5, 0.3, 0.2, 0.5
        transfer_saved = {"vram": 1.0, "ram": 0.5, "ssd": 0.0}[m.tier.value]
        mem_cost = m.size_bytes / (1024 ** 2)  # per-MiB penalty
        return (alpha * m.activation_frequency + beta * m.reuse_score +
                gamma * m.task_relevance + delta * transfer_saved
                - epsilon * mem_cost / 100.0)

    def decide(self, layer_key: str, expert_id: int, importance_rank: int,
               n_neurons: int, hidden: int) -> PlacementDecision:
        """§30/§31: pick GPU or CPU by comparing T_gpu vs T_cpu for one block.

        T_gpu = transfer(RAM->VRAM, if not resident) + compute_GPU
        T_cpu = compute_CPU
        Residency follows VRAM budget filled by placement_score order (§29).
        """
        key = (layer_key, expert_id)
        m = self.meta.get(key) or BlockMeta(layer_key=layer_key, expert_id=expert_id,
                                            size_bytes=2 * n_neurons * hidden * 2)
        flops = flops_per_expert(n_neurons, hidden)
        t_compute_gpu = self.cost.compute_ms(flops, self.cost.gpu_tflops)
        t_compute_cpu = self.cost.compute_ms(flops, self.cost.cpu_tflops)

        resident = m.tier == Tier.VRAM
        if resident:
            t_gpu = t_compute_gpu
            transfers = []
        elif m.tier == Tier.RAM:
            t_gpu = self.cost.transfer_ms(m.size_bytes, self.cost.ram_to_vram_gbps) \
                + t_compute_gpu
            transfers = [(Tier.RAM, Tier.VRAM)]
        else:  # SSD
            t_gpu = self.cost.transfer_ms(m.size_bytes, self.cost.ssd_to_ram_gbps) \
                + self.cost.transfer_ms(m.size_bytes, self.cost.ram_to_vram_gbps) \
                + t_compute_gpu
            transfers = [(Tier.SSD, Tier.RAM), (Tier.RAM, Tier.VRAM)]

        # §31: if T_cpu < T_gpu execute on CPU (block never needs to reach VRAM)
        if t_cpu_ok := (t_compute_cpu < t_gpu):
            dev = Device.CPU
            lat = t_compute_cpu
            reason = "cpu cheaper than transfer+gpu compute"
        else:
            dev = Device.GPU
            lat = t_gpu
            reason = "resident/cheaper on gpu"

        return PlacementDecision(layer_key=layer_key, expert_id=expert_id,
                                 device=dev, transfers=transfers,
                                 est_latency_ms=lat, reason=reason)

    def fill_vram_cache(self, candidates: Sequence[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """§29: fill VRAM with the highest placement_score blocks under budget."""
        scored = []
        for key in candidates:
            m = self.meta.get(key)
            if m is not None and not m.universal_core:
                scored.append((self.placement_score(m), key))
            elif m is not None:
                scored.append((float("inf"), key))  # core always resident
        scored.sort(key=lambda t: -t[0])
        pinned: List[Tuple[str, int]] = []
        for _, key in scored:
            m = self.meta.get(key)
            if m is None:
                continue
            if m.universal_core:
                pinned.append(key)
                self.vram_used += m.size_bytes
            elif self.vram_used + m.size_bytes <= self.vram_budget:
                m.tier = Tier.VRAM
                self.vram_used += m.size_bytes
                pinned.append(key)
        return pinned

    def token_plan(self, layer_key: str, ranked_experts: Sequence[int],
                   n_neurons: int, hidden: int, max_blocks: int) -> List[PlacementDecision]:
        """Assemble one layer's execution plan: top blocks by router importance,
        each scheduled independently (§46 steps 4-6)."""
        decisions = []
        for rank, e in enumerate(ranked_experts[:max_blocks]):
            decisions.append(self.decide(layer_key, e, rank, n_neurons, hidden))
        return decisions

    def token_latency_ms(self, plan: Sequence[PlacementDecision]) -> float:
        """§32/§35: worst-case (critical path) latency of a plan. Transfers of
        non-resident blocks serialize with compute in this simple model; async
        overlap is Milestone 8's prefetch job."""
        if not plan:
            return 0.0
        return max(d.est_latency_ms for d in plan) + \
            sum(d.est_latency_ms for d in plan if d.transfers) * 0.5  # partial overlap


def simulate_step_latency(dense_bytes: int, sparse_plan: Sequence[PlacementDecision]) -> Dict[str, float]:
    """§77: theoretical dense vs sparse latency for one layer-step.

    dense_bytes: bytes the full dense FFN would read (weights only).
    """
    cost = CostModel()
    t_dense_ms = dense_bytes / (cost.vram_read_gbps * 1e9) * 1e3
    t_sparse_ms = max((d.est_latency_ms for d in sparse_plan), default=0.0)
    return {
        "dense_ms": t_dense_ms,
        "sparse_ms": t_sparse_ms,
        "speedup_theoretical": t_dense_ms / max(t_sparse_ms, 1e-9),
        "bytes_dense": dense_bytes,
        "bytes_sparse": sum(d.est_latency_ms and 0 for d in sparse_plan) or
        sum(2 * 1 for d in sparse_plan),  # refined by callers with real sizes
    }
