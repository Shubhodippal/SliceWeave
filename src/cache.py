"""Phase 7 — expert access traces + tiered cache simulation (§24-§25, §42-§43,
Milestone 7).

Simulates the memory hierarchy on recorded expert traces before any runtime
engineering (§77: model memory behaviour before kernel work):

    L1 VRAM (hot)  <- L2 RAM (warm)  <- L3 SSD (cold)
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class AccessRecord:
    """One expert demand (§96 logging columns)."""
    step: int            # global step ordering (token*layers + layer, or time)
    layer_key: str
    expert_id: int
    token_id: int
    bytes_size: int = 0  # block size in bytes (from metadata, §45)


@dataclass
class CacheStats:
    vram_hits: int = 0
    ram_hits: int = 0
    ssd_misses: int = 0
    vram_evictions: int = 0
    ram_evictions: int = 0
    bytes_vram_read: int = 0
    bytes_ram_read: int = 0
    bytes_ssd_read: int = 0
    bytes_prefetched_wasted: int = 0

    def hit_rates(self) -> Dict[str, float]:
        total = self.vram_hits + self.ram_hits + self.ssd_misses
        if total == 0:
            return {"vram_hit_rate": 0.0, "ram_hit_rate": 0.0, "ssd_miss_rate": 0.0}
        return {
            "vram_hit_rate": self.vram_hits / total,
            "ram_hit_rate": self.ram_hits / total,
            "ssd_miss_rate": self.ssd_misses / total,
        }

    def traffic(self) -> Dict[str, int]:
        """MTR (§39) input: bytes moved per tier."""
        return {"vram": self.bytes_vram_read, "ram": self.bytes_ram_read,
                "ssd": self.bytes_ssd_read}


class TieredCache:
    """Two-level LRU/LFU simulation over an expert access trace (§24).

    vram_capacity / ram_capacity in bytes (0 = unlimited).
    policy: "lru" (recency) or "lfu" (frequency, §52 baselines).
    """

    def __init__(self, vram_capacity: int, ram_capacity: int, policy: str = "lru",
                 block_bytes: int = 1):
        self.vram_cap = vram_capacity
        self.ram_cap = ram_capacity
        self.policy = policy
        self.block_bytes = block_bytes
        self.vram: "OrderedDict[Tuple[str, int], int]" = OrderedDict()  # key -> freq
        self.ram: "OrderedDict[Tuple[str, int], int]" = OrderedDict()
        self.vram_bytes = 0
        self.ram_bytes = 0
        self.stats = CacheStats()
        self.prefetched: set = set()

    def _bump(self, cache: "OrderedDict", key) -> None:
        if self.policy == "lfu":
            cache[key] = cache.get(key, 0) + 1
        else:  # lru
            cache[key] = cache.pop(key) if key in cache else cache.get(key, 0)

    def _evict(self, cache: "OrderedDict", cap: int, cur_bytes: int, which: str) -> int:
        evicted = 0
        while cur_bytes > cap and cache:
            k, _ = cache.popitem(last=False)
            sz = k[2] if len(k) > 2 else self.block_bytes
            cur_bytes -= sz
            evicted += 1
            if which == "vram":
                self.stats.vram_evictions += 1
            else:
                self.stats.ram_evictions += 1
        return cur_bytes

    def access(self, rec: AccessRecord) -> str:
        """Process one demand. Returns tier served from: vram|ram|ssd."""
        key = (rec.layer_key, rec.expert_id)
        size = rec.bytes_size or self.block_bytes

        if key in self.vram:
            self.stats.vram_hits += 1
            self.stats.bytes_vram_read += size
            self._bump(self.vram, key)
            self.vram.pop(key)
            self.vram[key] = self.vram.get(key, 0)
            return "vram"

        if key in self.ram:
            self.stats.ram_hits += 1
            self.stats.bytes_ram_read += size
            self._bump(self.ram, key)
            # promote into VRAM if it fits (hot blocks move up, §29)
            if self.vram_cap == 0 or self.vram_bytes + size <= self.vram_cap:
                self.vram[key] = 1
                self.vram_bytes += size
                self.ram.pop(key)
                self.ram_bytes -= size
            return "ram"

        # cold: from SSD, land in RAM (§25: SSD -> RAM, not SSD -> GPU per token)
        self.stats.ssd_misses += 1
        self.stats.bytes_ssd_read += size
        self.ram[key] = 1
        self.ram_bytes += size
        self.ram_bytes = self._evict(self.ram, self.ram_cap, self.ram_bytes, "ram")
        # opportunistic VRAM fill
        if self.vram_cap == 0 or self.vram_bytes + size <= self.vram_cap:
            self.vram[key] = 1
            self.vram_bytes += size
            self.ram.pop(key)
            self.ram_bytes -= size
        else:
            self.vram_bytes = self._evict(self.vram, self.vram_cap,
                                          self.vram_bytes, "vram")
        return "ssd"

    def prefetch(self, keys: Iterable[Tuple[str, int]],
                 sizes: Optional[Dict] = None,
                 needed_soon: Optional[set] = None) -> None:
        """§26/§42: move predicted blocks up one tier before demand.

        needed_soon: keys the predictor believes will be demanded — used only to
        account wasted prefetch bytes when the prediction misses.
        """
        for key in keys:
            if key in self.vram:
                continue
            size = (sizes or {}).get(key, self.block_bytes)
            if key in self.ram:
                # RAM -> VRAM prefetch
                if self.vram_cap == 0 or self.vram_bytes + size <= self.vram_cap:
                    self.ram.pop(key)
                    self.ram_bytes -= size
                    self.vram[key] = 1
                    self.vram_bytes += size
                else:
                    self.vram_bytes = self._evict(self.vram, self.vram_cap,
                                                  self.vram_bytes, "vram")
                    self.ram.pop(key)
                    self.ram_bytes -= size
                    self.vram[key] = 1
                    self.vram_bytes += size
            else:
                # SSD -> RAM prefetch
                self.ram[key] = 1
                self.ram_bytes += size
                self.ram_bytes = self._evict(self.ram, self.ram_cap,
                                             self.ram_bytes, "ram")
            if needed_soon is not None and key not in needed_soon:
                self.stats.bytes_prefetched_wasted += size


def reuse_distance_histogram(records: Sequence[AccessRecord]) -> Dict[int, int]:
    """§43 expert reuse distance: gap (in accesses) between repeats of a block."""
    from typing import Any as _Any
    from .metrics import expert_reuse_distance
    stream: List[_Any] = [(r.layer_key, r.expert_id) for r in records]
    gaps = expert_reuse_distance(stream)  # type: ignore[arg-type]
    hist: Dict[int, int] = {}
    for g in gaps:
        hist[g] = hist.get(g, 0) + 1
    return dict(sorted(hist.items()))


def trace_from_lists(layer_keys: Sequence[str], expert_ids: Sequence,
                     token_ids: Sequence[int],
                     sizes: Optional[Dict] = None) -> List[AccessRecord]:
    """Flatten per-step expert selections into an ordered access trace.

    layer_keys/expert_ids: per step (token), the selected experts of each layer.
    """
    records: List[AccessRecord] = []
    step = 0
    sizes = sizes or {}
    for t, (experts) in enumerate(expert_ids):
        for li, layer in enumerate(layer_keys):
            for e in experts[li]:
                key = (layer, e)
                records.append(AccessRecord(step=step, layer_key=layer,
                                            expert_id=int(e), token_id=token_ids[t],
                                            bytes_size=sizes.get(key, 0)))
                step += 1
    return records
