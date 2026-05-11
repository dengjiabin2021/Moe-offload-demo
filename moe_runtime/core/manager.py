from __future__ import annotations

import threading
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from .adapter import MoEAdapter
from .cache import ExpertCache
from .config import OffloadConfig
from .expert import ExpertId, ExpertWeights
from .index import TensorIndex


class ExpertManager:
    def __init__(self, index: TensorIndex, adapter: MoEAdapter, config: OffloadConfig) -> None:
        self.index = index
        self.adapter = adapter
        self.config = config
        self.cpu_cache: ExpertCache[ExpertWeights] = ExpertCache(config.cpu_cache_bytes, name="cpu")
        self.gpu_cache: ExpertCache[ExpertWeights] = ExpertCache(config.gpu_cache_bytes, name="gpu")
        self.disk_loads = 0
        self.prefetches = 0
        self._lock = threading.RLock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moe-prefetch")

    def get(self, expert_id: ExpertId) -> ExpertWeights:
        with self._lock:
            gpu_weights = self.gpu_cache.get(expert_id)
            if gpu_weights is not None:
                return gpu_weights

            cpu_weights = self.cpu_cache.get(expert_id)
            if cpu_weights is None:
                cpu_weights = self.adapter.load_expert_weights(self.index, expert_id)
                cpu_weights = self._maybe_pin(cpu_weights)
                self.cpu_cache.put(expert_id, cpu_weights, cpu_weights.nbytes)
                self.disk_loads += 1

            gpu_weights = cpu_weights.to(
                self.config.device,
                dtype=self.config.dtype,
                non_blocking=self.config.pin_memory,
            )
            self.gpu_cache.put(expert_id, gpu_weights, gpu_weights.nbytes)
            return gpu_weights

    def prefetch(self, expert_ids: Iterable[ExpertId]) -> None:
        if not self.config.prefetch:
            return
        for expert_id in expert_ids:
            if self.gpu_cache.peek(expert_id) is None:
                self.prefetches += 1
                self.get(expert_id)

    def prefetch_async(self, expert_ids: Iterable[ExpertId]) -> Future | None:
        ids = tuple(expert_ids)
        if not self.config.prefetch or not ids:
            return None
        return self._prefetch_executor.submit(self.prefetch, ids)

    def stats(self) -> dict[str, float | int]:
        return {
            "disk_loads": self.disk_loads,
            "prefetches": self.prefetches,
            "cpu_cache_items": len(self.cpu_cache),
            "cpu_cache_bytes": self.cpu_cache.current_bytes,
            "cpu_cache_hit_rate": self.cpu_cache.stats.hit_rate,
            "gpu_cache_items": len(self.gpu_cache),
            "gpu_cache_bytes": self.gpu_cache.current_bytes,
            "gpu_cache_hit_rate": self.gpu_cache.stats.hit_rate,
            "gpu_cache_evictions": self.gpu_cache.stats.evictions,
            "cpu_cache_evictions": self.cpu_cache.stats.evictions,
        }

    def _maybe_pin(self, weights: ExpertWeights) -> ExpertWeights:
        if self.config.pin_memory and self.config.device.type == "cuda":
            return weights.pin_memory()
        return weights
