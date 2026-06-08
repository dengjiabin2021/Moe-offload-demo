"""探测代理 / 包装器模块。

提供零侵入的代理类，包装 TensorLoader 和 CacheManager，
在不修改原始代码的前提下拦截关键操作并记录指标。

设计原则：
- 只依赖接口契约（load_tensors_by_names / get_expert 等），不依赖具体模型。
- 所有耗时用 time.perf_counter() + torch.cuda.synchronize() 保证 GPU 操作计时准确。
- 通过 phase_ref（可变列表引用）感知当前阶段，无需直接依赖 Profiler。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


# ==========================================================================
# Phase 字符串常量（避免循环导入，tracker 和 profiler 共享）
# ==========================================================================

PHASE_IDLE = "idle"
PHASE_LOADING = "loading"
PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"


# ==========================================================================
# IO 追踪器
# ==========================================================================

@dataclass
class IOTracker:
    """IO 操作追踪器：记录各阶段的磁盘 IO 和 CPU↔GPU 传输。

    所有字节数以 Bytes 为单位，时间以秒为单位。
    通过 phase_ref 感知当前处于哪个阶段。
    """

    # —— 加载阶段 ——
    loading_disk_bytes: int = 0
    loading_disk_time_s: float = 0.0
    loading_cpu_gpu_bytes: int = 0
    loading_cpu_gpu_time_s: float = 0.0

    # —— 预填充阶段 ——
    prefill_disk_bytes: int = 0
    prefill_disk_time_s: float = 0.0
    prefill_cpu_gpu_bytes: int = 0
    prefill_cpu_gpu_time_s: float = 0.0

    # —— 解码阶段 ——
    decode_disk_bytes: int = 0
    decode_disk_time_s: float = 0.0
    decode_cpu_gpu_bytes: int = 0
    decode_cpu_gpu_time_s: float = 0.0

    # —— 每次 forward 内部的数据搬运累计耗时（由 Profiler 在 pre_forward_hook 中重置）——
    per_forward_data_time_s: float = 0.0

    # 指向 Profiler 当前阶段的引用（由 Profiler.wrap() 注入）
    _phase_ref: list[str] = field(default_factory=lambda: [PHASE_IDLE])

    # ---- 公开记录接口 ----

    def record_disk_io(self, bytes_read: int, time_s: float) -> None:
        """记录一次磁盘 IO 操作。"""
        phase = self._phase_ref[0]
        if phase == PHASE_LOADING:
            self.loading_disk_bytes += bytes_read
            self.loading_disk_time_s += time_s
        elif phase == PHASE_PREFILL:
            self.prefill_disk_bytes += bytes_read
            self.prefill_disk_time_s += time_s
        elif phase == PHASE_DECODE:
            self.decode_disk_bytes += bytes_read
            self.decode_disk_time_s += time_s

    def record_cpu_gpu_transfer(self, bytes_transferred: int, time_s: float) -> None:
        """记录一次 CPU→GPU 数据传输（含耗时）。"""
        phase = self._phase_ref[0]
        if phase == PHASE_LOADING:
            self.loading_cpu_gpu_bytes += bytes_transferred
            self.loading_cpu_gpu_time_s += time_s
        elif phase == PHASE_PREFILL:
            self.prefill_cpu_gpu_bytes += bytes_transferred
            self.prefill_cpu_gpu_time_s += time_s
        elif phase == PHASE_DECODE:
            self.decode_cpu_gpu_bytes += bytes_transferred
            self.decode_cpu_gpu_time_s += time_s

        # 同时累计到 per-forward 计数器
        self.per_forward_data_time_s += time_s

    def reset_per_forward(self) -> None:
        """重置 per-forward 数据搬运耗时（在每次 forward 开始前调用）。"""
        self.per_forward_data_time_s = 0.0


# ==========================================================================
# 缓存追踪器
# ==========================================================================

@dataclass
class CachePhaseStats:
    """单个阶段的缓存统计。"""
    gpu_hits: int = 0
    cpu_hits: int = 0
    disk_fetches: int = 0

    @property
    def total_requests(self) -> int:
        return self.gpu_hits + self.cpu_hits + self.disk_fetches

    @property
    def gpu_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.gpu_hits / self.total_requests

    @property
    def cpu_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cpu_hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.disk_fetches / self.total_requests


@dataclass
class CacheTracker:
    """缓存命中 / 未命中追踪器，按阶段拆分，同时记录逐层统计。

    通过 phase_ref 感知当前阶段。
    """

    prefill: CachePhaseStats = field(default_factory=CachePhaseStats)
    decode: CachePhaseStats = field(default_factory=CachePhaseStats)
    _phase_ref: list[str] = field(default_factory=lambda: [PHASE_IDLE])

    # 逐层统计（layer_idx → CachePhaseStats，跨阶段汇总）
    _layer_stats: dict[int, CachePhaseStats] = field(default_factory=dict)

    # ---- 公开记录接口 ----

    def record_gpu_hit(self, _time_s: float = 0.0, *, layer_idx: int | None = None) -> None:
        """记录一次 GPU 缓存命中。"""
        self._get_phase_stats().gpu_hits += 1
        if layer_idx is not None:
            self._get_layer_stats(layer_idx).gpu_hits += 1

    def record_cpu_hit(self, _time_s: float = 0.0, *, layer_idx: int | None = None) -> None:
        """记录一次 CPU 缓存命中。"""
        self._get_phase_stats().cpu_hits += 1
        if layer_idx is not None:
            self._get_layer_stats(layer_idx).cpu_hits += 1

    def record_disk_fetch(self, _time_s: float = 0.0, *, layer_idx: int | None = None) -> None:
        """记录一次磁盘回源（未命中）。"""
        self._get_phase_stats().disk_fetches += 1
        if layer_idx is not None:
            self._get_layer_stats(layer_idx).disk_fetches += 1

    # ---- 汇总 ----

    @property
    def gpu_hits(self) -> int:
        return self.prefill.gpu_hits + self.decode.gpu_hits

    @property
    def cpu_hits(self) -> int:
        return self.prefill.cpu_hits + self.decode.cpu_hits

    @property
    def disk_fetches(self) -> int:
        return self.prefill.disk_fetches + self.decode.disk_fetches

    @property
    def total_requests(self) -> int:
        return self.gpu_hits + self.cpu_hits + self.disk_fetches

    @property
    def gpu_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.gpu_hits / self.total_requests

    @property
    def cpu_hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.cpu_hits / self.total_requests

    # ---- 逐层统计 ----

    @property
    def layer_stats(self) -> dict[int, CachePhaseStats]:
        """返回按层聚合的缓存统计（按 layer_idx 排序）。"""
        return dict(sorted(self._layer_stats.items()))

    # ---- 内部 ----

    def _get_phase_stats(self) -> CachePhaseStats:
        phase = self._phase_ref[0]
        if phase == PHASE_PREFILL:
            return self.prefill
        elif phase == PHASE_DECODE:
            return self.decode
        # loading 阶段通常不会有缓存请求，但兜底返回 prefill
        return self.prefill

    def _get_layer_stats(self, layer_idx: int) -> CachePhaseStats:
        """获取或创建某层的缓存统计。"""
        if layer_idx not in self._layer_stats:
            self._layer_stats[layer_idx] = CachePhaseStats()
        return self._layer_stats[layer_idx]


# ==========================================================================
# ProfiledTensorLoader —— 包装 ManifestTensorLoader
# ==========================================================================

class ProfiledTensorLoader:
    """ManifestTensorLoader 的探测代理。

    拦截 load_tensors_by_names / load_tensors_to_buffer 两个核心 IO 方法，
    记录每次磁盘读取的字节数和耗时。
    """

    def __init__(self, real_loader: Any, io_tracker: IOTracker) -> None:
        self._real = real_loader
        self._io_tracker = io_tracker

        # 缓存 manifest 中每个 tensor 的 nbytes，用于推断 IO 量
        self._tensor_nbytes_cache: dict[str, int] = {}

    # ---- 代理需要暴露给外部的方法 ----

    def load_tensors_by_names(self, *, tensor_names: list[str], device: str) -> dict[str, torch.Tensor]:
        """代理：记录 IO 后委托给真实 loader。"""
        # 估算即将读取的总字节数
        total_bytes = sum(self._get_tensor_nbytes(name) for name in tensor_names)

        t0 = time.perf_counter()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        result = self._real.load_tensors_by_names(tensor_names=tensor_names, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        self._io_tracker.record_disk_io(bytes_read=total_bytes, time_s=elapsed)
        return result

    def load_tensors_to_buffer(
        self, *, tensor_names: list[str], buffer: torch.Tensor, slot_idx: int
    ) -> None:
        """代理：记录 IO 后委托给真实 loader。"""
        total_bytes = sum(self._get_tensor_nbytes(name) for name in tensor_names)

        t0 = time.perf_counter()
        if buffer.device.type == "cuda":
            torch.cuda.synchronize()
        self._real.load_tensors_to_buffer(tensor_names=tensor_names, buffer=buffer, slot_idx=slot_idx)
        if buffer.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        self._io_tracker.record_disk_io(bytes_read=total_bytes, time_s=elapsed)

    def load_tensor_by_name(self, *, tensor_name: str, device: str) -> torch.Tensor:
        """代理：记录单 tensor IO。"""
        total_bytes = self._get_tensor_nbytes(tensor_name)

        t0 = time.perf_counter()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        result = self._real.load_tensor_by_name(tensor_name=tensor_name, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        self._io_tracker.record_disk_io(bytes_read=total_bytes, time_s=elapsed)
        return result

    def get_tensor_info(self, *, tensor_name: str) -> tuple:
        """透传：不涉及 IO，直接委托。"""
        return self._real.get_tensor_info(tensor_name=tensor_name)

    def has_tensor(self, *, tensor_name: str) -> bool:
        """透传：不涉及 IO，直接委托。"""
        return self._real.has_tensor(tensor_name=tensor_name)

    # ---- 内部辅助 ----

    def _get_tensor_nbytes(self, tensor_name: str) -> int:
        """获取某个 tensor 的字节数（从 manifest 缓存）。"""
        if tensor_name not in self._tensor_nbytes_cache:
            # 从真实 loader 的 manifest 中读取 nbytes
            entry = self._real._read_manifest_entry(tensor_name=tensor_name)
            self._tensor_nbytes_cache[tensor_name] = entry["nbytes"]
        return self._tensor_nbytes_cache[tensor_name]

    # ---- 属性透传 ----

    def __getattr__(self, name: str) -> Any:
        """将未拦截的属性访问全部透传给真实 loader。"""
        return getattr(self._real, name)


# ==========================================================================
# ProfiledCacheManager —— 包装 ExpertCacheManager
# ==========================================================================

class ProfiledCacheManager:
    """ExpertCacheManager 的探测代理。

    拦截 get_expert() 调用，通过预查确定是 GPU 命中 / CPU 命中 / 磁盘回源，
    并记录耗时、缓存统计和 CPU→GPU 数据传输量。
    """

    def __init__(
        self,
        real_cache: Any,
        cache_tracker: CacheTracker,
        io_tracker: IOTracker,
    ) -> None:
        self._real = real_cache
        self._cache_tracker = cache_tracker
        self._io_tracker = io_tracker

        # 预计算单个专家的字节数（用于统计 CPU→GPU 传输量）
        self._expert_bytes: int = self._compute_expert_bytes()

    # ---- 核心拦截：get_expert ----

    def get_expert(self, layer_idx: int, expert_idx: int) -> dict[str, torch.Tensor]:
        """获取专家权重，同时记录缓存命中 / 未命中统计。

        策略：先调用 _find_expert_in_slots 预查专家位置，
        再委托给真实 get_expert，根据预查结果记录相应指标。
        """
        has_cuda = torch.cuda.is_available()

        # —— 预查：判断专家当前在 GPU / CPU / 磁盘 ——
        gpu_slot = self._real._find_expert_in_slots(
            self._real._gpu_slots,
            self._real._gpu_layer_slot_map,
            layer_idx,
            expert_idx,
        )
        cpu_slot = None
        if gpu_slot is None:
            cpu_slot = self._real._find_expert_in_slots(
                self._real._cpu_slots,
                self._real._cpu_layer_slot_map,
                layer_idx,
                expert_idx,
            )

        # —— 计时执行 ——
        if has_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = self._real.get_expert(layer_idx, expert_idx)
        if has_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # —— 根据预查结果记录指标 ——
        if gpu_slot is not None:
            self._cache_tracker.record_gpu_hit(elapsed, layer_idx=layer_idx)
        elif cpu_slot is not None:
            self._cache_tracker.record_cpu_hit(elapsed, layer_idx=layer_idx)
            # CPU 命中意味着一次 CPU→GPU 数据传输
            self._io_tracker.record_cpu_gpu_transfer(
                bytes_transferred=self._expert_bytes,
                time_s=elapsed,
            )
        else:
            self._cache_tracker.record_disk_fetch(elapsed, layer_idx=layer_idx)
            # 磁盘回源：一次磁盘 IO + 一次 CPU→GPU 传输
            # 注意：磁盘 IO 的精确字节数和耗时由 ProfiledTensorLoader 独立记录，
            # 这里只记录 CPU→GPU 部分。
            self._io_tracker.record_cpu_gpu_transfer(
                bytes_transferred=self._expert_bytes,
                time_s=elapsed,  # 包含磁盘 IO + copy 的总时间，近似
            )

        return result

    # ---- 属性透传 ----

    def __getattr__(self, name: str) -> Any:
        """将未拦截的属性访问全部透传给真实 cache_manager。

        包括 step()、_gpu_buffer、_cpu_buffer 等所有内部访问。
        """
        return getattr(self._real, name)

    # ---- 内部辅助 ----

    def _compute_expert_bytes(self) -> int:
        """根据真实 cache_manager 的内部信息计算单个专家的字节数。"""
        expert_elems = getattr(self._real, "_expert_elems", 0)
        expert_dtype = getattr(self._real, "_expert_dtype", torch.float16)
        element_size = torch.empty((), dtype=expert_dtype).element_size()
        return expert_elems * element_size
