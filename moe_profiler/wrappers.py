"""异步 MoE Profiler 的统计数据结构。

本模块只保存统计器和常量，不接触运行时逻辑。真正的 monkeypatch 注入在
``moe_profiler.profiler`` 中完成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any


PHASE_IDLE = "idle"
PHASE_LOADING = "loading"
PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"


@dataclass
class TimeStats:
    """一类操作的调用次数和 wall-clock 耗时统计。"""

    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def record(self, elapsed_s: float) -> None:
        """记录一次耗时。"""

        self.count += 1
        self.total_s += elapsed_s
        self.max_s = max(self.max_s, elapsed_s)

    @property
    def avg_s(self) -> float:
        """返回平均耗时。"""

        if self.count == 0:
            return 0.0
        return self.total_s / self.count

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        return {
            "count": self.count,
            "total_s": self.total_s,
            "avg_s": self.avg_s,
            "max_s": self.max_s,
        }


@dataclass
class TransferStats(TimeStats):
    """带字节数的 IO / 搬运统计。"""

    bytes: int = 0

    def record_transfer(self, bytes_count: int, elapsed_s: float) -> None:
        """记录一次数据传输。"""

        self.bytes += int(bytes_count)
        self.record(elapsed_s)

    @property
    def mb(self) -> float:
        """返回总传输量，单位 MB。"""

        return self.bytes / 1024**2

    @property
    def throughput_mbps(self) -> float:
        """返回平均吞吐，单位 MB/s。"""

        if self.total_s <= 0:
            return 0.0
        return self.mb / self.total_s

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        data = super().to_dict()
        data.update(
            {
                "bytes": self.bytes,
                "mb": self.mb,
                "throughput_mbps": self.throughput_mbps,
            }
        )
        return data


@dataclass
class CachePhaseStats:
    """一个阶段内的 async expert 位置统计。"""

    gpu_hits: int = 0
    cpu_hits: int = 0
    disk_misses: int = 0

    @property
    def total(self) -> int:
        """返回本阶段 expert 请求总数。"""

        return self.gpu_hits + self.cpu_hits + self.disk_misses

    @property
    def gpu_hit_rate(self) -> float:
        """返回 GPU 命中率。"""

        return self.gpu_hits / self.total if self.total else 0.0

    @property
    def cpu_hit_rate(self) -> float:
        """返回 CPU 命中率。"""

        return self.cpu_hits / self.total if self.total else 0.0

    @property
    def disk_miss_rate(self) -> float:
        """返回磁盘 miss 率。"""

        return self.disk_misses / self.total if self.total else 0.0

    def record(self, kind: str) -> None:
        """记录一次 expert 位置分类。"""

        if kind == "gpu":
            self.gpu_hits += 1
        elif kind == "cpu":
            self.cpu_hits += 1
        elif kind == "disk":
            self.disk_misses += 1
        else:
            raise ValueError(f"Unsupported cache kind: {kind}")

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        return {
            "gpu_hits": self.gpu_hits,
            "cpu_hits": self.cpu_hits,
            "disk_misses": self.disk_misses,
            "total": self.total,
            "gpu_hit_rate": self.gpu_hit_rate,
            "cpu_hit_rate": self.cpu_hit_rate,
            "disk_miss_rate": self.disk_miss_rate,
        }


@dataclass
class WaitRecord:
    """一次 block 等待任意 expert ready 的阻塞记录。"""

    layer_idx: int
    pending_count: int
    elapsed_s: float
    ready_expert_idx: int
    phase: str
    dominant_reason: str
    enter_counts: dict[str, int] = field(default_factory=dict)
    exit_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        return {
            "layer_idx": self.layer_idx,
            "pending_count": self.pending_count,
            "elapsed_s": self.elapsed_s,
            "ready_expert_idx": self.ready_expert_idx,
            "phase": self.phase,
            "dominant_reason": self.dominant_reason,
            "enter_counts": dict(self.enter_counts),
            "exit_counts": dict(self.exit_counts),
        }


@dataclass
class ResourceSample:
    """后台调度器某一时刻的队列和 inflight 状态采样。"""

    timestamp_s: float
    phase: str
    waiting_cpu_len: int
    loading_cpu_count: int
    waiting_gpu_len: int
    loading_gpu_count: int
    ready_count: int
    compute_pending_count: int

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        return {
            "timestamp_s": self.timestamp_s,
            "phase": self.phase,
            "waiting_cpu_len": self.waiting_cpu_len,
            "loading_cpu_count": self.loading_cpu_count,
            "waiting_gpu_len": self.waiting_gpu_len,
            "loading_gpu_count": self.loading_gpu_count,
            "ready_count": self.ready_count,
            "compute_pending_count": self.compute_pending_count,
        }


@dataclass
class AsyncCacheTracker:
    """按 prefill/decode/layer 统计 expert 命中位置。"""

    phase_ref: list[str]
    prefill: CachePhaseStats = field(default_factory=CachePhaseStats)
    decode: CachePhaseStats = field(default_factory=CachePhaseStats)
    other: CachePhaseStats = field(default_factory=CachePhaseStats)
    layer_stats: dict[int, CachePhaseStats] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, kind: str, layer_idx: int | None) -> None:
        """记录一次 expert 位置分类。"""

        with self._lock:
            self._phase_stats().record(kind)
            if layer_idx is not None:
                self.layer_stats.setdefault(int(layer_idx), CachePhaseStats()).record(kind)

    @property
    def total(self) -> CachePhaseStats:
        """聚合所有阶段的命中统计。"""

        result = CachePhaseStats()
        for stats in (self.prefill, self.decode, self.other):
            result.gpu_hits += stats.gpu_hits
            result.cpu_hits += stats.cpu_hits
            result.disk_misses += stats.disk_misses
        return result

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        with self._lock:
            return {
                "total": self.total.to_dict(),
                "prefill": self.prefill.to_dict(),
                "decode": self.decode.to_dict(),
                "other": self.other.to_dict(),
                "by_layer": {
                    str(layer_idx): stats.to_dict()
                    for layer_idx, stats in sorted(self.layer_stats.items())
                },
            }

    def _phase_stats(self) -> CachePhaseStats:
        """返回当前阶段对应的统计桶。"""

        phase = self.phase_ref[0]
        if phase == PHASE_PREFILL:
            return self.prefill
        if phase == PHASE_DECODE:
            return self.decode
        return self.other


@dataclass
class AsyncStageTracker:
    """异步 CacheManager 内部状态机的等待/执行耗时统计。

    统计项含义：
    - waiting_cpu：disk miss 后，在 CPU slot 可用前排队的时间。
    - loading_cpu：磁盘线程读取 expert 到 CPU buffer 的时间。
    - waiting_gpu：CPU 已有 expert，等待 GPU slot 可写的时间。
    - loading_gpu：H2D copy 提交后，到 event 完成的时间。
    - compute_pending：block release 后，到计算 event 完成的保护时间。
    - wait_any_ready：block 因没有 ready expert 而阻塞等待的时间。
    """

    public_methods: dict[str, TimeStats] = field(default_factory=dict)
    scheduler_methods: dict[str, TimeStats] = field(default_factory=dict)
    stage_waits: dict[str, TimeStats] = field(default_factory=dict)
    wait_reasons: dict[str, TimeStats] = field(default_factory=dict)
    wait_reasons_by_phase: dict[str, dict[str, TimeStats]] = field(default_factory=dict)
    wait_by_layer: dict[int, TimeStats] = field(default_factory=dict)
    wait_reason_by_layer: dict[int, dict[str, TimeStats]] = field(default_factory=dict)
    wait_records: list[WaitRecord] = field(default_factory=list)
    resource_samples: list[ResourceSample] = field(default_factory=list)
    disk_io: TransferStats = field(default_factory=TransferStats)
    resident_io: TransferStats = field(default_factory=TransferStats)
    io_by_phase: dict[str, dict[str, TransferStats]] = field(default_factory=dict)
    disk_worker_capacity: int = 2
    _starts: dict[str, dict[tuple[int, int], float]] = field(default_factory=dict, repr=False)
    _start_phases: dict[str, dict[tuple[int, int], str]] = field(default_factory=dict, repr=False)
    _resource_base_s: float | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_disk_worker_capacity(self, capacity: int) -> None:
        """设置磁盘读取 worker 容量。

        这个值来自 async CacheManager 实际创建的线程池，只用于计算 profiler
        里的 busy ratio；不参与运行时调度。
        """

        if capacity <= 0:
            return
        with self._lock:
            self.disk_worker_capacity = int(capacity)

    def record_public_method(self, name: str, elapsed_s: float) -> None:
        """记录一次 CacheManager 公开接口耗时。"""

        with self._lock:
            self.public_methods.setdefault(name, TimeStats()).record(elapsed_s)

    def record_scheduler_method(self, name: str, elapsed_s: float, progressed: bool) -> None:
        """记录一次后台调度函数耗时。"""

        key = f"{name}.progressed" if progressed else f"{name}.idle"
        with self._lock:
            self.scheduler_methods.setdefault(key, TimeStats()).record(elapsed_s)
            self.scheduler_methods.setdefault(name, TimeStats()).record(elapsed_s)

    def start_stage(
        self,
        stage: str,
        key: tuple[int, int],
        timestamp_s: float,
        phase: str | None = None,
    ) -> None:
        """标记某个 expert 进入异步阶段。"""

        with self._lock:
            self._starts.setdefault(stage, {})[key] = timestamp_s
            if phase is not None:
                self._start_phases.setdefault(stage, {})[key] = phase

    def finish_stage(self, stage: str, key: tuple[int, int], timestamp_s: float) -> tuple[float, str | None] | None:
        """标记某个 expert 离开异步阶段，并累计等待时间。"""

        with self._lock:
            started_at = self._starts.setdefault(stage, {}).pop(key, None)
            if started_at is None:
                return None
            phase = self._start_phases.setdefault(stage, {}).pop(key, None)
            elapsed_s = max(0.0, timestamp_s - started_at)
            self.stage_waits.setdefault(stage, TimeStats()).record(elapsed_s)
            return elapsed_s, phase

    def record_disk_io(self, bytes_count: int, elapsed_s: float, phase: str | None = None) -> None:
        """记录 routed expert 的磁盘到 CPU buffer 读取。"""

        with self._lock:
            self.disk_io.record_transfer(bytes_count, elapsed_s)
            if phase is not None:
                self._phase_io(phase, "disk_to_cpu").record_transfer(bytes_count, elapsed_s)

    def record_resident_io(
        self,
        bytes_count: int,
        elapsed_s: float,
        phase: str | None = None,
        *,
        includes_h2d: bool = False,
    ) -> None:
        """记录模型常驻权重加载。"""

        with self._lock:
            self.resident_io.record_transfer(bytes_count, elapsed_s)
            if phase is not None:
                self._phase_io(phase, "disk_to_cpu").record_transfer(bytes_count, elapsed_s)
                if includes_h2d:
                    self._phase_io(phase, "cpu_to_gpu").record_transfer(bytes_count, elapsed_s)

    def record_cpu_gpu_io(self, bytes_count: int, elapsed_s: float, phase: str | None) -> None:
        """记录 CPU 到 GPU 的 H2D 搬运。"""

        if phase is None:
            return
        with self._lock:
            self._phase_io(phase, "cpu_to_gpu").record_transfer(bytes_count, elapsed_s)

    def record_wait_any_ready(
        self,
        *,
        layer_idx: int,
        pending_count: int,
        elapsed_s: float,
        ready_expert_idx: int,
        phase: str,
        dominant_reason: str,
        enter_counts: dict[str, int],
        exit_counts: dict[str, int],
    ) -> None:
        """记录 block 在 wait_any_ready 中真实阻塞的一次等待。

        这里统计的是 Python/block 控制流真正停住的 wall-clock 时间，不是某个
        expert 生命周期阶段的累加时间。因此它更适合判断生成延迟到底卡在 CPU
        队列、磁盘读取、H2D 还是 GPU 计算保护上。
        """

        reason = dominant_reason or "unknown"
        with self._lock:
            self.wait_reasons.setdefault(reason, TimeStats()).record(elapsed_s)
            self.wait_reasons_by_phase.setdefault(phase, {}).setdefault(reason, TimeStats()).record(elapsed_s)
            self.wait_by_layer.setdefault(layer_idx, TimeStats()).record(elapsed_s)
            self.wait_reason_by_layer.setdefault(layer_idx, {}).setdefault(reason, TimeStats()).record(elapsed_s)
            self.wait_records.append(
                WaitRecord(
                    layer_idx=layer_idx,
                    pending_count=pending_count,
                    elapsed_s=elapsed_s,
                    ready_expert_idx=ready_expert_idx,
                    phase=phase,
                    dominant_reason=reason,
                    enter_counts=dict(enter_counts),
                    exit_counts=dict(exit_counts),
                )
            )

    def record_resource_sample(
        self,
        *,
        timestamp_s: float,
        phase: str,
        waiting_cpu_len: int,
        loading_cpu_count: int,
        waiting_gpu_len: int,
        loading_gpu_count: int,
        ready_count: int,
        compute_pending_count: int,
    ) -> None:
        """记录后台调度器队列长度和 inflight 数量。

        采样点用于后续做时间加权平均，从而估算磁盘 worker、H2D copy 和计算
        保护队列是否被持续压满。
        """

        with self._lock:
            if self._resource_base_s is None:
                self._resource_base_s = timestamp_s
            relative_s = max(0.0, timestamp_s - self._resource_base_s)
            self.resource_samples.append(
                ResourceSample(
                    timestamp_s=relative_s,
                    phase=phase,
                    waiting_cpu_len=waiting_cpu_len,
                    loading_cpu_count=loading_cpu_count,
                    waiting_gpu_len=waiting_gpu_len,
                    loading_gpu_count=loading_gpu_count,
                    ready_count=ready_count,
                    compute_pending_count=compute_pending_count,
                )
            )

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        with self._lock:
            return {
                "public_methods": {
                    name: stats.to_dict()
                    for name, stats in sorted(self.public_methods.items())
                },
                "scheduler_methods": {
                    name: stats.to_dict()
                    for name, stats in sorted(self.scheduler_methods.items())
                },
                "stage_waits": {
                    name: stats.to_dict()
                    for name, stats in sorted(self.stage_waits.items())
                },
                "wait_reasons": {
                    name: stats.to_dict()
                    for name, stats in sorted(self.wait_reasons.items())
                },
                "wait_reasons_by_phase": {
                    phase: {
                        name: stats.to_dict()
                        for name, stats in sorted(reason_stats.items())
                    }
                    for phase, reason_stats in sorted(self.wait_reasons_by_phase.items())
                },
                "wait_by_layer": self._wait_by_layer_dict_locked(),
                "wait_records": [record.to_dict() for record in self.wait_records],
                "resource_utilization": self._resource_utilization_dict_locked(),
                "resource_utilization_by_phase": self._resource_utilization_by_phase_dict_locked(),
                "resource_samples": [sample.to_dict() for sample in self.resource_samples],
                "disk_io": self.disk_io.to_dict(),
                "resident_io": self.resident_io.to_dict(),
                "io_by_phase": {
                    phase: {
                        direction: stats.to_dict()
                        for direction, stats in sorted(direction_stats.items())
                    }
                    for phase, direction_stats in sorted(self.io_by_phase.items())
                },
            }

    def _phase_io(self, phase: str, direction: str) -> TransferStats:
        """获取某个阶段和方向的 IO 统计桶。"""

        phase_stats = self.io_by_phase.setdefault(phase, {})
        return phase_stats.setdefault(direction, TransferStats())

    def _wait_by_layer_dict_locked(self) -> dict[str, Any]:
        """汇总每层 wait_any_ready 的真实阻塞时间。"""

        result: dict[str, Any] = {}
        for layer_idx, stats in sorted(self.wait_by_layer.items()):
            reason_stats = self.wait_reason_by_layer.get(layer_idx, {})
            main_reason = ""
            main_reason_total_s = 0.0
            if reason_stats:
                main_reason, main_stats = max(
                    reason_stats.items(),
                    key=lambda item: item[1].total_s,
                )
                main_reason_total_s = main_stats.total_s

            data = stats.to_dict()
            data.update(
                {
                    "main_reason": main_reason,
                    "main_reason_total_s": main_reason_total_s,
                    "reasons": {
                        name: reason.to_dict()
                        for name, reason in sorted(reason_stats.items())
                    },
                }
            )
            result[str(layer_idx)] = data
        return result

    def _resource_utilization_dict_locked(self) -> dict[str, Any]:
        """根据调度器采样点估算后台资源利用率。

        该统计是时间加权估算：相邻两个采样点之间，认为前一个采样点的状态持续
        到后一个采样点。它不会影响运行时调度，只用于回答资源是否被压满。
        """

        return self._resource_utilization_for_samples_locked(self.resource_samples)

    def _resource_utilization_by_phase_dict_locked(self) -> dict[str, Any]:
        """按 phase 分别估算后台资源利用率。

        总览指标会把 loading、prefill、decode、idle 全部混到同一个时间窗口里。
        分阶段指标能看出 worker 是在 prefill 满载，还是 decode 供给不足。
        """

        phases = sorted({sample.phase for sample in self.resource_samples})
        return {
            phase: self._resource_utilization_for_samples_locked(
                [sample for sample in self.resource_samples if sample.phase == phase]
            )
            for phase in phases
        }

    def _resource_utilization_for_samples_locked(self, samples: list[ResourceSample]) -> dict[str, Any]:
        """对指定采样序列计算时间加权资源利用率。"""

        empty_result = {
            "observed_s": 0.0,
            "disk_worker_capacity": self.disk_worker_capacity,
            "disk_workers": self._resource_metric_dict(0.0, 0.0, 0.0, 0),
            "h2d": self._resource_metric_dict(0.0, 0.0, 0.0, 0),
            "compute_pending": self._resource_metric_dict(0.0, 0.0, 0.0, 0),
            "waiting_cpu_queue": self._queue_metric_dict(0.0, 0),
            "waiting_gpu_queue": self._queue_metric_dict(0.0, 0),
            "ready_queue": self._queue_metric_dict(0.0, 0),
        }
        if not samples:
            return empty_result

        observed_s = max(0.0, samples[-1].timestamp_s - samples[0].timestamp_s)
        max_loading_cpu = max(sample.loading_cpu_count for sample in samples)
        max_loading_gpu = max(sample.loading_gpu_count for sample in samples)
        max_compute = max(sample.compute_pending_count for sample in samples)
        max_waiting_cpu = max(sample.waiting_cpu_len for sample in samples)
        max_waiting_gpu = max(sample.waiting_gpu_len for sample in samples)
        max_ready = max(sample.ready_count for sample in samples)

        if observed_s <= 0 or len(samples) == 1:
            return {
                "observed_s": observed_s,
                "disk_worker_capacity": self.disk_worker_capacity,
                "disk_workers": self._resource_metric_dict(0.0, 0.0, 0.0, max_loading_cpu),
                "h2d": self._resource_metric_dict(0.0, 0.0, 0.0, max_loading_gpu),
                "compute_pending": self._resource_metric_dict(0.0, 0.0, 0.0, max_compute),
                "waiting_cpu_queue": self._queue_metric_dict(0.0, max_waiting_cpu),
                "waiting_gpu_queue": self._queue_metric_dict(0.0, max_waiting_gpu),
                "ready_queue": self._queue_metric_dict(0.0, max_ready),
            }

        disk_area = 0.0
        disk_active_s = 0.0
        h2d_area = 0.0
        h2d_active_s = 0.0
        compute_area = 0.0
        compute_active_s = 0.0
        waiting_cpu_area = 0.0
        waiting_gpu_area = 0.0
        ready_area = 0.0

        for index, sample in enumerate(samples[:-1]):
            next_sample = samples[index + 1]
            dt = max(0.0, next_sample.timestamp_s - sample.timestamp_s)
            if dt <= 0:
                continue

            disk_inflight = min(sample.loading_cpu_count, self.disk_worker_capacity)
            disk_area += disk_inflight * dt
            if disk_inflight > 0:
                disk_active_s += dt

            h2d_area += sample.loading_gpu_count * dt
            if sample.loading_gpu_count > 0:
                h2d_active_s += dt

            compute_area += sample.compute_pending_count * dt
            if sample.compute_pending_count > 0:
                compute_active_s += dt

            waiting_cpu_area += sample.waiting_cpu_len * dt
            waiting_gpu_area += sample.waiting_gpu_len * dt
            ready_area += sample.ready_count * dt

        return {
            "observed_s": observed_s,
            "disk_worker_capacity": self.disk_worker_capacity,
            "disk_workers": self._resource_metric_dict(
                busy_ratio=disk_area / (observed_s * max(1, self.disk_worker_capacity)),
                active_ratio=disk_active_s / observed_s,
                avg_inflight=disk_area / observed_s,
                max_inflight=max_loading_cpu,
            ),
            "h2d": self._resource_metric_dict(
                busy_ratio=h2d_active_s / observed_s,
                active_ratio=h2d_active_s / observed_s,
                avg_inflight=h2d_area / observed_s,
                max_inflight=max_loading_gpu,
            ),
            "compute_pending": self._resource_metric_dict(
                busy_ratio=compute_active_s / observed_s,
                active_ratio=compute_active_s / observed_s,
                avg_inflight=compute_area / observed_s,
                max_inflight=max_compute,
            ),
            "waiting_cpu_queue": self._queue_metric_dict(waiting_cpu_area / observed_s, max_waiting_cpu),
            "waiting_gpu_queue": self._queue_metric_dict(waiting_gpu_area / observed_s, max_waiting_gpu),
            "ready_queue": self._queue_metric_dict(ready_area / observed_s, max_ready),
        }

    @staticmethod
    def _resource_metric_dict(
        busy_ratio: float,
        active_ratio: float,
        avg_inflight: float,
        max_inflight: int,
    ) -> dict[str, Any]:
        """构造资源利用率指标字典。"""

        return {
            "busy_ratio": busy_ratio,
            "active_ratio": active_ratio,
            "avg_inflight": avg_inflight,
            "max_inflight": max_inflight,
        }

    @staticmethod
    def _queue_metric_dict(avg_len: float, max_len: int) -> dict[str, Any]:
        """构造队列长度指标字典。"""

        return {
            "avg_len": avg_len,
            "max_len": max_len,
        }
