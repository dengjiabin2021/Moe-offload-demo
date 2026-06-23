"""异步 MoE Profiler 报告对象。"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from .wrappers import AsyncCacheTracker, AsyncStageTracker, CachePhaseStats, PHASE_DECODE, PHASE_LOADING, PHASE_PREFILL


@dataclass
class MemorySample:
    """一次后台内存采样。"""

    timestamp_s: float
    phase: str
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    cpu_rss_mb: float

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        return asdict(self)


@dataclass
class AsyncProfileReport:
    """异步 expert 加载运行的一次完整 profile 报告。"""

    model_dir: str
    cache_manager_name: str
    build_time_s: float
    load_time_s: float
    generate_time_s: float
    prefill_time_s: float
    decode_time_s: float
    prefill_token_count: int
    decode_token_count: int
    output_text: str
    memory_timeline: list[MemorySample] = field(default_factory=list)
    cache_tracker: AsyncCacheTracker | None = None
    stage_tracker: AsyncStageTracker | None = None

    @property
    def tokens_per_second(self) -> float:
        """返回 generate 阶段的平均生成吞吐。"""

        if self.generate_time_s <= 0:
            return 0.0
        return self.decode_token_count / self.generate_time_s

    def print_summary(self) -> None:
        """打印终端摘要。"""

        sep = "=" * 88
        sub = "-" * 88
        print("\n" + sep)
        print("  异步 MoE Runtime Profiler 报告")
        print(sep)
        print(f"  模型目录: {self.model_dir}")
        print(f"  CacheManager: {self.cache_manager_name}")
        print()

        self._print_timing_section(sub)
        self._print_cache_section(sub)
        self._print_async_wait_section(sub)
        self._print_block_wait_reason_section(sub)
        self._print_resource_utilization_section(sub)
        self._print_layer_wait_topk_section(sub)
        self._print_io_section(sub)
        self._print_memory_section(sub)

        print(sep + "\n")

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的字典。"""

        return {
            "model_dir": self.model_dir,
            "cache_manager_name": self.cache_manager_name,
            "timing": {
                "build_time_s": self.build_time_s,
                "load_time_s": self.load_time_s,
                "generate_time_s": self.generate_time_s,
                "prefill_time_s": self.prefill_time_s,
                "decode_time_s": self.decode_time_s,
                "tokens_per_second": self.tokens_per_second,
                "prefill_token_count": self.prefill_token_count,
                "decode_token_count": self.decode_token_count,
            },
            "cache": self.cache_tracker.to_dict() if self.cache_tracker is not None else {},
            "async_stages": self.stage_tracker.to_dict() if self.stage_tracker is not None else {},
            "memory_timeline": [sample.to_dict() for sample in self.memory_timeline],
            "output_text": self.output_text,
        }

    def save_json(self, path: str | Path) -> None:
        """保存 JSON 报告。"""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save_memory_csv(self, path: str | Path) -> None:
        """保存内存采样 CSV。"""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timestamp_s",
                    "phase",
                    "gpu_allocated_mb",
                    "gpu_reserved_mb",
                    "cpu_rss_mb",
                ],
            )
            writer.writeheader()
            for sample in self.memory_timeline:
                writer.writerow(sample.to_dict())

    # ==================================================================
    # 终端输出
    # ==================================================================

    def _print_timing_section(self, sub: str) -> None:
        """打印阶段耗时。"""

        print("  [1] 阶段耗时")
        print(sub)
        print(f"  build_components: {self.build_time_s:>10.3f}s")
        print(f"  load_model:       {self.load_time_s:>10.3f}s")
        print(f"  generate:         {self.generate_time_s:>10.3f}s")
        print(f"  prefill:          {self.prefill_time_s:>10.3f}s  tokens={self.prefill_token_count}")
        print(f"  decode:           {self.decode_time_s:>10.3f}s  tokens={self.decode_token_count}")
        print(f"  tokens/s:         {self.tokens_per_second:>10.3f}")
        print()

    def _print_cache_section(self, sub: str) -> None:
        """打印缓存命中总览、分阶段统计和按层统计。"""

        print("  [2] 缓存命中")
        print(sub)
        if self.cache_tracker is None:
            print("  无缓存统计")
            print()
            return

        print("  阶段缓存命中 (prefill cache / decode cache)")
        print("  phase       gpu      cpu     disk    total    gpu_rate   cpu_rate   disk_rate")
        print("  " + "-" * 78)
        self._print_cache_row("total", self.cache_tracker.total)
        self._print_cache_row("prefill", self.cache_tracker.prefill)
        self._print_cache_row("decode", self.cache_tracker.decode)
        if self.cache_tracker.other.total:
            self._print_cache_row("other", self.cache_tracker.other)

        print()
        print("  按层缓存命中 (layer cache)")
        print("  layer       gpu      cpu     disk    total    gpu_rate   cpu_rate   disk_rate")
        print("  " + "-" * 78)
        for layer_idx, stats in sorted(self.cache_tracker.layer_stats.items()):
            self._print_cache_row(str(layer_idx), stats)
        print()

    @staticmethod
    def _print_cache_row(label: str, stats: CachePhaseStats) -> None:
        """打印一行缓存命中统计。"""

        print(
            f"  {label:<8}"
            f"{stats.gpu_hits:>7}"
            f"{stats.cpu_hits:>9}"
            f"{stats.disk_misses:>9}"
            f"{stats.total:>9}"
            f"{stats.gpu_hit_rate:>10.2%}"
            f"{stats.cpu_hit_rate:>11.2%}"
            f"{stats.disk_miss_rate:>11.2%}"
        )

    def _print_async_wait_section(self, sub: str) -> None:
        """打印异步状态机等待统计。"""

        print("  [3] 异步阶段等待")
        print(sub)
        if self.stage_tracker is None:
            print("  无异步阶段统计")
            print()
            return

        waits = self.stage_tracker.to_dict()["stage_waits"]
        for name in ("waiting_cpu", "loading_cpu", "waiting_gpu", "loading_gpu", "compute_pending"):
            stats = waits.get(name, {})
            print(
                f"  {name:<16}"
                f"count={stats.get('count', 0):>6}  "
                f"total={stats.get('total_s', 0.0):>10.3f}s  "
                f"avg={stats.get('avg_s', 0.0) * 1000:>10.3f}ms  "
                f"max={stats.get('max_s', 0.0) * 1000:>10.3f}ms"
            )

        public = self.stage_tracker.to_dict()["public_methods"]
        wait_ready = public.get("wait_any_ready", {})
        print(
            f"  {'wait_any_ready':<16}"
            f"count={wait_ready.get('count', 0):>6}  "
            f"total={wait_ready.get('total_s', 0.0):>10.3f}s  "
            f"avg={wait_ready.get('avg_s', 0.0) * 1000:>10.3f}ms  "
            f"max={wait_ready.get('max_s', 0.0) * 1000:>10.3f}ms"
        )
        print()
        print("  说明: 前五项是按 expert 生命周期累加，可能大于 generate；wait_any_ready 才是 block 实际阻塞时间。")
        print()

    def _print_block_wait_reason_section(self, sub: str) -> None:
        """打印 block 真实阻塞时间的原因归因。"""

        print("  [4] Block 等待归因")
        print(sub)
        if self.stage_tracker is None:
            print("  无 block 等待归因")
            print()
            return

        data = self.stage_tracker.to_dict()
        wait_reasons = data.get("wait_reasons", {})
        if not wait_reasons:
            print("  无 wait_any_ready 阻塞记录")
            print()
            return

        total_wait_s = sum(stats.get("total_s", 0.0) for stats in wait_reasons.values())
        print("  reason             count      total_s      avg_ms      max_ms     share")
        print("  " + "-" * 76)
        for reason, stats in sorted(wait_reasons.items(), key=lambda item: item[1].get("total_s", 0.0), reverse=True):
            total_s = stats.get("total_s", 0.0)
            share = total_s / total_wait_s if total_wait_s > 0 else 0.0
            print(
                f"  {reason:<16}"
                f"{stats.get('count', 0):>8}"
                f"{total_s:>13.3f}"
                f"{stats.get('avg_s', 0.0) * 1000:>12.3f}"
                f"{stats.get('max_s', 0.0) * 1000:>12.3f}"
                f"{share:>10.2%}"
            )

        print()
        print("  按阶段等待归因")
        print("  phase       reason             count      total_s      avg_ms")
        print("  " + "-" * 70)
        by_phase = data.get("wait_reasons_by_phase", {})
        for phase, reason_stats in sorted(by_phase.items()):
            for reason, stats in sorted(reason_stats.items(), key=lambda item: item[1].get("total_s", 0.0), reverse=True):
                print(
                    f"  {phase:<10}"
                    f"{reason:<16}"
                    f"{stats.get('count', 0):>8}"
                    f"{stats.get('total_s', 0.0):>13.3f}"
                    f"{stats.get('avg_s', 0.0) * 1000:>12.3f}"
                )
        print()

    def _print_resource_utilization_section(self, sub: str) -> None:
        """打印后台调度资源利用率。"""

        print("  [5] 后台资源利用率")
        print(sub)
        if self.stage_tracker is None:
            print("  无后台资源采样")
            print()
            return

        utilization = self.stage_tracker.to_dict().get("resource_utilization", {})
        observed_s = utilization.get("observed_s", 0.0)
        if observed_s <= 0:
            print("  无足够采样点")
            print()
            return

        print(f"  observed: {observed_s:.3f}s, disk_worker_capacity={utilization.get('disk_worker_capacity', 0)}")
        print("  resource           busy/active      avg_inflight      max_inflight")
        print("  " + "-" * 74)
        self._print_resource_row("disk_workers", utilization.get("disk_workers", {}), use_busy=True)
        self._print_resource_row("h2d", utilization.get("h2d", {}), use_busy=False)
        self._print_resource_row("compute_pending", utilization.get("compute_pending", {}), use_busy=False)

        print()
        print("  queue              avg_len          max_len")
        print("  " + "-" * 46)
        self._print_queue_row("waiting_cpu", utilization.get("waiting_cpu_queue", {}))
        self._print_queue_row("waiting_gpu", utilization.get("waiting_gpu_queue", {}))
        self._print_queue_row("ready", utilization.get("ready_queue", {}))
        print()

    @staticmethod
    def _print_resource_row(label: str, stats: dict[str, Any], *, use_busy: bool) -> None:
        """打印一行资源利用率。"""

        ratio_key = "busy_ratio" if use_busy else "active_ratio"
        print(
            f"  {label:<18}"
            f"{stats.get(ratio_key, 0.0):>12.2%}"
            f"{stats.get('avg_inflight', 0.0):>18.3f}"
            f"{stats.get('max_inflight', 0):>18}"
        )

    @staticmethod
    def _print_queue_row(label: str, stats: dict[str, Any]) -> None:
        """打印一行队列长度。"""

        print(
            f"  {label:<18}"
            f"{stats.get('avg_len', 0.0):>12.3f}"
            f"{stats.get('max_len', 0):>16}"
        )

    def _print_layer_wait_topk_section(self, sub: str, top_k: int = 10) -> None:
        """打印 block 等待最重的层。"""

        print("  [6] 按层等待 Top-K")
        print(sub)
        if self.stage_tracker is None:
            print("  无按层等待统计")
            print()
            return

        by_layer = self.stage_tracker.to_dict().get("wait_by_layer", {})
        if not by_layer:
            print("  无 wait_any_ready 阻塞记录")
            print()
            return

        rows = sorted(by_layer.items(), key=lambda item: item[1].get("total_s", 0.0), reverse=True)[:top_k]
        print("  layer      count      total_s      avg_ms      max_ms    main_reason")
        print("  " + "-" * 78)
        for layer_idx, stats in rows:
            print(
                f"  {layer_idx:<8}"
                f"{stats.get('count', 0):>8}"
                f"{stats.get('total_s', 0.0):>13.3f}"
                f"{stats.get('avg_s', 0.0) * 1000:>12.3f}"
                f"{stats.get('max_s', 0.0) * 1000:>12.3f}"
                f"    {stats.get('main_reason', '')}"
            )
        print()

    def _print_io_section(self, sub: str) -> None:
        """打印分阶段 IO 统计。"""

        print("  [7] IO 分阶段统计")
        print(sub)
        print("  phase       direction        size_mb      time_s   throughput_mb_s")
        print("  " + "-" * 70)

        if self.stage_tracker is None:
            print("  无 IO 统计")
            print()
            return

        data = self.stage_tracker.to_dict()
        io_by_phase = data.get("io_by_phase", {})
        phase_labels = [
            (PHASE_LOADING, "loading"),
            (PHASE_PREFILL, "prefill"),
            (PHASE_DECODE, "decode"),
        ]
        for phase_key, phase_label in phase_labels:
            phase_io = io_by_phase.get(phase_key, {})
            self._print_io_row(phase_label, "disk->CPU", phase_io.get("disk_to_cpu", {}))
            self._print_io_row(phase_label, "CPU->GPU", phase_io.get("cpu_to_gpu", {}))

        print()
        print("  说明: loading 表示常驻权重加载；prefill/decode 表示异步 routed expert 运行阶段。")
        print()

    @staticmethod
    def _print_io_row(label: str, direction: str, stats: dict[str, Any]) -> None:
        """打印一行 IO 统计。"""

        print(
            f"  {label:<10}"
            f"{direction:<14}"
            f"{stats.get('mb', 0.0):>12.2f}"
            f"{stats.get('total_s', 0.0):>12.3f}"
            f"{stats.get('throughput_mbps', 0.0):>18.2f}"
        )

    def _print_memory_section(self, sub: str) -> None:
        """打印分阶段内存峰值。"""

        print("  [8] 内存峰值")
        print(sub)
        for phase in (PHASE_LOADING, PHASE_PREFILL, PHASE_DECODE):
            samples = [sample for sample in self.memory_timeline if sample.phase == phase]
            gpu_alloc = max((sample.gpu_allocated_mb for sample in samples), default=0.0)
            gpu_reserved = max((sample.gpu_reserved_mb for sample in samples), default=0.0)
            cpu_rss = max((sample.cpu_rss_mb for sample in samples), default=0.0)
            print(
                f"  {phase:<8}"
                f"gpu_alloc={gpu_alloc:>10.1f} MB  "
                f"gpu_reserved={gpu_reserved:>10.1f} MB  "
                f"cpu_rss={cpu_rss:>10.1f} MB"
            )
        print()


# 兼容旧入口名称；当前实现已经是异步专用报告。
ProfileReport = AsyncProfileReport
