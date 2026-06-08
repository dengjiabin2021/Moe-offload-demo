"""Profile 报告模块。

ProfileReport 是一个纯数据类，提供：
- print_summary(): 终端友好的分栏摘要输出（CJK 字符宽度自适应）
- to_dict(): 转为可序列化字典
- save_json(): 导出 JSON 文件
- save_memory_csv(): 导出内存时间线 CSV
"""

from __future__ import annotations

import json
import csv
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .wrappers import CachePhaseStats, IOTracker


# ==========================================================================
# CJK 终端对齐工具
# ==========================================================================

def _wcwidth(ch: str) -> int:
    """返回单个字符在终端中的显示宽度（CJK 字符 = 2，其余 = 1）。"""
    ea = unicodedata.east_asian_width(ch)
    return 2 if ea in ("W", "F") else 1


def _dsp(s: str) -> int:
    """计算字符串在终端中的显示宽度。"""
    return sum(_wcwidth(ch) for ch in s)


def _rpad(s: str, target_width: int) -> str:
    """在字符串右侧填充空格，使其终端显示宽度达到 target_width。"""
    cur = _dsp(s)
    return s + " " * max(0, target_width - cur)


def _lpad(s: str, target_width: int) -> str:
    """在字符串左侧填充空格，使其终端显示宽度达到 target_width。"""
    cur = _dsp(s)
    return " " * max(0, target_width - cur) + s


# ==========================================================================
# ProfileReport
# ==========================================================================

@dataclass
class ProfileReport:
    """MoE Runtime 一次运行的全量探测报告。

    属性按指标类别分为五组：耗时、内存、吞吐量、缓存、附加数据。
    """

    # ================================================================
    # 耗时（秒）
    # ================================================================
    loading_time_s: float = 0.0
    prefill_time_s: float = 0.0
    prefill_gpu_time_s: float = 0.0
    prefill_data_time_s: float = 0.0
    decode_time_s: float = 0.0
    decode_gpu_time_s: float = 0.0
    decode_data_time_s: float = 0.0
    decode_gpu_time_per_token_s: float = 0.0
    prefill_token_count: int = 0
    decode_token_count: int = 0

    # ================================================================
    # 内存峰值（MB）
    # ================================================================
    loading_peak_gpu_mb: float = 0.0
    loading_peak_cpu_mb: float = 0.0
    prefill_peak_gpu_mb: float = 0.0
    prefill_peak_cpu_mb: float = 0.0
    decode_peak_gpu_mb: float = 0.0
    decode_peak_cpu_mb: float = 0.0

    # ================================================================
    # 内存时间线
    # ================================================================
    memory_timeline: list[Any] = field(default_factory=list)

    # ================================================================
    # 吞吐量（MB/s）
    # ================================================================
    loading_disk_throughput_mbps: float = 0.0
    prefill_disk_throughput_mbps: float = 0.0
    decode_disk_throughput_mbps: float = 0.0
    loading_cpu_gpu_throughput_mbps: float = 0.0
    prefill_cpu_gpu_throughput_mbps: float = 0.0
    decode_cpu_gpu_throughput_mbps: float = 0.0

    # ================================================================
    # 缓存统计
    # ================================================================
    gpu_cache_hits: int = 0
    cpu_cache_hits: int = 0
    disk_fetches: int = 0
    gpu_cache_hit_rate: float = 0.0
    cpu_cache_hit_rate: float = 0.0
    prefill_cache_stats: CachePhaseStats = field(default_factory=CachePhaseStats)
    decode_cache_stats: CachePhaseStats = field(default_factory=CachePhaseStats)

    # ================================================================
    # 附加数据
    # ================================================================
    total_forward_count: int = 0
    io_tracker_snapshot: IOTracker | None = None
    layer_cache_stats: dict[int, CachePhaseStats] = field(default_factory=dict)

    # ==================================================================
    # 终端摘要输出
    # ==================================================================

    # 各 section 统一的 label 显示宽度
    _LABEL_W = 38   # 标签栏目标显示宽度
    _VAL_W = 10     # 数值栏目标显示宽度

    def print_summary(self) -> None:
        """在终端打印格式化的探测摘要。"""
        w = 64
        sep = "=" * w
        sub = "-" * w

        print("\n" + sep)
        print("  MoE Runtime Profiler — 探测报告")
        print(sep)

        self._print_timing_section(sub)
        self._print_memory_section(sub)
        self._print_throughput_section(sub)
        self._print_cache_section(sub)

        print(sep + "\n")

    # ---- 内部：行格式化 ----

    @staticmethod
    def _row(label: str, value: str, label_w: int = 38) -> str:
        """格式化一行：标签左对齐（CJK 宽度感知），数值右对齐。"""
        return f"  {_rpad(label, label_w)} {value}"

    @staticmethod
    def _row_s(label: str, value: float, label_w: int = 38, unit: str = "s") -> str:
        """格式化浮点数值行。"""
        return ProfileReport._row(label, f"{value:>10.2f} {unit}", label_w)

    @staticmethod
    def _row_i(label: str, value: int, label_w: int = 38, unit: str = "") -> str:
        """格式化整型数值行。"""
        v = f"{value:>10d}{' ' + unit if unit else ''}"
        return ProfileReport._row(label, v, label_w)

    # ==================================================================
    # [1] 耗时分析
    # ==================================================================

    def _print_timing_section(self, sub: str) -> None:
        LW = self._LABEL_W
        print("\n  [1] 耗时分析")
        print(sub)
        print(self._row_s("加载阶段总耗时:",           self.loading_time_s, LW))
        print()
        print(self._row_s("预填充阶段 (wall-clock):",   self.prefill_time_s, LW))
        print(self._row_s("  |-- GPU 总耗时 (含搬运):",  self.prefill_gpu_time_s, LW))
        print(self._row_s("  |-- 数据搬运 (cache 追踪):", self.prefill_data_time_s, LW))
        compute_p = max(0, self.prefill_gpu_time_s - self.prefill_data_time_s)
        print(self._row_s("  |-- GPU compute (推算):",   compute_p, LW))
        print(self._row_i("  |-- 处理 token 数:",        self.prefill_token_count, LW))
        if self.prefill_token_count > 0 and self.prefill_time_s > 0:
            tps = self.prefill_token_count / self.prefill_time_s
            print(self._row("  |-- 吞吐:", f"{tps:>10.1f} tokens/s", LW))
        print()
        print(self._row_s("解码阶段 (wall-clock):",     self.decode_time_s, LW))
        print(self._row_s("  |-- GPU 总耗时 (含搬运):",  self.decode_gpu_time_s, LW))
        print(self._row_s("  |-- 数据搬运 (cache 追踪):", self.decode_data_time_s, LW))
        compute_d = max(0, self.decode_gpu_time_s - self.decode_data_time_s)
        print(self._row_s("  |-- GPU compute (推算):",   compute_d, LW))
        print(self._row_i("  |-- 生成 token 数:",        self.decode_token_count, LW))
        if self.decode_gpu_time_per_token_s > 0:
            ms = self.decode_gpu_time_per_token_s * 1000
            tps = 1.0 / self.decode_gpu_time_per_token_s if self.decode_gpu_time_per_token_s > 0 else 0.0
            print(self._row("  |-- 吞吐:", f"{tps:>10.1f} tokens/s", LW))
        print()
        total = self.loading_time_s + self.prefill_time_s + self.decode_time_s
        print(self._row_s("全程总耗时:", total, LW))

    # ==================================================================
    # [2] 内存峰值分析
    # ==================================================================

    def _print_memory_section(self, sub: str) -> None:
        print("\n  [2] 内存峰值分析 (MB)")
        print(sub)

        # 列：阶段(12) | GPU已分配(12) | GPU已保留(12) | CPU RSS(12)
        CW = 14  # column width (display)
        hdr = (
            f"  {_rpad('阶段', CW)}"
            f"{_rpad('GPU 已分配', CW)}"
            f"{_rpad('GPU 已保留', CW)}"
            f"{_rpad('CPU RSS', CW)}"
        )
        print(hdr)
        print(f"  {'-' * (_dsp(hdr) - 2)}")

        def _gpu_res_peak(samples, phase: str) -> float:
            s = [x for x in samples if x.phase == phase]
            return max((x.gpu_reserved_mb for x in s), default=0.0)

        for label, phase_key, gpu_alloc, cpu_peak in [
            ("加载",   "loading",  self.loading_peak_gpu_mb,   self.loading_peak_cpu_mb),
            ("预填充", "prefill",  self.prefill_peak_gpu_mb,   self.prefill_peak_cpu_mb),
            ("解码",   "decode",   self.decode_peak_gpu_mb,    self.decode_peak_cpu_mb),
        ]:
            gpu_res = _gpu_res_peak(self.memory_timeline, phase_key)
            vals = (
                f"{_rpad(label, CW)}"
                f"{_lpad(f'{gpu_alloc:,.1f} MB', CW)}"
                f"{_lpad(f'{gpu_res:,.1f} MB', CW)}"
                f"{_lpad(f'{cpu_peak:,.1f} MB', CW)}"
            )
            print(f"  {vals}")

        print()
        print(f"  共采集 {len(self.memory_timeline)} 个内存采样点")

    # ==================================================================
    # [3] 吞吐量分析
    # ==================================================================

    def _print_throughput_section(self, sub: str) -> None:
        print("\n  [3] 吞吐量分析 (MB/s)")
        print(sub)

        CW = 14
        hdr = (
            f"  {_rpad('阶段', CW)}"
            f"{_rpad('磁盘 IO', CW)}"
            f"{_rpad('CPU → GPU', CW)}"
        )
        print(hdr)
        print(f"  {'-' * (_dsp(hdr) - 2)}")

        for label, disk, cpugpu in [
            ("加载",   self.loading_disk_throughput_mbps,    self.loading_cpu_gpu_throughput_mbps),
            ("预填充", self.prefill_disk_throughput_mbps,    self.prefill_cpu_gpu_throughput_mbps),
            ("解码",   self.decode_disk_throughput_mbps,     self.decode_cpu_gpu_throughput_mbps),
        ]:
            vals = (
                f"{_rpad(label, CW)}"
                f"{_lpad(f'{disk:,.2f}', CW)}"
                f"{_lpad(f'{cpugpu:,.2f}', CW)}"
            )
            print(f"  {vals}")

        if self.io_tracker_snapshot is not None:
            io = self.io_tracker_snapshot
            print()
            print(f"  磁盘 IO 累计:   "
                  f"加载 {io.loading_disk_bytes/1024/1024:.1f} MB  |  "
                  f"预填充 {io.prefill_disk_bytes/1024/1024:.1f} MB  |  "
                  f"解码 {io.decode_disk_bytes/1024/1024:.1f} MB")
            print(f"  CPU → GPU 累计: "
                  f"加载 {io.loading_cpu_gpu_bytes/1024/1024:.1f} MB  |  "
                  f"预填充 {io.prefill_cpu_gpu_bytes/1024/1024:.1f} MB  |  "
                  f"解码 {io.decode_cpu_gpu_bytes/1024/1024:.1f} MB")

    # ==================================================================
    # [4] 缓存策略分析
    # ==================================================================

    def _print_cache_section(self, sub: str) -> None:
        print("\n  [4] 缓存策略分析")
        print(sub)

        total = self.gpu_cache_hits + self.cpu_cache_hits + self.disk_fetches
        if total == 0:
            print("  (未启用缓存或未发生专家请求)")
            return

        LW = 30
        print(self._row_i("总请求数:",                    total, LW))
        print(self._row("GPU 命中次数:", f"{self.gpu_cache_hits:>10d}  ({self.gpu_cache_hit_rate:>8.1%})", LW))
        print(self._row("CPU 命中次数:", f"{self.cpu_cache_hits:>10d}  ({self.cpu_cache_hit_rate:>8.1%})", LW))
        miss_rate = 1.0 - self.gpu_cache_hit_rate - self.cpu_cache_hit_rate
        print(self._row("磁盘回源 (未命中):", f"{self.disk_fetches:>10d}  ({miss_rate:>8.1%})", LW))

        # 分阶段
        for phase_label, stats in [
            ("预填充阶段", self.prefill_cache_stats),
            ("解码阶段",   self.decode_cache_stats),
        ]:
            if stats.total_requests == 0:
                continue
            print(f"\n  {phase_label}:")
            print(f"    请求数:      {stats.total_requests:>6d}")
            print(f"    GPU 命中:    {stats.gpu_hits:>6d}  ({stats.gpu_hit_rate:>8.1%})")
            print(f"    CPU 命中:    {stats.cpu_hits:>6d}  ({stats.cpu_hit_rate:>8.1%})")
            print(f"    磁盘回源:    {stats.disk_fetches:>6d}  ({stats.miss_rate:>8.1%})")

        # 逐层统计（表格）
        if self.layer_cache_stats:
            self._print_layer_cache_table()

    def _print_layer_cache_table(self) -> None:
        """打印逐层缓存统计表格。"""
        CW = 10  # 每列显示宽度
        cols = [
            ("Layer",    CW),
            ("GPU命中",  CW),
            ("CPU命中",  CW),
            ("磁盘",     CW),
            ("合计",     CW),
            ("命中率",   CW),
        ]
        # 表头
        hdr = "  " + "".join(_rpad(c, w) for c, w in cols)
        print(f"\n  {'─' * (_dsp(hdr) - 2)}")
        print(f"  逐层缓存统计")
        print(f"  {'─' * (_dsp(hdr) - 2)}")
        print(hdr)
        print(f"  {'-' * (_dsp(hdr) - 2)}")

        for layer_idx in sorted(self.layer_cache_stats.keys()):
            s = self.layer_cache_stats[layer_idx]
            t = s.total_requests
            rate = s.gpu_hit_rate if t > 0 else 0.0
            cells = [
                f"{layer_idx}",
                f"{s.gpu_hits}",
                f"{s.cpu_hits}",
                f"{s.disk_fetches}",
                f"{t}",
                f"{rate:.1%}",
            ]
            line = "  " + "".join(
                _rpad(c, w) if i == 0 else _lpad(c, w)
                for i, (c, w) in enumerate(zip(cells, [w for _, w in cols]))
            )
            print(line)

    # ==================================================================
    # JSON / CSV 导出（不变）
    # ==================================================================

    def to_dict(self) -> dict[str, Any]:
        """将所有探测数据转为可 JSON 序列化的字典。"""
        return {
            "timing": {
                "loading_time_s": self.loading_time_s,
                "prefill": {
                    "wall_clock_time_s": self.prefill_time_s,
                    "gpu_total_time_s": self.prefill_gpu_time_s,
                    "data_movement_time_s": self.prefill_data_time_s,
                    "gpu_compute_time_s": max(0, self.prefill_gpu_time_s - self.prefill_data_time_s),
                    "token_count": self.prefill_token_count,
                    "tokens_per_second": (
                        self.prefill_token_count / self.prefill_time_s
                        if self.prefill_time_s > 0
                        else 0
                    ),
                },
                "decode": {
                    "wall_clock_time_s": self.decode_time_s,
                    "gpu_total_time_s": self.decode_gpu_time_s,
                    "data_movement_time_s": self.decode_data_time_s,
                    "gpu_compute_time_s": max(0, self.decode_gpu_time_s - self.decode_data_time_s),
                    "token_count": self.decode_token_count,
                    "gpu_ms_per_token": self.decode_gpu_time_per_token_s * 1000,
                    "gpu_tokens_per_second": (
                        1.0 / self.decode_gpu_time_per_token_s
                        if self.decode_gpu_time_per_token_s > 0
                        else 0
                    ),
                },
                "total_time_s": self.loading_time_s + self.prefill_time_s + self.decode_time_s,
            },
            "memory_peak_mb": {
                "loading": {"gpu": self.loading_peak_gpu_mb, "cpu": self.loading_peak_cpu_mb},
                "prefill": {"gpu": self.prefill_peak_gpu_mb, "cpu": self.prefill_peak_cpu_mb},
                "decode":  {"gpu": self.decode_peak_gpu_mb,  "cpu": self.decode_peak_cpu_mb},
            },
            "memory_timeline": [
                {
                    "timestamp_s": s.timestamp_s,
                    "phase": s.phase,
                    "gpu_allocated_mb": s.gpu_allocated_mb,
                    "gpu_reserved_mb": s.gpu_reserved_mb,
                    "cpu_rss_mb": s.cpu_rss_mb,
                }
                for s in self.memory_timeline
            ],
            "throughput_mbps": {
                "loading":  {"disk_io": self.loading_disk_throughput_mbps,  "cpu_to_gpu": self.loading_cpu_gpu_throughput_mbps},
                "prefill":  {"disk_io": self.prefill_disk_throughput_mbps,  "cpu_to_gpu": self.prefill_cpu_gpu_throughput_mbps},
                "decode":   {"disk_io": self.decode_disk_throughput_mbps,   "cpu_to_gpu": self.decode_cpu_gpu_throughput_mbps},
            },
            "cache": {
                "total_requests": self.gpu_cache_hits + self.cpu_cache_hits + self.disk_fetches,
                "gpu_hits":       self.gpu_cache_hits,
                "gpu_hit_rate":   self.gpu_cache_hit_rate,
                "cpu_hits":       self.cpu_cache_hits,
                "cpu_hit_rate":   self.cpu_cache_hit_rate,
                "disk_fetches":   self.disk_fetches,
                "disk_miss_rate": (1.0 - self.gpu_cache_hit_rate - self.cpu_cache_hit_rate),
                "prefill": {
                    "requests":    self.prefill_cache_stats.total_requests,
                    "gpu_hits":    self.prefill_cache_stats.gpu_hits,
                    "gpu_hit_rate": self.prefill_cache_stats.gpu_hit_rate,
                    "cpu_hits":    self.prefill_cache_stats.cpu_hits,
                    "cpu_hit_rate": self.prefill_cache_stats.cpu_hit_rate,
                    "disk_fetches": self.prefill_cache_stats.disk_fetches,
                    "miss_rate":   self.prefill_cache_stats.miss_rate,
                },
                "decode": {
                    "requests":    self.decode_cache_stats.total_requests,
                    "gpu_hits":    self.decode_cache_stats.gpu_hits,
                    "gpu_hit_rate": self.decode_cache_stats.gpu_hit_rate,
                    "cpu_hits":    self.decode_cache_stats.cpu_hits,
                    "cpu_hit_rate": self.decode_cache_stats.cpu_hit_rate,
                    "disk_fetches": self.decode_cache_stats.disk_fetches,
                    "miss_rate":   self.decode_cache_stats.miss_rate,
                },
            },
            "meta": {
                "total_forward_passes": self.total_forward_count,
            },
            "layer_cache": {
                str(layer_idx): {
                    "gpu_hits": s.gpu_hits,
                    "cpu_hits": s.cpu_hits,
                    "disk_fetches": s.disk_fetches,
                    "total_requests": s.total_requests,
                    "gpu_hit_rate": s.gpu_hit_rate,
                    "cpu_hit_rate": s.cpu_hit_rate,
                    "miss_rate": s.miss_rate,
                }
                for layer_idx, s in sorted(self.layer_cache_stats.items())
            },
        }

    def save_json(self, filepath: str | Path) -> None:
        """导出全量探测数据为 JSON 文件。"""
        filepath = Path(filepath)
        data = self.to_dict()
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"[Profiler] 报告已保存到: {filepath.resolve()}")

    def save_memory_csv(self, filepath: str | Path) -> None:
        """导出内存时间线为 CSV 文件。"""
        filepath = Path(filepath)
        with filepath.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "phase", "gpu_allocated_mb", "gpu_reserved_mb", "cpu_rss_mb"])
            for s in self.memory_timeline:
                writer.writerow([
                    f"{s.timestamp_s:.3f}",
                    s.phase,
                    f"{s.gpu_allocated_mb:.2f}",
                    f"{s.gpu_reserved_mb:.2f}",
                    f"{s.cpu_rss_mb:.2f}",
                ])
        print(f"[Profiler] 内存时间线已保存到: {filepath.resolve()}")
