"""零侵入 Profiler 主类。

Profiler 是整个探测系统的唯一入口，用户只需要：
1. 创建 Profiler 实例
2. 调用 profiler.wrap(engine) 注入探测代理
3. 正常使用 engine
4. 调用 profiler.report() 获取报告
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from .wrappers import (
    PHASE_IDLE,
    PHASE_LOADING,
    PHASE_PREFILL,
    PHASE_DECODE,
    ProfiledTensorLoader,
    ProfiledCacheManager,
    IOTracker,
    CacheTracker,
)
from .reporter import ProfileReport

# ==========================================================================
# 内存采样点
# ==========================================================================

@dataclass
class MemorySample:
    """单次内存采样点（后台线程周期采集）。"""

    timestamp_s: float  # 自采样开始的秒数
    phase: str  # 当前阶段: loading | prefill | decode
    gpu_allocated_mb: float  # torch.cuda.memory_allocated() / 1024²
    gpu_reserved_mb: float  # torch.cuda.memory_reserved() / 1024²
    cpu_rss_mb: float  # 进程 RSS，psutil 不可用时为 0


# ==========================================================================
# 后台内存采样线程
# ==========================================================================

class _MemorySampler:
    """后台线程，按固定间隔采样 GPU / CPU 内存使用。

    线程安全：采样数据通过锁保护；phase_ref 的读取在 CPython GIL 下原子。
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self._interval_s = interval_s
        self._samples: list[MemorySample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float = 0.0
        self._phase_ref: list[str] = [PHASE_IDLE]

    def start(self, phase_ref: list[str]) -> None:
        """启动后台采样线程。

        Args:
            phase_ref: 指向 Profiler._phase 的可变列表引用，
                       采样线程读取 phase_ref[0] 获取当前阶段。
        """
        self._t0 = time.perf_counter()
        self._phase_ref = phase_ref
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="profiler-mem-sampler")
        self._thread.start()

    def stop(self) -> list[MemorySample]:
        """停止采样并返回所有采样点。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            return list(self._samples)

    def _run(self) -> None:
        """后台采样循环。"""
        # 尝试导入 psutil（可选依赖）
        try:
            import psutil
            process = psutil.Process()
            has_psutil = True
        except ImportError:
            process = None
            has_psutil = False

        has_cuda = torch.cuda.is_available()

        while not self._stop.is_set():
            ts = time.perf_counter() - self._t0

            # GPU 显存
            gpu_alloc = 0.0
            gpu_reserved = 0.0
            if has_cuda:
                gpu_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_reserved = torch.cuda.memory_reserved() / (1024 * 1024)

            # CPU 内存（RSS）
            cpu_rss = 0.0
            if has_psutil and process is not None:
                try:
                    cpu_rss = process.memory_info().rss / (1024 * 1024)
                except Exception:
                    cpu_rss = 0.0

            sample = MemorySample(
                timestamp_s=ts,
                phase=self._phase_ref[0],
                gpu_allocated_mb=gpu_alloc,
                gpu_reserved_mb=gpu_reserved,
                cpu_rss_mb=cpu_rss,
            )
            with self._lock:
                self._samples.append(sample)

            self._stop.wait(self._interval_s)


# ==========================================================================
# Profiler 主类
# ==========================================================================

class Profiler:
    """MoE Runtime 零侵入性能探测器。

    使用方式::

        profiler = Profiler()
        profiler.wrap(engine)        # ← 唯一需要在现有代码中添加的一行
        engine.build_components()
        engine.load_model()
        engine.generate(input_ids=..., max_new_tokens=128)
        report = profiler.report()
        report.print_summary()       # 终端友好摘要
        report.save_json("out.json") # 结构化数据导出
    """

    def __init__(self) -> None:
        # ---- 阶段追踪 ----
        # 使用单元素列表作为可变引用，供 MemorySampler / IOTracker / CacheTracker 共享读取
        self._phase: list[str] = [PHASE_IDLE]

        # ---- 各阶段 wall-clock 时间边界 ----
        self._t_load_start: float = 0.0
        self._t_load_end: float = 0.0
        self._t_prefill_start: float = 0.0
        self._t_prefill_end: float = 0.0
        self._t_decode_start: float = 0.0
        self._t_decode_end: float = 0.0

        # ---- 内存采样器 ----
        self._sampler = _MemorySampler(interval_s=0.1)

        # ---- IO 和缓存追踪器 ----
        self._io_tracker = IOTracker()
        self._cache_tracker = CacheTracker()

        # ---- Forward pass 计数器 ----
        self._forward_count: int = 0
        self._prefill_gpu_time_s: float = 0.0
        self._prefill_data_time_s: float = 0.0
        self._decode_data_time_s: float = 0.0
        self._prefill_token_count: int = 0
        self._decode_forward_gpu_times: list[float] = []

        # ---- CUDA events (用于 GPU 侧精确定时) ----
        self._cuda_start_event: Any = None
        self._has_cuda: bool = torch.cuda.is_available()

        # ---- 引擎引用（用于在 wrap 后访问内部组件） ----
        self._engine: Any = None

    # ==================================================================
    # 公开 API
    # ==================================================================

    def wrap(self, engine: Any) -> None:
        """将探测代理注入到 MoeRuntimeEngine。

        **必须**在 ``build_components()`` 之前调用。

        该方法包装了三个关键方法：
        - build_components(): 注入 ProfiledTensorLoader / ProfiledCacheManager
        - load_model(): 标记加载阶段结束，注册 forward hooks
        - generate(): 标记解码阶段结束
        """
        self._engine = engine

        # 让 IO / Cache 追踪器共享阶段引用
        self._io_tracker._phase_ref = self._phase
        self._cache_tracker._phase_ref = self._phase

        # ---- 1. 包装 build_components ----
        original_build = engine.build_components

        def profiled_build() -> None:
            self._phase[0] = PHASE_LOADING
            self._t_load_start = time.perf_counter()
            self._sampler.start(self._phase)

            # 执行原始构建逻辑
            original_build()

            # --- 注入 ProfiledTensorLoader ---
            if engine.tensor_loader is not None:
                engine.tensor_loader = ProfiledTensorLoader(
                    engine.tensor_loader, self._io_tracker
                )
                if engine.model_runtime is not None:
                    engine.model_runtime.tensor_loader = engine.tensor_loader

            # --- 注入 ProfiledCacheManager ---
            if engine.cache_manager is not None:
                engine.cache_manager = ProfiledCacheManager(
                    engine.cache_manager, self._cache_tracker, self._io_tracker
                )
                # 更新 cache_manager 内部的 loader 引用（指向 profiled loader）
                engine.cache_manager._real.loader = engine.tensor_loader
                if engine.model_runtime is not None:
                    engine.model_runtime.cache_manager = engine.cache_manager

        engine.build_components = profiled_build

        # ---- 2. 包装 load_model ----
        original_load = engine.load_model

        def profiled_load() -> None:
            result = original_load()
            self._t_load_end = time.perf_counter()

            # 注册 forward hooks 用于 phase 检测和 GPU 计时
            self._register_forward_hooks(engine)

            # 如果配置了 cache_manager，更新 block 中的引用
            # （load_model 内部 patch_moe_block 传入的是当时的 cache_manager，
            #   在 profiled_build 中已替换 engine.model_runtime.cache_manager，
            #   load_model 拿到的就是 wrapped 版本，无需额外处理）
            return result

        engine.load_model = profiled_load

        # ---- 3. 包装 generate ----
        original_generate = engine.generate

        def profiled_generate(*args: Any, **kwargs: Any) -> Any:
            result = original_generate(*args, **kwargs)
            self._t_decode_end = time.perf_counter()
            self._phase[0] = PHASE_IDLE
            return result

        engine.generate = profiled_generate

    def report(self) -> "ProfileReport":
        """停止采样，汇总所有探测数据，返回结构化报告。"""

        # 停止内存采样
        memory_samples = self._sampler.stop()

        # ---- 阶段耗时 ----
        load_time = (
            self._t_load_end - self._t_load_start if self._t_load_end > 0 else 0.0
        )
        prefill_time = (
            self._t_prefill_end - self._t_prefill_start
            if self._t_prefill_end > 0
            else 0.0
        )
        decode_time = (
            self._t_decode_end - self._t_decode_start
            if self._t_decode_end > 0
            else 0.0
        )

        # ---- 解码阶段 per-token 耗时 ----
        decode_tokens = len(self._decode_forward_gpu_times)
        decode_gpu_total = sum(self._decode_forward_gpu_times)
        decode_gpu_per_token = decode_gpu_total / decode_tokens if decode_tokens > 0 else 0.0

        # ---- 预填充阶段 GPU 耗时（第一次 forward） ----
        prefill_gpu_time = self._prefill_gpu_time_s
        prefill_data_time = self._prefill_data_time_s
        decode_data_time = self._decode_data_time_s

        # ---- 各阶段显存峰值 ----
        def peak_in_phase(
            samples: list[MemorySample], phase_name: str
        ) -> tuple[float, float]:
            phase_samples = [s for s in samples if s.phase == phase_name]
            if not phase_samples:
                return 0.0, 0.0
            return (
                max(s.gpu_allocated_mb for s in phase_samples),
                max(s.cpu_rss_mb for s in phase_samples),
            )

        load_gpu_peak, load_cpu_peak = peak_in_phase(memory_samples, PHASE_LOADING)
        prefill_gpu_peak, prefill_cpu_peak = peak_in_phase(memory_samples, PHASE_PREFILL)
        decode_gpu_peak, decode_cpu_peak = peak_in_phase(memory_samples, PHASE_DECODE)

        # ---- 磁盘 IO 吞吐量 (MB/s) ----
        def safe_mbps(bytes_val: int, time_val: float) -> float:
            if time_val <= 0:
                return 0.0
            return bytes_val / (1024 * 1024) / time_val

        load_disk_mbps = safe_mbps(
            self._io_tracker.loading_disk_bytes, load_time
        )
        prefill_disk_mbps = safe_mbps(
            self._io_tracker.prefill_disk_bytes, prefill_time
        )
        decode_disk_mbps = safe_mbps(
            self._io_tracker.decode_disk_bytes, decode_time
        )

        # ---- CPU→GPU 吞吐量 (MB/s) ----
        load_cpu_gpu_mbps = safe_mbps(
            self._io_tracker.loading_cpu_gpu_bytes, load_time
        )
        prefill_cpu_gpu_mbps = safe_mbps(
            self._io_tracker.prefill_cpu_gpu_bytes, prefill_time
        )
        decode_cpu_gpu_mbps = safe_mbps(
            self._io_tracker.decode_cpu_gpu_bytes, decode_time
        )

        return ProfileReport(
            # —— 耗时 ——
            loading_time_s=load_time,
            prefill_time_s=prefill_time,
            prefill_gpu_time_s=prefill_gpu_time,
            prefill_data_time_s=prefill_data_time,
            decode_time_s=decode_time,
            decode_gpu_time_s=decode_gpu_total,
            decode_data_time_s=decode_data_time,
            decode_gpu_time_per_token_s=decode_gpu_per_token,
            prefill_token_count=self._prefill_token_count,
            decode_token_count=decode_tokens,
            # —— 内存峰值 ——
            loading_peak_gpu_mb=load_gpu_peak,
            loading_peak_cpu_mb=load_cpu_peak,
            prefill_peak_gpu_mb=prefill_gpu_peak,
            prefill_peak_cpu_mb=prefill_cpu_peak,
            decode_peak_gpu_mb=decode_gpu_peak,
            decode_peak_cpu_mb=decode_cpu_peak,
            # —— 内存时间线 ——
            memory_timeline=memory_samples,
            # —— 吞吐量 ——
            loading_disk_throughput_mbps=load_disk_mbps,
            prefill_disk_throughput_mbps=prefill_disk_mbps,
            decode_disk_throughput_mbps=decode_disk_mbps,
            loading_cpu_gpu_throughput_mbps=load_cpu_gpu_mbps,
            prefill_cpu_gpu_throughput_mbps=prefill_cpu_gpu_mbps,
            decode_cpu_gpu_throughput_mbps=decode_cpu_gpu_mbps,
            # —— 缓存 ——
            gpu_cache_hits=self._cache_tracker.gpu_hits,
            cpu_cache_hits=self._cache_tracker.cpu_hits,
            disk_fetches=self._cache_tracker.disk_fetches,
            gpu_cache_hit_rate=self._cache_tracker.gpu_hit_rate,
            cpu_cache_hit_rate=self._cache_tracker.cpu_hit_rate,
            prefill_cache_stats=self._cache_tracker.prefill,
            decode_cache_stats=self._cache_tracker.decode,
            # —— 附加数据 ——
            total_forward_count=self._forward_count,
            io_tracker_snapshot=self._io_tracker,
            layer_cache_stats=self._cache_tracker.layer_stats,
        )

    # ==================================================================
    # 内部：Forward Hook 注册
    # ==================================================================

    def _register_forward_hooks(self, engine: Any) -> None:
        """在模型上注册 forward hooks，用于：
        1. 检测 prefill → decode 阶段切换
        2. 用 CUDA Event 精确测量每次 forward 的 GPU 耗时
        """
        model = engine.model_runtime.model
        if model is None:
            return

        profiler_ref = self  # 闭包引用

        def pre_forward_hook(module: Any, args: tuple, kwargs: dict) -> None:
            # ---- 阶段检测 ----
            if profiler_ref._forward_count == 0:
                profiler_ref._phase[0] = PHASE_PREFILL
                profiler_ref._t_prefill_start = time.perf_counter()

                # 从输入推断 prefill token 数
                # HuggingFace generate() 可能把 input_ids 放在 args[0] 或 kwargs 中
                input_tensor = None
                if args and hasattr(args[0], "shape"):
                    input_tensor = args[0]
                elif kwargs and "input_ids" in kwargs:
                    input_tensor = kwargs["input_ids"]

                if input_tensor is not None and len(input_tensor.shape) >= 2:
                    profiler_ref._prefill_token_count = int(input_tensor.shape[1])
            else:
                profiler_ref._phase[0] = PHASE_DECODE
                if profiler_ref._forward_count == 1:
                    profiler_ref._t_decode_start = time.perf_counter()

            # ---- CUDA Event 开始计时 ----
            if profiler_ref._has_cuda:
                profiler_ref._cuda_start_event = torch.cuda.Event(enable_timing=True)
                profiler_ref._cuda_start_event.record()

            # ---- 重置 per-forward 数据搬运计时 ----
            profiler_ref._io_tracker.reset_per_forward()

        def forward_hook(module: Any, args: tuple, output: Any) -> Any:
            # ---- CUDA Event 结束计时 ----
            if profiler_ref._has_cuda and profiler_ref._cuda_start_event is not None:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = profiler_ref._cuda_start_event.elapsed_time(end_event)
                elapsed_s = elapsed_ms / 1000.0
            else:
                elapsed_s = 0.0

            # 本次 forward 内部的数据搬运耗时（由 ProfiledCacheManager 累计）
            data_time = profiler_ref._io_tracker.per_forward_data_time_s

            if profiler_ref._forward_count == 0:
                # ---- 预填充阶段结束 ----
                profiler_ref._t_prefill_end = time.perf_counter()
                profiler_ref._prefill_gpu_time_s = elapsed_s
                profiler_ref._prefill_data_time_s = data_time
            else:
                # ---- 解码阶段：逐 token 记录 ----
                profiler_ref._decode_forward_gpu_times.append(elapsed_s)
                profiler_ref._decode_data_time_s += data_time

            profiler_ref._forward_count += 1
            return output

        model.register_forward_pre_hook(pre_forward_hook, with_kwargs=True)
        model.register_forward_hook(forward_hook)
