"""异步 MoE Runtime Profiler。

Profiler 只通过运行期 monkeypatch 采集指标，不修改 ``moe_runtime`` 的逻辑代码。
它面向当前异步接口：

- ``request_expert``
- ``is_expert_ready``
- ``wait_any_ready``
- ``get_ready_expert``
- ``release_expert``

同时补充异步状态机内部阶段的等待时间统计。
"""

from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any

import torch

from .reporter import AsyncProfileReport, MemorySample
from .wrappers import (
    AsyncCacheTracker,
    AsyncStageTracker,
    PHASE_DECODE,
    PHASE_IDLE,
    PHASE_LOADING,
    PHASE_PREFILL,
)


class _MemorySampler:
    """后台内存采样线程。

    采样逻辑低频运行，不参与 CacheManager 热路径；CPU RSS 依赖 psutil，
    如果环境中没有安装 psutil，则 CPU RSS 记为 0。
    """

    def __init__(self, interval_s: float) -> None:
        self._interval_s = interval_s
        self._samples: list[MemorySample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._phase_ref: list[str] = [PHASE_IDLE]
        self._start_s = 0.0

    def start(self, phase_ref: list[str]) -> None:
        """启动后台采样。"""

        if self._thread is not None and self._thread.is_alive():
            return
        self._phase_ref = phase_ref
        self._start_s = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="async-profiler-memory", daemon=True)
        self._thread.start()

    def stop(self) -> list[MemorySample]:
        """停止采样并返回采样点。"""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            return list(self._samples)

    def _run(self) -> None:
        """采样循环。"""

        try:
            import psutil

            process = psutil.Process()
        except Exception:
            process = None

        while not self._stop.is_set():
            timestamp_s = time.perf_counter() - self._start_s
            gpu_allocated_mb = 0.0
            gpu_reserved_mb = 0.0
            if torch.cuda.is_available():
                gpu_allocated_mb = torch.cuda.memory_allocated() / 1024**2
                gpu_reserved_mb = torch.cuda.memory_reserved() / 1024**2

            cpu_rss_mb = 0.0
            if process is not None:
                try:
                    cpu_rss_mb = process.memory_info().rss / 1024**2
                except Exception:
                    cpu_rss_mb = 0.0

            sample = MemorySample(
                timestamp_s=timestamp_s,
                phase=self._phase_ref[0],
                gpu_allocated_mb=gpu_allocated_mb,
                gpu_reserved_mb=gpu_reserved_mb,
                cpu_rss_mb=cpu_rss_mb,
            )
            with self._lock:
                self._samples.append(sample)

            self._stop.wait(self._interval_s)


class Profiler:
    """异步 MoE Runtime 的运行期 profile 注入器。"""

    def __init__(self, *, memory_sample_interval_s: float = 0.1) -> None:
        """初始化 profiler 状态。"""

        self._phase: list[str] = [PHASE_IDLE]
        self._memory_sampler = _MemorySampler(memory_sample_interval_s)
        self._cache_tracker = AsyncCacheTracker(self._phase)
        self._stage_tracker = AsyncStageTracker()

        self._engine: Any | None = None
        self._build_time_s = 0.0
        self._load_time_s = 0.0
        self._generate_time_s = 0.0
        self._prefill_start_s = 0.0
        self._prefill_end_s = 0.0
        self._decode_start_s = 0.0
        self._decode_end_s = 0.0
        self._prefill_token_count = 0
        self._decode_token_count = 0
        self._forward_count = 0
        self._output_text = ""
        self._tokenizer: Any | None = None
        self._expert_bytes = 0

    def wrap(self, engine: Any, *, tokenizer: Any | None = None) -> None:
        """把 profiler 注入到 ``MoeRuntimeEngine``。

        必须在 ``engine.build_components()`` 前调用。注入后，Profiler 会在
        build 阶段 patch loader/cache manager，在 load 阶段注册 forward hook。
        """

        self._engine = engine
        self._tokenizer = tokenizer
        self._patch_engine_methods(engine)

    def report(self) -> AsyncProfileReport:
        """停止采样并生成报告。"""

        memory_timeline = self._memory_sampler.stop()
        engine = self._engine
        cache_name = ""
        model_dir = ""
        if engine is not None:
            model_dir = str(engine.config.model_dir)
            if getattr(engine, "cache_manager", None) is not None:
                cache_name = type(engine.cache_manager).__name__

        return AsyncProfileReport(
            model_dir=model_dir,
            cache_manager_name=cache_name,
            build_time_s=self._build_time_s,
            load_time_s=self._load_time_s,
            generate_time_s=self._generate_time_s,
            prefill_time_s=max(0.0, self._prefill_end_s - self._prefill_start_s),
            decode_time_s=max(0.0, self._decode_end_s - self._decode_start_s),
            prefill_token_count=self._prefill_token_count,
            decode_token_count=self._decode_token_count,
            output_text=self._output_text,
            memory_timeline=memory_timeline,
            cache_tracker=self._cache_tracker,
            stage_tracker=self._stage_tracker,
        )

    # ==================================================================
    # Engine patch
    # ==================================================================

    def _patch_engine_methods(self, engine: Any) -> None:
        """包装 engine 的 build/load/generate 三个入口。"""

        original_build = engine.build_components
        original_load = engine.load_model
        original_generate = engine.generate

        def profiled_build_components() -> None:
            self._phase[0] = PHASE_LOADING
            self._memory_sampler.start(self._phase)
            started_at = time.perf_counter()
            original_build()
            self._build_time_s = time.perf_counter() - started_at

            if engine.tensor_loader is not None:
                self._patch_tensor_loader(engine.tensor_loader)
            if engine.cache_manager is not None:
                self._patch_async_cache_manager(engine.cache_manager)

        def profiled_load_model() -> None:
            started_at = time.perf_counter()
            original_load()
            self._load_time_s = time.perf_counter() - started_at
            self._register_forward_hooks(engine)

        def profiled_generate(*args: Any, **kwargs: Any) -> Any:
            started_at = time.perf_counter()
            result = original_generate(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._generate_time_s = time.perf_counter() - started_at
            self._decode_end_s = time.perf_counter()
            self._phase[0] = PHASE_IDLE
            self._capture_output_text(result)
            return result

        engine.build_components = profiled_build_components
        engine.load_model = profiled_load_model
        engine.generate = profiled_generate

    # ==================================================================
    # TensorLoader patch
    # ==================================================================

    def _patch_tensor_loader(self, loader: Any) -> None:
        """原地包装 TensorLoader 的 IO 方法。"""

        if getattr(loader, "_async_profiler_patched", False):
            return
        loader._async_profiler_patched = True

        self._wrap_loader_method(loader, "load_tensors_by_names", routed_expert_io=False)
        self._wrap_loader_method(loader, "load_tensor_by_name", routed_expert_io=False)
        self._wrap_loader_method(loader, "load_tensors_to_buffer", routed_expert_io=True)

    def _wrap_loader_method(self, loader: Any, method_name: str, *, routed_expert_io: bool) -> None:
        """包装单个 TensorLoader 方法。"""

        if not hasattr(loader, method_name):
            return

        original = getattr(loader, method_name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tensor_names = self._extract_tensor_names(method_name, args, kwargs)
            total_bytes = sum(self._tensor_nbytes(loader, name) for name in tensor_names)
            phase = self._phase[0]
            target_device = self._extract_target_device(method_name, args, kwargs)
            started_at = time.perf_counter()
            result = original(*args, **kwargs)
            elapsed_s = time.perf_counter() - started_at
            if routed_expert_io:
                self._stage_tracker.record_disk_io(total_bytes, elapsed_s, phase)
            else:
                self._stage_tracker.record_resident_io(
                    total_bytes,
                    elapsed_s,
                    phase,
                    includes_h2d=target_device.startswith("cuda"),
                )
            return result

        setattr(loader, method_name, wrapper)

    @staticmethod
    def _extract_tensor_names(method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
        """从 TensorLoader 调用参数中提取 tensor 名称。"""

        if method_name == "load_tensor_by_name":
            tensor_name = kwargs.get("tensor_name")
            if tensor_name is None and args:
                tensor_name = args[0]
            return [tensor_name] if tensor_name is not None else []

        tensor_names = kwargs.get("tensor_names")
        if tensor_names is None and args:
            tensor_names = args[0]
        return list(tensor_names or [])

    @staticmethod
    def _extract_target_device(method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        """从 TensorLoader 调用参数中推断目标设备。"""

        if method_name in ("load_tensor_by_name", "load_tensors_by_names"):
            device = kwargs.get("device")
            if device is None and len(args) >= 2:
                device = args[1]
            return str(device or "cpu")

        buffer = kwargs.get("buffer")
        if buffer is None and len(args) >= 2:
            buffer = args[1]
        if buffer is not None and hasattr(buffer, "device"):
            return str(buffer.device)
        return "cpu"

    @staticmethod
    def _tensor_nbytes(loader: Any, tensor_name: str) -> int:
        """从 manifest 推断 tensor 字节数。"""

        try:
            entry = loader._read_manifest_entry(tensor_name=tensor_name)
            return int(entry["nbytes"])
        except Exception:
            shape, elems, dtype = loader.get_tensor_info(tensor_name=tensor_name)
            return int(elems) * torch.empty((), dtype=dtype).element_size()

    # ==================================================================
    # Async CacheManager patch
    # ==================================================================

    def _patch_async_cache_manager(self, cache_manager: Any) -> None:
        """原地包装异步 CacheManager。

        这里依赖的是当前 async CacheManager 暴露的接口和内部状态表。所有包装都只读
        或记录统计，不改变原始状态机决策。
        """

        if getattr(cache_manager, "_async_profiler_patched", False):
            return
        cache_manager._async_profiler_patched = True

        self._wrap_public_cache_methods(cache_manager)
        self._wrap_enqueue(cache_manager)
        self._expert_bytes = self._compute_expert_bytes(cache_manager)
        self._wrap_scheduler_method(cache_manager, "_schedule_disk_reads_locked")
        self._wrap_scheduler_method(cache_manager, "_complete_finished_disk_reads_locked")
        self._wrap_scheduler_method(cache_manager, "_schedule_h2d_locked")
        self._wrap_scheduler_method(cache_manager, "_complete_finished_h2d_locked")
        self._wrap_scheduler_method(cache_manager, "_complete_finished_compute_locked")

    def _wrap_public_cache_methods(self, cache_manager: Any) -> None:
        """包装 CacheManager 公开接口。"""

        self._wrap_timed_public_method(cache_manager, "request_expert")
        self._wrap_timed_public_method(cache_manager, "get_ready_expert")

        original_is_ready = cache_manager.is_expert_ready

        def is_ready_wrapper(*args: Any, **kwargs: Any) -> Any:
            layer_idx, expert_idx = self._layer_expert_from_call(args, kwargs)
            key = (int(layer_idx), int(expert_idx))
            will_count_gpu_hit = self._will_is_ready_create_gpu_hit(cache_manager, key)
            started_at = time.perf_counter()
            result = original_is_ready(*args, **kwargs)
            elapsed_s = time.perf_counter() - started_at
            self._stage_tracker.record_public_method("is_expert_ready", elapsed_s)
            if will_count_gpu_hit and result:
                self._cache_tracker.record("gpu", key[0])
            return result

        cache_manager.is_expert_ready = is_ready_wrapper

        original_wait = cache_manager.wait_any_ready

        def wait_wrapper(*args: Any, **kwargs: Any) -> Any:
            layer_idx, expert_ids = self._layer_expert_ids_from_wait_call(args, kwargs)
            pending_keys = [(layer_idx, expert_idx) for expert_idx in expert_ids]
            enter_counts, enter_states = self._snapshot_pending_states(cache_manager, pending_keys)
            started_at = time.perf_counter()
            result = original_wait(*args, **kwargs)
            elapsed_s = time.perf_counter() - started_at
            exit_counts, _ = self._snapshot_pending_states(cache_manager, pending_keys)
            ready_expert_idx = int(result)
            dominant_reason = self._dominant_wait_reason(
                ready_key=(layer_idx, ready_expert_idx),
                enter_counts=enter_counts,
                enter_states=enter_states,
            )
            self._stage_tracker.record_public_method("wait_any_ready", elapsed_s)
            self._stage_tracker.record_wait_any_ready(
                layer_idx=layer_idx,
                pending_count=len(expert_ids),
                elapsed_s=elapsed_s,
                ready_expert_idx=ready_expert_idx,
                phase=self._phase[0],
                dominant_reason=dominant_reason,
                enter_counts=enter_counts,
                exit_counts=exit_counts,
            )
            return result

        cache_manager.wait_any_ready = wait_wrapper

        original_release = cache_manager.release_expert

        def release_wrapper(*args: Any, **kwargs: Any) -> Any:
            layer_idx, expert_idx = self._layer_expert_from_call(args, kwargs)
            key = (int(layer_idx), int(expert_idx))
            started_at = time.perf_counter()
            result = original_release(*args, **kwargs)
            elapsed_s = time.perf_counter() - started_at
            self._stage_tracker.record_public_method("release_expert", elapsed_s)
            self._stage_tracker.start_stage("compute_pending", key, time.perf_counter(), self._phase[0])
            return result

        cache_manager.release_expert = release_wrapper

    def _wrap_timed_public_method(self, cache_manager: Any, method_name: str) -> None:
        """包装一个只需要计数/计时的公开方法。"""

        original = getattr(cache_manager, method_name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started_at = time.perf_counter()
            result = original(*args, **kwargs)
            elapsed_s = time.perf_counter() - started_at
            self._stage_tracker.record_public_method(method_name, elapsed_s)
            return result

        setattr(cache_manager, method_name, wrapper)

    def _wrap_enqueue(self, cache_manager: Any) -> None:
        """包装 _enqueue_request，记录 CPU/GPU/disk 位置分类和排队起点。"""

        original = cache_manager._enqueue_request

        def wrapper(key: tuple[int, int]) -> Any:
            norm_key = (int(key[0]), int(key[1]))
            kind = self._classify_expert_location(cache_manager, norm_key)
            result = original(norm_key)
            self._cache_tracker.record(kind, norm_key[0])
            now_s = time.perf_counter()
            if kind == "cpu":
                self._stage_tracker.start_stage("waiting_gpu", norm_key, now_s, self._phase[0])
            elif kind == "disk":
                self._stage_tracker.start_stage("waiting_cpu", norm_key, now_s, self._phase[0])
            return result

        cache_manager._enqueue_request = wrapper

    def _wrap_scheduler_method(self, cache_manager: Any, method_name: str) -> None:
        """包装异步调度器内部方法，并在状态集合变化时补充阶段计时。"""

        original = getattr(cache_manager, method_name)

        def wrapper() -> bool:
            before = self._snapshot_async_sets(cache_manager)
            started_at = time.perf_counter()
            self._record_resource_sample(cache_manager, started_at)
            progressed = original()
            ended_at = time.perf_counter()
            after = self._snapshot_async_sets(cache_manager)
            self._record_resource_sample(cache_manager, ended_at)
            self._record_stage_transitions(method_name, before, after, ended_at)
            self._stage_tracker.record_scheduler_method(method_name, ended_at - started_at, bool(progressed))
            return progressed

        setattr(cache_manager, method_name, wrapper)

    @staticmethod
    def _will_is_ready_create_gpu_hit(cache_manager: Any, key: tuple[int, int]) -> bool:
        """判断 is_expert_ready 是否会把 GPU used slot 转成 ready。"""

        if key in cache_manager._ready or key in cache_manager._requests:
            return False
        gpu_slot = cache_manager._find_expert_in_slots(
            cache_manager._gpu_slots,
            cache_manager._gpu_layer_slot_map,
            key[0],
            key[1],
        )
        return gpu_slot is not None

    @staticmethod
    def _classify_expert_location(cache_manager: Any, key: tuple[int, int]) -> str:
        """按当前 slot 表判断 expert 位于 GPU、CPU 还是磁盘。"""

        gpu_slot = cache_manager._find_expert_in_slots(
            cache_manager._gpu_slots,
            cache_manager._gpu_layer_slot_map,
            key[0],
            key[1],
        )
        if gpu_slot is not None:
            return "gpu"

        cpu_slot = cache_manager._find_expert_in_slots(
            cache_manager._cpu_slots,
            cache_manager._cpu_layer_slot_map,
            key[0],
            key[1],
        )
        if cpu_slot is not None:
            return "cpu"

        return "disk"

    @staticmethod
    def _layer_expert_ids_from_wait_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, list[int]]:
        """从 wait_any_ready 调用中提取 layer_idx 和 pending expert 列表。"""

        layer_idx = kwargs.get("layer_idx", args[0] if len(args) > 0 else None)
        expert_ids = kwargs.get("expert_ids", args[1] if len(args) > 1 else None)
        if layer_idx is None or expert_ids is None:
            raise ValueError("wait_any_ready call must contain layer_idx and expert_ids.")
        return int(layer_idx), [int(expert_idx) for expert_idx in expert_ids]

    def _snapshot_pending_states(
        self,
        cache_manager: Any,
        keys: list[tuple[int, int]],
    ) -> tuple[dict[str, int], dict[tuple[int, int], str]]:
        """读取一组 pending expert 当前在异步状态机中的位置。

        这里会短暂借用 CacheManager 的条件变量锁，确保快照不会和后台调度线程
        的集合修改交错。该锁只用于读集合，不做任何调度决策。
        """

        lock = getattr(cache_manager, "_cv", None)
        if lock is None:
            return self._snapshot_pending_states_unlocked(cache_manager, keys)
        with lock:
            return self._snapshot_pending_states_unlocked(cache_manager, keys)

    def _snapshot_pending_states_unlocked(
        self,
        cache_manager: Any,
        keys: list[tuple[int, int]],
    ) -> tuple[dict[str, int], dict[tuple[int, int], str]]:
        """在调用方已处理锁语义时读取 pending 状态。"""

        counts: dict[str, int] = {}
        states: dict[tuple[int, int], str] = {}
        for key in keys:
            state = self._classify_pending_state(cache_manager, key)
            states[key] = state
            counts[state] = counts.get(state, 0) + 1
        return counts, states

    @staticmethod
    def _classify_pending_state(cache_manager: Any, key: tuple[int, int]) -> str:
        """把一个 expert key 分类到异步状态机的可解释状态。"""

        if key in cache_manager._ready:
            return "ready"
        if key in cache_manager._loading_gpu:
            return "loading_gpu"
        if key in cache_manager._waiting_gpu:
            return "waiting_gpu"
        if key in cache_manager._loading_cpu:
            return "loading_cpu"
        if key in cache_manager._waiting_cpu:
            return "waiting_cpu"
        if key in cache_manager._compute_pending:
            return "compute_pending"
        if key in cache_manager._requests:
            return "requested"
        return "missing"

    @staticmethod
    def _dominant_wait_reason(
        *,
        ready_key: tuple[int, int],
        enter_counts: dict[str, int],
        enter_states: dict[tuple[int, int], str],
    ) -> str:
        """选择本次 block 阻塞的主因。

        优先使用最终唤醒 block 的那个 expert 在进入等待时的状态；如果它没有
        出现在快照中，则退化为 pending 集合中数量最多的状态。
        """

        ready_state = enter_states.get(ready_key)
        if ready_state:
            return ready_state
        if not enter_counts:
            return "unknown"
        return max(enter_counts.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _snapshot_async_sets(cache_manager: Any) -> dict[str, set[tuple[int, int]]]:
        """复制异步状态集合，用于比较调度前后变化。"""

        return {
            "waiting_cpu": set(cache_manager._waiting_cpu),
            "loading_cpu": set(cache_manager._loading_cpu),
            "waiting_gpu": set(cache_manager._waiting_gpu),
            "loading_gpu": set(cache_manager._loading_gpu),
            "compute_pending": set(cache_manager._compute_pending),
        }

    def _record_resource_sample(self, cache_manager: Any, timestamp_s: float) -> None:
        """记录后台调度器当前队列和 inflight 状态。"""

        self._stage_tracker.record_resource_sample(
            timestamp_s=timestamp_s,
            phase=self._phase[0],
            waiting_cpu_len=len(cache_manager._waiting_cpu),
            loading_cpu_count=len(cache_manager._loading_cpu),
            waiting_gpu_len=len(cache_manager._waiting_gpu),
            loading_gpu_count=len(cache_manager._loading_gpu),
            ready_count=len(cache_manager._ready),
            compute_pending_count=len(cache_manager._compute_pending),
        )

    def _record_stage_transitions(
        self,
        method_name: str,
        before: dict[str, set[tuple[int, int]]],
        after: dict[str, set[tuple[int, int]]],
        timestamp_s: float,
    ) -> None:
        """根据状态集合变化记录异步阶段进入/离开时间。"""

        if method_name == "_schedule_disk_reads_locked":
            for key in after["loading_cpu"] - before["loading_cpu"]:
                finished = self._stage_tracker.finish_stage("waiting_cpu", key, timestamp_s)
                phase = finished[1] if finished is not None else self._phase[0]
                self._stage_tracker.start_stage("loading_cpu", key, timestamp_s, phase)
            return

        if method_name == "_complete_finished_disk_reads_locked":
            phase_by_key: dict[tuple[int, int], str | None] = {}
            for key in before["loading_cpu"] - after["loading_cpu"]:
                finished = self._stage_tracker.finish_stage("loading_cpu", key, timestamp_s)
                phase_by_key[key] = finished[1] if finished is not None else self._phase[0]
            for key in after["waiting_gpu"] - before["waiting_gpu"]:
                phase = phase_by_key.get(key, self._phase[0])
                self._stage_tracker.start_stage("waiting_gpu", key, timestamp_s, phase)
            return

        if method_name == "_schedule_h2d_locked":
            for key in after["loading_gpu"] - before["loading_gpu"]:
                finished = self._stage_tracker.finish_stage("waiting_gpu", key, timestamp_s)
                phase = finished[1] if finished is not None else self._phase[0]
                self._stage_tracker.start_stage("loading_gpu", key, timestamp_s, phase)
            return

        if method_name == "_complete_finished_h2d_locked":
            for key in before["loading_gpu"] - after["loading_gpu"]:
                finished = self._stage_tracker.finish_stage("loading_gpu", key, timestamp_s)
                if finished is not None:
                    elapsed_s, phase = finished
                    self._stage_tracker.record_cpu_gpu_io(self._expert_bytes, elapsed_s, phase)
            return

        if method_name == "_complete_finished_compute_locked":
            for key in before["compute_pending"] - after["compute_pending"]:
                self._stage_tracker.finish_stage("compute_pending", key, timestamp_s)

    @staticmethod
    def _compute_expert_bytes(cache_manager: Any) -> int:
        """根据 CacheManager 内部布局计算单个 expert 的字节数。"""

        expert_elems = int(getattr(cache_manager, "_expert_elems", 0))
        expert_dtype = getattr(cache_manager, "_expert_dtype", torch.float16)
        return expert_elems * torch.empty((), dtype=expert_dtype).element_size()

    @staticmethod
    def _layer_expert_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[int, int]:
        """从 CacheManager 方法调用中提取 layer_idx 和 expert_idx。"""

        layer_idx = kwargs.get("layer_idx", args[0] if len(args) > 0 else None)
        expert_idx = kwargs.get("expert_idx", args[1] if len(args) > 1 else None)
        if layer_idx is None or expert_idx is None:
            raise ValueError("CacheManager call must contain layer_idx and expert_idx.")
        return int(layer_idx), int(expert_idx)

    # ==================================================================
    # Forward hook
    # ==================================================================

    def _register_forward_hooks(self, engine: Any) -> None:
        """注册模型 forward hook，用于划分 prefill/decode。"""

        model = engine.model_runtime.model
        if model is None:
            return

        def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if self._forward_count == 0:
                self._phase[0] = PHASE_PREFILL
                self._prefill_start_s = time.perf_counter()
                self._prefill_token_count = self._infer_input_tokens(args, kwargs)
            else:
                self._phase[0] = PHASE_DECODE
                if self._forward_count == 1:
                    self._decode_start_s = time.perf_counter()

        def post_hook(_module: Any, _args: tuple[Any, ...], _output: Any) -> Any:
            if self._forward_count == 0:
                self._prefill_end_s = time.perf_counter()
            self._forward_count += 1
            return _output

        model.register_forward_pre_hook(pre_hook, with_kwargs=True)
        model.register_forward_hook(post_hook)

    @staticmethod
    def _infer_input_tokens(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
        """从模型 forward 参数里推断 prefill token 数。"""

        input_tensor = None
        if args and hasattr(args[0], "shape"):
            input_tensor = args[0]
        elif "input_ids" in kwargs and hasattr(kwargs["input_ids"], "shape"):
            input_tensor = kwargs["input_ids"]
        elif "inputs_embeds" in kwargs and hasattr(kwargs["inputs_embeds"], "shape"):
            input_tensor = kwargs["inputs_embeds"]

        if input_tensor is None or len(input_tensor.shape) < 2:
            return 0
        return int(input_tensor.shape[1])

    def _capture_output_text(self, output_ids: Any) -> None:
        """如果传入 tokenizer，则保存生成文本，便于报告核对。"""

        if self._tokenizer is None:
            return
        try:
            self._output_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if hasattr(output_ids, "shape") and output_ids.ndim >= 2:
                self._decode_token_count = max(0, int(output_ids.shape[-1]) - self._prefill_token_count)
        except Exception:
            self._output_text = ""
