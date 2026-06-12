from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from math import prod
import threading
from typing import Any

import torch

from moe_runtime.config import RuntimeConfig
from moe_runtime.loader import ManifestTensorLoader


@dataclass
class _SlotInfo:
    """缓存槽位元信息。"""

    layer_idx: int | None = None
    expert_idx: int | None = None
    status: str = "free"  # free | used | protected | loading
    last_used_step: int = 0
    generation: int = 0


@dataclass
class _ExpertRequest:
    """一次 expert 请求的内部调度状态。"""

    layer_idx: int
    expert_idx: int
    stage: str  # waiting_cpu | loading_cpu | waiting_gpu | loading_gpu | ready | compute_pending
    cpu_slot_idx: int | None = None
    gpu_slot_idx: int | None = None
    cpu_generation: int | None = None
    gpu_generation: int | None = None
    disk_future: Future | None = None
    h2d_event: torch.cuda.Event | None = None
    compute_event: torch.cuda.Event | None = None
    error: BaseException | None = None


class AsyncExpertCacheManager:
    """异步专家缓存管理器。

    block 只提交 expert 请求、查询 ready、取 GPU 权重 view、计算后 release。
    CacheManager 自己的后台调度线程负责推进磁盘 IO、H2D 和计算完成后的 slot 释放。
    """

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        adapter: Any,
        tensor_loader: ManifestTensorLoader,
        model_config: Any | None = None,
    ) -> None:
        self.config = runtime_config
        self.adapter = adapter
        self.loader = tensor_loader
        self.model_config = model_config
        self._cpu_device = "cpu"
        self._gpu_device = str(self.config.routed_expert_device)
        self._is_cuda_target = self._gpu_device.startswith("cuda") and torch.cuda.is_available()

        self._routed_expert_layers = self._resolve_routed_expert_layers()
        self._cpu_cache_capacity = self._filter_cache_capacity(self.config.max_cpu_cache_capacity)
        self._gpu_cache_capacity = self._filter_cache_capacity(self.config.max_gpu_cache_capacity)

        self._global_step = 0
        self._expert_elems: int = 0
        self._expert_layout: dict[str, list[int]] = {}

        self._gpu_layer_slot_map: dict[int, tuple[int, int]] = {}
        self._cpu_layer_slot_map: dict[int, tuple[int, int]] = {}

        self._gpu_slots: list[_SlotInfo] = []
        self._cpu_slots: list[_SlotInfo] = []

        self._requests: dict[tuple[int, int], _ExpertRequest] = {}
        self._waiting_cpu: deque[tuple[int, int]] = deque()
        self._waiting_gpu: deque[tuple[int, int]] = deque()
        self._loading_cpu: set[tuple[int, int]] = set()
        self._loading_gpu: set[tuple[int, int]] = set()
        self._ready: set[tuple[int, int]] = set()
        self._compute_pending: set[tuple[int, int]] = set()

        self._expert_elems, self._expert_layout, self._expert_dtype = self._probe_expert()
        self._num_cpu_experts, self._cpu_layer_slot_map = self._build_layer_slot_mapping(
            self._cpu_cache_capacity
        )
        self._num_gpu_experts, self._gpu_layer_slot_map = self._build_layer_slot_mapping(
            self._gpu_cache_capacity
        )

        if self._num_cpu_experts <= 0:
            raise ValueError("AsyncExpertCacheManager requires at least one CPU cache slot.")
        if self._num_gpu_experts <= 0:
            raise ValueError("AsyncExpertCacheManager requires at least one GPU cache slot.")

        self._cpu_buffer = torch.empty(
            self._num_cpu_experts,
            self._expert_elems,
            dtype=self._expert_dtype,
            device=self._cpu_device,
            pin_memory=self._is_cuda_target,
        )
        self._gpu_buffer = torch.empty(
            self._num_gpu_experts,
            self._expert_elems,
            dtype=self._expert_dtype,
            device=self._gpu_device,
        )

        for layer_idx, (start, end) in self._cpu_layer_slot_map.items():
            for _ in range(start, end):
                self._cpu_slots.append(_SlotInfo(layer_idx=layer_idx))

        for layer_idx, (start, end) in self._gpu_layer_slot_map.items():
            for _ in range(start, end):
                self._gpu_slots.append(_SlotInfo(layer_idx=layer_idx))

        self._disk_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="moe-disk")
        self._h2d_stream = torch.cuda.Stream() if self._is_cuda_target else None

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._shutdown = False
        self._scheduler = threading.Thread(
            target=self._scheduler_loop,
            name="moe-cache-scheduler",
            daemon=True,
        )
        self._scheduler.start()

    # ==================================================================
    # block 可见接口
    # ==================================================================

    def step(self) -> None:
        with self._cv:
            self._global_step += 1
            self._cv.notify_all()

    def request_expert(self, layer_idx: int, expert_idx: int) -> None:
        """登记 expert 需求并唤醒后台调度器。"""

        key = (layer_idx, int(expert_idx))
        with self._cv:
            if key in self._requests:
                self._raise_if_error_locked(key)
                self._cv.notify_all()
                return

            gpu_slot_idx = self._find_expert_in_slots(
                self._gpu_slots,
                self._gpu_layer_slot_map,
                key[0],
                key[1],
            )
            if gpu_slot_idx is not None:
                slot = self._gpu_slots[gpu_slot_idx]
                slot.status = "protected"
                self._requests[key] = _ExpertRequest(
                    layer_idx=key[0],
                    expert_idx=key[1],
                    stage="ready",
                    gpu_slot_idx=gpu_slot_idx,
                    gpu_generation=slot.generation,
                )
                self._ready.add(key)
                self._cv.notify_all()
                return

            cpu_slot_idx = self._find_expert_in_slots(
                self._cpu_slots,
                self._cpu_layer_slot_map,
                key[0],
                key[1],
            )
            if cpu_slot_idx is not None:
                slot = self._cpu_slots[cpu_slot_idx]
                slot.status = "protected"
                self._requests[key] = _ExpertRequest(
                    layer_idx=key[0],
                    expert_idx=key[1],
                    stage="waiting_gpu",
                    cpu_slot_idx=cpu_slot_idx,
                    cpu_generation=slot.generation,
                )
                self._waiting_gpu.append(key)
                self._cv.notify_all()
                return

            self._requests[key] = _ExpertRequest(
                layer_idx=key[0],
                expert_idx=key[1],
                stage="waiting_cpu",
            )
            self._waiting_cpu.append(key)
            self._cv.notify_all()

    def is_expert_ready(self, layer_idx: int, expert_idx: int) -> bool:
        """只查询 ready 状态，不推进调度。"""

        key = (layer_idx, int(expert_idx))
        with self._lock:
            self._raise_if_error_locked(key)
            return key in self._ready

    def wait_any_ready(self, layer_idx: int, expert_ids: list[int]) -> int:
        """阻塞直到 expert_ids 中任意一个 expert ready。"""

        keys = [(layer_idx, int(expert_idx)) for expert_idx in expert_ids]
        with self._cv:
            for key in keys:
                if key not in self._requests:
                    self._enqueue_request_locked(key)
            self._cv.notify_all()

            while True:
                for key in keys:
                    self._raise_if_error_locked(key)
                    if key in self._ready:
                        return key[1]
                self._cv.wait(timeout=0.001)

    def wait_until_any_ready(self, layer_idx: int, expert_ids: list[int]) -> int:
        """兼容旧命名。"""

        return self.wait_any_ready(layer_idx, expert_ids)

    def get_ready_expert(self, layer_idx: int, expert_idx: int) -> dict[str, torch.Tensor]:
        """返回 protected GPU slot 中的专家权重 view。"""

        key = (layer_idx, int(expert_idx))
        with self._lock:
            self._raise_if_error_locked(key)
            req = self._requests.get(key)
            if req is None or key not in self._ready or req.gpu_slot_idx is None:
                raise RuntimeError(f"Expert is not ready: layer={layer_idx}, expert={expert_idx}")

            slot = self._gpu_slots[req.gpu_slot_idx]
            if (
                slot.status != "protected"
                or slot.layer_idx != key[0]
                or slot.expert_idx != key[1]
                or slot.generation != req.gpu_generation
            ):
                raise RuntimeError(
                    f"Ready expert slot state is invalid: layer={layer_idx}, expert={expert_idx}"
                )
            return self._view_gpu_slot(req.gpu_slot_idx)

    def release_expert(self, layer_idx: int, expert_idx: int) -> None:
        """记录计算完成事件；真正释放由后台调度器完成。"""

        key = (layer_idx, int(expert_idx))
        with self._cv:
            req = self._requests.get(key)
            if req is None:
                return
            self._raise_if_error_locked(key)
            if key not in self._ready or req.gpu_slot_idx is None:
                raise RuntimeError(f"Cannot release non-ready expert: layer={layer_idx}, expert={expert_idx}")

            self._ready.discard(key)
            req.stage = "compute_pending"
            if self._is_cuda_target:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                req.compute_event = event
            else:
                req.compute_event = None
            self._compute_pending.add(key)
            self._cv.notify_all()

    def get_expert(self, layer_idx: int, expert_idx: int) -> dict[str, torch.Tensor]:
        """兼容旧 block 的同步接口。

        旧接口无法在计算结束后显式 release，因此这里不适合长期用于异步模式。
        当前仅作为过渡兼容路径保留。
        """

        expert_idx = int(expert_idx)
        self.request_expert(layer_idx, expert_idx)
        if not self.is_expert_ready(layer_idx, expert_idx):
            self.wait_any_ready(layer_idx, [expert_idx])
        return self.get_ready_expert(layer_idx, expert_idx)

    # ==================================================================
    # 后台调度器
    # ==================================================================

    def _scheduler_loop(self) -> None:
        while True:
            with self._cv:
                if self._shutdown:
                    return
                progressed = self._pump_locked()
                if not progressed:
                    self._cv.wait(timeout=0.0005)

    def _pump_locked(self) -> bool:
        """非阻塞调度 tick。调用方必须持有 self._cv。"""

        progressed = False
        progressed |= self._complete_finished_compute_locked()
        progressed |= self._complete_finished_h2d_locked()
        progressed |= self._complete_finished_disk_reads_locked()
        progressed |= self._schedule_h2d_locked()
        progressed |= self._schedule_disk_reads_locked()
        if progressed:
            self._cv.notify_all()
        return progressed

    def _complete_finished_compute_locked(self) -> bool:
        progressed = False
        for key in list(self._compute_pending):
            req = self._requests.get(key)
            if req is None:
                self._compute_pending.discard(key)
                progressed = True
                continue
            if req.compute_event is not None and not req.compute_event.query():
                continue
            if req.gpu_slot_idx is not None:
                slot = self._gpu_slots[req.gpu_slot_idx]
                if slot.generation == req.gpu_generation and slot.status == "protected":
                    slot.status = "used"
                    slot.last_used_step = self._global_step
            self._compute_pending.discard(key)
            self._requests.pop(key, None)
            progressed = True
        return progressed

    def _complete_finished_h2d_locked(self) -> bool:
        progressed = False
        for key in list(self._loading_gpu):
            req = self._requests.get(key)
            if req is None:
                self._loading_gpu.discard(key)
                progressed = True
                continue
            if req.h2d_event is not None and not req.h2d_event.query():
                continue
            if req.gpu_slot_idx is None:
                self._loading_gpu.discard(key)
                progressed = True
                continue

            gpu_slot = self._gpu_slots[req.gpu_slot_idx]
            if gpu_slot.generation != req.gpu_generation:
                req.error = RuntimeError(f"Stale GPU slot generation for expert {key}")
                self._loading_gpu.discard(key)
                progressed = True
                continue

            gpu_slot.status = "protected"
            req.stage = "ready"
            self._loading_gpu.discard(key)
            self._ready.add(key)

            if req.cpu_slot_idx is not None:
                cpu_slot = self._cpu_slots[req.cpu_slot_idx]
                if cpu_slot.generation == req.cpu_generation and cpu_slot.status == "protected":
                    cpu_slot.status = "used"
                    cpu_slot.last_used_step = self._global_step
            progressed = True
        return progressed

    def _complete_finished_disk_reads_locked(self) -> bool:
        progressed = False
        for key in list(self._loading_cpu):
            req = self._requests.get(key)
            if req is None:
                self._loading_cpu.discard(key)
                progressed = True
                continue
            if req.disk_future is None or not req.disk_future.done():
                continue

            try:
                req.disk_future.result()
            except BaseException as exc:
                req.error = exc
                self._loading_cpu.discard(key)
                progressed = True
                continue

            if req.cpu_slot_idx is None:
                self._loading_cpu.discard(key)
                progressed = True
                continue

            cpu_slot = self._cpu_slots[req.cpu_slot_idx]
            if cpu_slot.generation != req.cpu_generation:
                req.error = RuntimeError(f"Stale CPU slot generation for expert {key}")
                self._loading_cpu.discard(key)
                progressed = True
                continue

            cpu_slot.status = "protected"
            req.stage = "waiting_gpu"
            self._loading_cpu.discard(key)
            self._waiting_gpu.append(key)
            progressed = True
        return progressed

    def _schedule_h2d_locked(self) -> bool:
        progressed = False
        while self._waiting_gpu:
            key = self._waiting_gpu[0]
            req = self._requests.get(key)
            if req is None or req.stage != "waiting_gpu":
                self._waiting_gpu.popleft()
                progressed = True
                continue
            if req.cpu_slot_idx is None:
                self._waiting_gpu.popleft()
                req.stage = "waiting_cpu"
                self._waiting_cpu.append(key)
                progressed = True
                continue

            gpu_slot_idx = self._pick_slot(self._gpu_slots, self._gpu_layer_slot_map, req.layer_idx)
            if gpu_slot_idx is None:
                return progressed

            self._waiting_gpu.popleft()
            gpu_slot = self._gpu_slots[gpu_slot_idx]
            gpu_slot.layer_idx = req.layer_idx
            gpu_slot.expert_idx = req.expert_idx
            gpu_slot.status = "loading"
            gpu_slot.generation += 1

            req.gpu_slot_idx = gpu_slot_idx
            req.gpu_generation = gpu_slot.generation
            req.stage = "loading_gpu"

            if self._is_cuda_target:
                assert self._h2d_stream is not None
                with torch.cuda.stream(self._h2d_stream):
                    self._gpu_buffer[gpu_slot_idx, :].copy_(
                        self._cpu_buffer[req.cpu_slot_idx, :],
                        non_blocking=True,
                    )
                    event = torch.cuda.Event()
                    event.record(self._h2d_stream)
                req.h2d_event = event
            else:
                self._gpu_buffer[gpu_slot_idx, :].copy_(self._cpu_buffer[req.cpu_slot_idx, :])
                req.h2d_event = None

            self._loading_gpu.add(key)
            progressed = True
        return progressed

    def _schedule_disk_reads_locked(self) -> bool:
        progressed = False
        while self._waiting_cpu and len(self._loading_cpu) < 2:
            key = self._waiting_cpu[0]
            req = self._requests.get(key)
            if req is None or req.stage != "waiting_cpu":
                self._waiting_cpu.popleft()
                progressed = True
                continue

            cpu_slot_idx = self._pick_slot(self._cpu_slots, self._cpu_layer_slot_map, req.layer_idx)
            if cpu_slot_idx is None:
                return progressed

            self._waiting_cpu.popleft()
            cpu_slot = self._cpu_slots[cpu_slot_idx]
            cpu_slot.layer_idx = req.layer_idx
            cpu_slot.expert_idx = req.expert_idx
            cpu_slot.status = "loading"
            cpu_slot.generation += 1

            req.cpu_slot_idx = cpu_slot_idx
            req.cpu_generation = cpu_slot.generation
            req.stage = "loading_cpu"

            spec = self.adapter.routed_expert_weight_name(req.layer_idx, req.expert_idx)
            req.disk_future = self._disk_pool.submit(
                self.loader.load_tensors_to_buffer,
                tensor_names=list(spec.values()),
                buffer=self._cpu_buffer,
                slot_idx=cpu_slot_idx,
            )
            self._loading_cpu.add(key)
            progressed = True
        return progressed

    # ==================================================================
    # 请求与 slot 辅助
    # ==================================================================

    def _enqueue_request_locked(self, key: tuple[int, int]) -> None:
        if key in self._requests:
            return

        gpu_slot_idx = self._find_expert_in_slots(
            self._gpu_slots,
            self._gpu_layer_slot_map,
            key[0],
            key[1],
        )
        if gpu_slot_idx is not None:
            slot = self._gpu_slots[gpu_slot_idx]
            slot.status = "protected"
            self._requests[key] = _ExpertRequest(
                layer_idx=key[0],
                expert_idx=key[1],
                stage="ready",
                gpu_slot_idx=gpu_slot_idx,
                gpu_generation=slot.generation,
            )
            self._ready.add(key)
            return

        cpu_slot_idx = self._find_expert_in_slots(
            self._cpu_slots,
            self._cpu_layer_slot_map,
            key[0],
            key[1],
        )
        if cpu_slot_idx is not None:
            slot = self._cpu_slots[cpu_slot_idx]
            slot.status = "protected"
            self._requests[key] = _ExpertRequest(
                layer_idx=key[0],
                expert_idx=key[1],
                stage="waiting_gpu",
                cpu_slot_idx=cpu_slot_idx,
                cpu_generation=slot.generation,
            )
            self._waiting_gpu.append(key)
            return

        self._requests[key] = _ExpertRequest(layer_idx=key[0], expert_idx=key[1], stage="waiting_cpu")
        self._waiting_cpu.append(key)

    def _raise_if_error_locked(self, key: tuple[int, int]) -> None:
        req = self._requests.get(key)
        if req is not None and req.error is not None:
            raise RuntimeError(f"Expert request failed for {key}") from req.error

    @staticmethod
    def _find_expert_in_slots(
        slots: list[_SlotInfo],
        layer_slot_map: dict[int, tuple[int, int]],
        layer_idx: int,
        expert_idx: int,
    ) -> int | None:
        slot_range = layer_slot_map.get(layer_idx)
        if slot_range is None:
            return None

        for slot_idx in range(slot_range[0], slot_range[1]):
            slot = slots[slot_idx]
            if (
                slot.layer_idx == layer_idx
                and slot.expert_idx == expert_idx
                and slot.status in ("used", "protected")
            ):
                return slot_idx
        return None

    def _view_gpu_slot(self, gpu_slot_idx: int) -> dict[str, torch.Tensor]:
        offset = 0
        result: dict[str, torch.Tensor] = {}
        for role, shape in self._expert_layout.items():
            numel = prod(shape)
            result[role] = self._gpu_buffer[gpu_slot_idx, offset : offset + numel].view(
                torch.Size(shape)
            )
            offset += numel
        return result

    @staticmethod
    def _pick_slot(
        slots: list[_SlotInfo],
        layer_slot_map: dict[int, tuple[int, int]],
        layer_idx: int,
    ) -> int | None:
        slot_range = layer_slot_map.get(layer_idx)
        if slot_range is None:
            return None

        free_idx: int | None = None
        used_idx: int | None = None
        used_step = 2**63

        for slot_idx in range(slot_range[0], slot_range[1]):
            slot = slots[slot_idx]
            if slot.status == "free":
                free_idx = slot_idx
                break
            if slot.status == "used" and slot.last_used_step < used_step:
                used_step = slot.last_used_step
                used_idx = slot_idx

        return free_idx if free_idx is not None else used_idx

    # ==================================================================
    # 初始化辅助
    # ==================================================================

    def _resolve_routed_expert_layers(self) -> list[int]:
        if self.model_config is not None and hasattr(self.adapter, "routed_expert_layers"):
            return list(self.adapter.routed_expert_layers(self.model_config))
        raise RuntimeError("AsyncExpertCacheManager 需要 model_config 和 adapter.routed_expert_layers() 来确定专家层。")

    def _filter_cache_capacity(self, cap: dict[int, int] | None) -> dict[int, int]:
        if not cap:
            return {}

        routed_layers = set(self._routed_expert_layers)
        return {
            layer_idx: count
            for layer_idx, count in cap.items()
            if layer_idx in routed_layers and count > 0
        }

    def _probe_expert(self) -> tuple[int, dict[str, list[int]], torch.dtype]:
        spec = None
        for layer_idx in self._routed_expert_layers:
            candidate = self.adapter.routed_expert_weight_name(layer_idx, 0)
            if all(self.loader.has_tensor(tensor_name=name) for name in candidate.values()):
                spec = candidate
                break

        if spec is None:
            raise KeyError("No routed expert tensor found in manifest for expert probing.")

        expert_elems = 0
        layout: dict[str, list[int]] = {}
        torch_dtype: torch.dtype | None = None

        for role_name, tensor_name in spec.items():
            shape, elems, dtype = self.loader.get_tensor_info(tensor_name=tensor_name)
            expert_elems += elems
            layout[role_name] = shape
            torch_dtype = dtype

        if torch_dtype is None:
            raise RuntimeError("Failed to probe expert dtype.")

        return expert_elems, layout, torch_dtype

    @staticmethod
    def _build_layer_slot_mapping(cap: dict[int, int]) -> tuple[int, dict[int, tuple[int, int]]]:
        layer_slot_map: dict[int, tuple[int, int]] = {}
        offset = 0
        for layer_idx in sorted(cap):
            count = cap[layer_idx]
            layer_slot_map[layer_idx] = (offset, offset + count)
            offset += count
        return offset, layer_slot_map

    def close(self) -> None:
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        self._scheduler.join(timeout=1.0)
        self._disk_pool.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


ExpertCacheManager = AsyncExpertCacheManager
