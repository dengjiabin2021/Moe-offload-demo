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


# CacheManager 内部统一用 (layer_idx, expert_idx) 作为一个 expert 的唯一键。
ExpertKey = tuple[int, int]

# 磁盘读取线程数。当前一个请求会读取一个完整 expert，2 个 worker 可以让磁盘回源和 H2D/计算重叠。
_MAX_DISK_WORKERS = 2

# 后台调度线程没有可推进任务时的短等待时间，单位秒。
_SCHEDULER_WAIT_SECONDS = 0.0005

# block 阻塞等待任意 expert ready 时的轮询等待时间，单位秒。
_WAIT_READY_SECONDS = 0.001


@dataclass
class _SlotInfo:
    """缓存 slot 的元信息。

    CPU/GPU buffer 都是二维连续张量，每一行是一个完整 expert。
    _SlotInfo 与 buffer 的行一一对应，用来描述这一行当前存的是哪个 expert、是否可读、是否可覆盖。
    """

    # slot 归属的 transformer 层。slot 分配按层隔离，同一层只使用自己的 slot 范围。
    layer_idx: int | None = None

    # 当前 slot 中存放的 routed expert 编号；free slot 没有有效 expert。
    expert_idx: int | None = None

    # 四状态：free 空槽；used 可读可淘汰；protected 可读不可淘汰；loading 正在写入不可读不可写。
    status: str = "free"

    # LRU 依据。只有 used slot 可以被覆盖，last_used_step 越小越早被覆盖。
    last_used_step: int = 0

    # slot 每次被重新写入都会递增 generation，用来识别异步事件完成时是否已经过期。
    generation: int = 0


@dataclass
class _ExpertRequest:
    """一次 expert 请求在 CacheManager 内部的调度状态。

    block 只知道请求某个 expert、查询是否 ready、ready 后取权重并 release。
    这里保存的 stage/slot/event/future 是 CacheManager 自己推进 CPU/GPU/磁盘状态机所需的信息。
    """

    # 请求对应的模型层和 expert 编号。
    layer_idx: int
    expert_idx: int

    # 调度阶段：waiting_cpu、loading_cpu、waiting_gpu、loading_gpu、ready、compute_pending。
    stage: str

    # CPU/GPU slot 下标，以及发起异步任务时对应的 generation。
    cpu_slot_idx: int | None = None
    gpu_slot_idx: int | None = None
    cpu_generation: int | None = None
    gpu_generation: int | None = None

    # 磁盘到 CPU 的后台线程任务。
    disk_future: Future | None = None

    # CPU 到 GPU 的 CUDA stream 事件；非 CUDA 目标设备时为 None。
    h2d_event: torch.cuda.Event | None = None

    # block release 后记录的计算完成事件，用来保护正在被 GPU 计算使用的 slot。
    compute_event: torch.cuda.Event | None = None

    # 异步任务里的异常会挂到 request 上，由 block 下一次查询时抛出。
    error: BaseException | None = None


class asyncExpertCacheManager:
    """异步 routed expert 缓存管理器。

    设计边界：
    - block 只调用 request_expert / is_expert_ready / wait_any_ready / get_ready_expert / release_expert。
    - CacheManager 独立维护 CPU/GPU slot 四状态，并由后台调度线程推进磁盘读取和 H2D 拷贝。
    - GPU 上已经命中的 expert 会直接 protected，然后交给 block 立刻计算。
    - GPU 上未命中的 expert 会排队加载；加载完成后状态变为 protected，保证马上计算前不会被 LRU 覆盖。
    """

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        adapter: Any,
        tensor_loader: ManifestTensorLoader,
        model_config: Any | None = None,
    ) -> None:
        """初始化缓存布局、异步调度队列和后台调度线程。"""

        # 保存运行期依赖。adapter 负责专家权重命名，loader 负责按 manifest 读权重。
        self.config = runtime_config
        self.adapter = adapter
        self.loader = tensor_loader
        self.model_config = model_config

        # CPU buffer 固定放 CPU；GPU buffer 使用配置里的 routed_expert_device。
        self._cpu_device = "cpu"
        self._gpu_device = str(self.config.routed_expert_device)
        self._is_cuda_target = self._gpu_device.startswith("cuda") and torch.cuda.is_available()

        # 只为真实 routed expert 层建立缓存，避免 DeepSeek 这类前几层 dense 的模型浪费 slot。
        self._routed_expert_layers = self._resolve_routed_expert_layers()
        self._cpu_cache_capacity = self._filter_cache_capacity(self.config.max_cpu_cache_capacity)
        self._gpu_cache_capacity = self._filter_cache_capacity(self.config.max_gpu_cache_capacity)
        self._validate_cache_capacity()

        # 全局 step 由 ModelRuntime 的 forward hook 推进，用于 LRU。
        self._global_step = 0

        # 单个 expert 展平后的元素数，以及从展平 buffer 还原 gate/up/down 的 shape 布局。
        self._expert_elems: int = 0
        self._expert_layout: dict[str, list[int]] = {}

        # layer_idx -> [start, end) 的 slot 范围，CPU/GPU 各一份。
        self._gpu_layer_slot_map: dict[int, tuple[int, int]] = {}
        self._cpu_layer_slot_map: dict[int, tuple[int, int]] = {}

        # 每个 buffer row 对应一条 slot 元信息。
        self._gpu_slots: list[_SlotInfo] = []
        self._cpu_slots: list[_SlotInfo] = []

        # request 表和调度队列。所有字段只在 self._cv 锁内读写。
        self._requests: dict[ExpertKey, _ExpertRequest] = {}
        self._waiting_cpu: deque[ExpertKey] = deque()
        self._waiting_gpu: deque[ExpertKey] = deque()
        self._loading_cpu: set[ExpertKey] = set()
        self._loading_gpu: set[ExpertKey] = set()
        self._ready: set[ExpertKey] = set()
        self._compute_pending: set[ExpertKey] = set()

        # 根据第一个可用 routed expert 探测每个 expert 的展平大小、切分布局和 dtype。
        self._expert_elems, self._expert_layout, self._expert_dtype = self._probe_expert()

        # 根据每层容量配置，把各层 slot 映射到连续 buffer 的行范围。
        self._num_cpu_experts, self._cpu_layer_slot_map = self._build_layer_slot_mapping(self._cpu_cache_capacity)
        self._num_gpu_experts, self._gpu_layer_slot_map = self._build_layer_slot_mapping(self._gpu_cache_capacity)

        # CPU buffer 使用 pinned memory，可以让 CUDA H2D copy_ 真正异步。
        self._cpu_buffer = torch.empty(
            self._num_cpu_experts,
            self._expert_elems,
            dtype=self._expert_dtype,
            device=self._cpu_device,
            pin_memory=self._is_cuda_target
        )

        # GPU buffer 是 block 计算时直接读取的专家权重池。
        self._gpu_buffer = torch.empty(
            self._num_gpu_experts,
            self._expert_elems,
            dtype=self._expert_dtype,
            device=self._gpu_device,
        )

        # 初始化 CPU slot 元信息。slot 的 layer_idx 固定，expert_idx 等待实际加载时填写。
        for layer_idx, (start, end) in self._cpu_layer_slot_map.items():
            for _ in range(start, end):
                self._cpu_slots.append(_SlotInfo(layer_idx=layer_idx))

        # 初始化 GPU slot 元信息。
        for layer_idx, (start, end) in self._gpu_layer_slot_map.items():
            for _ in range(start, end):
                self._gpu_slots.append(_SlotInfo(layer_idx=layer_idx))

        # 磁盘读取放到线程池里执行，使 mmap 读取和 GPU 计算/H2D 能重叠。
        self._disk_pool = ThreadPoolExecutor(
            max_workers=_MAX_DISK_WORKERS,
            thread_name_prefix="moe-disk",
        )

        # H2D 使用独立 CUDA stream，避免阻塞当前计算 stream。
        self._h2d_stream = torch.cuda.Stream() if self._is_cuda_target else None

        # 同一把锁保护 request 表、slot 状态和所有调度队列。
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

        # 后台调度线程只做非阻塞推进；真正等待只发生在 block 的 wait_any_ready。
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
        """推进 LRU step。

        ModelRuntime 在每次 forward 结束后调用它。
        step 越大，越久没有被访问的 used slot 越容易被 _pick_slot 选中覆盖。
        """

        with self._cv:
            self._global_step += 1
            self._cv.notify_all()

    def request_expert(self, layer_idx: int, expert_idx: int) -> None:
        """提交一个 expert 需求。

        这个函数只负责把需求放入合适的状态或队列：
        - GPU 命中：used -> protected，并加入 ready。
        - CPU 命中：CPU used -> protected，并加入 waiting_gpu。
        - CPU/GPU 都未命中：加入 waiting_cpu，等待磁盘回源。
        """

        key = (layer_idx, expert_idx)
        with self._cv:
            self._raise_if_error(key)

            # 已经有同一个 expert 的请求时，不重复入队；后台调度器会继续推进原请求。
            if key not in self._requests:
                self._enqueue_request(key)

            self._cv.notify_all()

    def is_expert_ready(self, layer_idx: int, expert_idx: int) -> bool:
        """查询 expert 是否已经可在 GPU 上计算。

        查询不是纯读操作：如果 expert 已经在 GPU used slot 中，本函数会立刻把它保护为
        protected 并标记 ready，避免 block 准备计算时被后台 LRU 覆盖。
        """

        key = (layer_idx, expert_idx)
        with self._cv:
            self._raise_if_error(key)

            # 已经 ready 的 expert 可以直接交给 block 计算。
            if key in self._ready:
                return True

            # request 存在但还未 ready，说明正在等磁盘、H2D 或上一轮计算完成。
            if key in self._requests:
                return False

            # 没有 request 时，如果 GPU 里已有 used slot，可以直接保护并返回 ready。
            gpu_slot_idx = self._find_expert_in_slots(
                self._gpu_slots,
                self._gpu_layer_slot_map,
                key[0],
                key[1],
            )
            if gpu_slot_idx is None:
                return False

            gpu_slot = self._gpu_slots[gpu_slot_idx]
            # ready expert 必须 protected，避免 block 计算前或计算中被 LRU 覆盖。
            gpu_slot.status = "protected"
            self._requests[key] = _ExpertRequest(
                layer_idx=key[0],
                expert_idx=key[1],
                stage="ready",
                gpu_slot_idx=gpu_slot_idx,
                gpu_generation=gpu_slot.generation,
            )
            self._ready.add(key)

            self._cv.notify_all()
            return True

    def wait_any_ready(self, layer_idx: int, expert_ids: list[int]) -> int:
        """阻塞等待一组 expert 中任意一个变为 ready。

        block 只有在当前没有 ready expert、但仍有 pending expert 时才调用它。
        所有 pending expert 都必须已经通过 request_expert 进入调度状态机。
        本函数不修复丢失的 request；如果状态机不一致，直接报错，方便定位问题。
        """

        if not expert_ids:
            raise ValueError("wait_any_ready requires at least one expert id.")

        keys = [(layer_idx, expert_idx) for expert_idx in expert_ids]
        with self._cv:
            # pending expert 必须已经由 block 调用 request_expert 提交过。
            for key in keys:
                if key not in self._ready and key not in self._requests:
                    raise RuntimeError(f"Pending expert has not been requested: {key}")

            while True:
                for key in keys:
                    self._raise_if_error(key)

                    # 任意一个 expert ready 后立即返回，由 block 获取权重并提交计算。
                    if key in self._ready:
                        return key[1]

                    if key not in self._requests:
                        raise RuntimeError(f"Pending expert request disappeared before ready: {key}")

                self._cv.wait(timeout=_WAIT_READY_SECONDS)

    def get_ready_expert(self, layer_idx: int, expert_idx: int) -> dict[str, torch.Tensor]:
        """返回一个 ready expert 的 GPU 权重 view。

        只有处于 protected 状态的 GPU slot 才允许被读取。
        返回值是对 _gpu_buffer 对应行的 view，不会产生额外权重拷贝。
        """

        key = (layer_idx, expert_idx)
        with self._lock:
            self._raise_if_error(key)
            req = self._requests.get(key)
            if req is None or key not in self._ready or req.gpu_slot_idx is None:
                raise RuntimeError(f"Expert is not ready: layer={layer_idx}, expert={expert_idx}")

            slot = self._gpu_slots[req.gpu_slot_idx]

            # generation 校验可以避免异步事件落后时读到已经被覆盖的 slot。
            if (slot.status != "protected" or slot.layer_idx != key[0] or slot.expert_idx != key[1] or slot.generation != req.gpu_generation):
                raise RuntimeError(f"Ready expert slot state is invalid: layer={layer_idx}, expert={expert_idx}")

            return self._view_gpu_slot(req.gpu_slot_idx)

    def release_expert(self, layer_idx: int, expert_idx: int) -> None:
        """通知 CacheManager：block 已经把该 expert 的计算提交到当前 stream。

        release 不会立即把 protected 改回 used。
        CUDA 目标设备上会记录当前 stream 的 event，后台调度器等 event 完成后才允许该 slot 被 LRU 覆盖。
        """

        key = (layer_idx, expert_idx)
        with self._cv:
            req = self._requests.get(key)
            if req is None:
                raise RuntimeError(f"Cannot release unknown expert: layer={layer_idx}, expert={expert_idx}")

            self._raise_if_error(key)

            if key not in self._ready or req.gpu_slot_idx is None:
                raise RuntimeError(f"Cannot release non-ready expert: layer={layer_idx}, expert={expert_idx}")

            # 从 ready 集合移出后，block 不能再取这个 expert；slot 仍 protected，直到计算 event 完成。
            self._ready.discard(key)
            req.stage = "compute_pending"

            if self._is_cuda_target:
                # 当前 stream 上前面的 expert matmul/index_add_ 都完成后，这个 event 才会变为 ready。
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                req.compute_event = event
            else:
                # CPU 目标设备没有异步 GPU 计算，release 返回时计算已经完成。
                req.compute_event = None

            self._compute_pending.add(key)
            self._cv.notify_all()

    # ==================================================================
    # 后台调度器
    # ==================================================================

    def _scheduler_loop(self) -> None:
        """后台调度线程主循环。

        该循环不等待某一个具体 expert，而是持续非阻塞地推进：
        计算完成检查、H2D 完成检查、磁盘读取完成检查、新 H2D 任务、新磁盘任务。
        """

        while True:
            with self._cv:
                if self._shutdown:
                    return

                progressed = self._pump()
                if not progressed:
                    self._cv.wait(timeout=_SCHEDULER_WAIT_SECONDS)

    def _pump(self) -> bool:
        """执行一次非阻塞调度 tick。

        调用方必须持有 self._cv。
        返回 True 表示至少推进了一个状态，调用方可以唤醒等待 ready 的 block。
        """

        progressed = False

        # 先回收计算完成的 GPU slot，让后续 H2D 有机会拿到可覆盖 slot。
        progressed |= self._complete_finished_compute_locked()

        # 再检查 H2D 完成，把 loading GPU slot 转为 protected ready。
        progressed |= self._complete_finished_h2d_locked()

        # 再检查磁盘读取完成，把 loading CPU slot 转为 protected，并放入 waiting_gpu。
        progressed |= self._complete_finished_disk_reads_locked()

        # 优先调度 CPU -> GPU，让 CPU hit 或刚读完磁盘的 expert 尽快可算。
        progressed |= self._schedule_h2d_locked()

        # 最后补充磁盘回源任务，持续填充 CPU cache。
        progressed |= self._schedule_disk_reads_locked()

        if progressed:
            self._cv.notify_all()
        return progressed

    def _complete_finished_compute_locked(self) -> bool:
        """检查哪些 GPU 计算已经完成，并释放对应 protected slot。

        状态流转：protected + compute_pending -> used。
        compute_pending 中的每个 key 都必须有完整 request 和 GPU slot 信息。
        """

        progressed = False
        for key in list(self._compute_pending):
            req = self._requests.get(key)
            if req is None:
                raise RuntimeError(f"Compute pending expert has no request: {key}")

            if req.stage != "compute_pending":
                raise RuntimeError(f"Compute pending expert has invalid stage: {key}, stage={req.stage}")

            if req.gpu_slot_idx is None:
                raise RuntimeError(f"Compute pending expert has no GPU slot: {key}")

            # CUDA event 未完成时，slot 仍然 protected，不能被覆盖。
            if req.compute_event is not None and not req.compute_event.query():
                continue

            slot = self._gpu_slots[req.gpu_slot_idx]

            # 计算完成后，slot 必须仍然是同一个 protected expert；否则说明 slot 被错误覆盖。
            if (
                slot.generation != req.gpu_generation
                or slot.status != "protected"
                or slot.layer_idx != req.layer_idx
                or slot.expert_idx != req.expert_idx
            ):
                raise RuntimeError(f"Compute pending expert slot state is invalid: {key}")

            slot.status = "used"
            slot.last_used_step = self._global_step

            self._compute_pending.discard(key)
            self._requests.pop(key, None)
            progressed = True

        return progressed

    def _complete_finished_h2d_locked(self) -> bool:
        """检查 CPU -> GPU 拷贝是否完成。

        状态流转：
        - GPU: loading -> protected
        - CPU: protected -> used
        loading_gpu 中的 request 必须同时持有源 CPU slot 和目标 GPU slot。
        """

        progressed = False
        for key in list(self._loading_gpu):
            req = self._requests.get(key)
            if req is None:
                raise RuntimeError(f"Loading GPU expert has no request: {key}")

            if req.stage != "loading_gpu":
                raise RuntimeError(f"Loading GPU expert has invalid stage: {key}, stage={req.stage}")

            # CUDA H2D event 未完成时，GPU slot 仍处于 loading，block 不能读取。
            if req.h2d_event is not None and not req.h2d_event.query():
                continue

            if req.gpu_slot_idx is None:
                raise RuntimeError(f"Loading GPU expert has no GPU slot: {key}")

            if req.cpu_slot_idx is None:
                raise RuntimeError(f"Loading GPU expert has no CPU source slot: {key}")

            gpu_slot = self._gpu_slots[req.gpu_slot_idx]
            if (
                gpu_slot.generation != req.gpu_generation
                or gpu_slot.status != "loading"
                or gpu_slot.layer_idx != req.layer_idx
                or gpu_slot.expert_idx != req.expert_idx
            ):
                raise RuntimeError(f"Loading GPU expert slot state is invalid: {key}")

            # 先确认 CPU 源 slot 仍然是同一个 protected expert，再提交 GPU ready 状态。
            cpu_slot = self._cpu_slots[req.cpu_slot_idx]
            if (
                cpu_slot.generation != req.cpu_generation
                or cpu_slot.status != "protected"
                or cpu_slot.layer_idx != req.layer_idx
                or cpu_slot.expert_idx != req.expert_idx
            ):
                raise RuntimeError(f"Loading GPU expert CPU source slot state is invalid: {key}")

            # H2D 完成后直接 protected，因为这个 expert 是当前 block 正在等待计算的。
            gpu_slot.status = "protected"
            req.stage = "ready"
            self._loading_gpu.discard(key)
            self._ready.add(key)

            # CPU 源 slot 不再被 H2D 使用，可以重新变为 used，后续允许 LRU 覆盖。
            cpu_slot.status = "used"
            cpu_slot.last_used_step = self._global_step

            progressed = True

        return progressed

    def _complete_finished_disk_reads_locked(self) -> bool:
        """检查磁盘读取线程是否完成。

        状态流转：CPU loading -> protected -> waiting_gpu。
        CPU slot 转 protected 是为了保证随后的 H2D 拷贝不会被 CPU LRU 覆盖。
        loading_cpu 中的 request 必须持有磁盘 future 和目标 CPU slot。
        """

        progressed = False
        for key in list(self._loading_cpu):
            req = self._requests.get(key)
            if req is None:
                raise RuntimeError(f"Loading CPU expert has no request: {key}")

            if req.stage != "loading_cpu":
                raise RuntimeError(f"Loading CPU expert has invalid stage: {key}, stage={req.stage}")

            if req.disk_future is None:
                raise RuntimeError(f"Loading CPU expert has no disk future: {key}")

            # 磁盘线程尚未完成，不能读取或移动 CPU slot。
            if not req.disk_future.done():
                continue

            try:
                req.disk_future.result()
            except BaseException as exc:
                raise RuntimeError(f"Disk read failed for expert {key}") from exc

            if req.cpu_slot_idx is None:
                raise RuntimeError(f"Loading CPU expert has no CPU slot: {key}")

            cpu_slot = self._cpu_slots[req.cpu_slot_idx]
            if (
                cpu_slot.generation != req.cpu_generation
                or cpu_slot.status != "loading"
                or cpu_slot.layer_idx != req.layer_idx
                or cpu_slot.expert_idx != req.expert_idx
            ):
                raise RuntimeError(f"Loading CPU expert slot state is invalid: {key}")

            # 磁盘已经写完整个 expert，CPU slot 可读但暂时不可覆盖，等待 H2D。
            cpu_slot.status = "protected"
            req.stage = "waiting_gpu"
            self._loading_cpu.discard(key)
            self._waiting_gpu.append(key)
            progressed = True

        return progressed

    def _schedule_h2d_locked(self) -> bool:
        """为 waiting_gpu 队列调度 CPU -> GPU 拷贝。

        如果 GPU 没有 free/used slot，函数立即返回，不阻塞。
        后续等 block release 或计算 event 完成后，后台线程会再次尝试。
        waiting_gpu 队列只允许保存 stage 为 waiting_gpu 且 CPU slot 已经 protected 的请求。
        """

        progressed = False
        while self._waiting_gpu:
            key = self._waiting_gpu[0]
            req = self._requests.get(key)

            if req is None:
                raise RuntimeError(f"Waiting GPU expert has no request: {key}")

            if req.stage != "waiting_gpu":
                raise RuntimeError(f"Waiting GPU expert has invalid stage: {key}, stage={req.stage}")

            if req.cpu_slot_idx is None:
                raise RuntimeError(f"Waiting GPU expert has no CPU source slot: {key}")

            cpu_slot = self._cpu_slots[req.cpu_slot_idx]
            if (
                cpu_slot.generation != req.cpu_generation
                or cpu_slot.status != "protected"
                or cpu_slot.layer_idx != req.layer_idx
                or cpu_slot.expert_idx != req.expert_idx
            ):
                raise RuntimeError(f"Waiting GPU expert CPU source slot state is invalid: {key}")

            gpu_slot_idx = self._pick_slot(self._gpu_slots, self._gpu_layer_slot_map, req.layer_idx)
            if gpu_slot_idx is None:
                return progressed

            self._waiting_gpu.popleft()
            gpu_slot = self._gpu_slots[gpu_slot_idx]

            # 选中的 GPU slot 立即进入 loading，防止其他请求读写或覆盖。
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
                # 非 CUDA 目标设备没有独立 H2D stream，copy_ 是同步完成的。
                self._gpu_buffer[gpu_slot_idx, :].copy_(self._cpu_buffer[req.cpu_slot_idx, :])
                req.h2d_event = None

            self._loading_gpu.add(key)
            progressed = True

        return progressed

    def _schedule_disk_reads_locked(self) -> bool:
        """为 waiting_cpu 队列调度磁盘 -> CPU 读取。

        磁盘读取由 ThreadPoolExecutor 执行，因此不会阻塞后台调度线程。
        同时最多允许 _MAX_DISK_WORKERS 个 expert 正在回源，避免无界并发抢占内存带宽。
        waiting_cpu 队列只允许保存 stage 为 waiting_cpu、且尚未占用 CPU slot 的请求。
        """

        progressed = False
        while self._waiting_cpu and len(self._loading_cpu) < _MAX_DISK_WORKERS:
            key = self._waiting_cpu[0]
            req = self._requests.get(key)

            if req is None:
                raise RuntimeError(f"Waiting CPU expert has no request: {key}")

            if req.stage != "waiting_cpu":
                raise RuntimeError(f"Waiting CPU expert has invalid stage: {key}, stage={req.stage}")

            if req.cpu_slot_idx is not None:
                raise RuntimeError(f"Waiting CPU expert already has CPU slot: {key}")

            cpu_slot_idx = self._pick_slot(self._cpu_slots, self._cpu_layer_slot_map, req.layer_idx)
            if cpu_slot_idx is None:
                return progressed

            self._waiting_cpu.popleft()
            cpu_slot = self._cpu_slots[cpu_slot_idx]

            # CPU slot 进入 loading 后，后台磁盘线程拥有写入权。
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

    def _enqueue_request(self, key: ExpertKey) -> None:
        """按 GPU hit、CPU hit、disk miss 的顺序把请求放入调度系统。

        调用方必须持有 self._cv。
        """

        if key in self._requests:
            return

        # GPU used/protected 命中时不搬运，直接 protected 并 ready。
        gpu_slot_idx = self._find_expert_in_slots(
            self._gpu_slots,
            self._gpu_layer_slot_map,
            key[0],
            key[1],
        )

        if gpu_slot_idx is not None:
            gpu_slot = self._gpu_slots[gpu_slot_idx]
            # ready expert 必须 protected，避免 block 计算前或计算中被 LRU 覆盖。
            gpu_slot.status = "protected"
            self._requests[key] = _ExpertRequest(
                layer_idx=key[0],
                expert_idx=key[1],
                stage="ready",
                gpu_slot_idx=gpu_slot_idx,
                gpu_generation=gpu_slot.generation,
            )
            self._ready.add(key)
            return

        # CPU 命中时先保护 CPU slot，随后等待后台调度 H2D。
        cpu_slot_idx = self._find_expert_in_slots(
            self._cpu_slots,
            self._cpu_layer_slot_map,
            key[0],
            key[1],
        )

        if cpu_slot_idx is not None:
            cpu_slot = self._cpu_slots[cpu_slot_idx]
            cpu_slot.status = "protected"
            self._requests[key] = _ExpertRequest(
                layer_idx=key[0],
                expert_idx=key[1],
                stage="waiting_gpu",
                cpu_slot_idx=cpu_slot_idx,
                cpu_generation=cpu_slot.generation,
            )
            self._waiting_gpu.append(key)
            return

        # CPU/GPU 都没有该 expert，只能从磁盘回源。
        self._requests[key] = _ExpertRequest(
            layer_idx=key[0],
            expert_idx=key[1],
            stage="waiting_cpu",
        )
        self._waiting_cpu.append(key)

    # ==================================================================
    # request 和 slot 辅助函数
    # ==================================================================


    @staticmethod
    def _find_expert_in_slots(
        slots: list[_SlotInfo],
        layer_slot_map: dict[int, tuple[int, int]],
        layer_idx: int,
        expert_idx: int,
    ) -> int | None:
        """在某一层的 slot 范围内查找可读 expert。

        loading 不可读，free 没有有效数据。
        used/protected 表示这一行有完整 expert，可作为命中。
        """

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
        """把 GPU buffer 的一行按照 expert layout 切回 gate/up/down 张量 view。"""

        offset = 0
        result: dict[str, torch.Tensor] = {}

        for role, shape in self._expert_layout.items():
            numel = prod(shape)

            # 这里返回的是 view，不复制权重；block 的 linear 会直接读 _gpu_buffer。
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
        """在指定层选择一个可写 slot。

        优先级：
        1. free：空槽，写入成本最低。
        2. used 中 last_used_step 最小的 slot：LRU 覆盖。

        protected 和 loading 都不可覆盖。
        """

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

    def _raise_if_error(self, key: ExpertKey) -> None:
        """如果某个异步任务失败，将后台异常重新抛给 block。"""

        req = self._requests.get(key)
        if req is not None and req.error is not None:
            raise RuntimeError(f"Expert request failed for {key}") from req.error

    # ==================================================================
    # 初始化辅助函数
    # ==================================================================

    def _resolve_routed_expert_layers(self) -> list[int]:
        """从 adapter 获取真实 routed expert 层。"""

        if self.model_config is not None and hasattr(self.adapter, "routed_expert_layers"):
            return list(self.adapter.routed_expert_layers(self.model_config))

        raise RuntimeError("ExpertCacheManager 需要 model_config 和 adapter.routed_expert_layers() 来确定专家层。")

    def _filter_cache_capacity(self, cap: dict[int, int] | None) -> dict[int, int]:
        """过滤缓存容量配置，只保留真实 routed expert 层。"""

        if not cap:
            return {}

        routed_layers = set(self._routed_expert_layers)
        return {
            layer_idx: count
            for layer_idx, count in cap.items()
            if layer_idx in routed_layers and count > 0
        }

    def _validate_cache_capacity(self) -> None:
        """检查每个 routed expert 层是否都有 CPU/GPU slot。

        异步 block 会在每个 routed expert 层请求专家。
        如果某一层缺 CPU 或 GPU slot，调度器无法推进，必须初始化阶段直接报错。
        """

        missing_cpu = [
            layer_idx
            for layer_idx in self._routed_expert_layers
            if self._cpu_cache_capacity.get(layer_idx, 0) <= 0
        ]
        missing_gpu = [
            layer_idx
            for layer_idx in self._routed_expert_layers
            if self._gpu_cache_capacity.get(layer_idx, 0) <= 0
        ]

        if missing_cpu:
            raise ValueError(f"CPU cache capacity is missing for routed expert layers: {missing_cpu}")
        if missing_gpu:
            raise ValueError(f"GPU cache capacity is missing for routed expert layers: {missing_gpu}")

    def _probe_expert(self) -> tuple[int, dict[str, list[int]], torch.dtype]:
        """探测单个 routed expert 的展平大小、布局和 dtype。

        adapter 给出 gate/up/down 等权重名。
        loader 从 manifest 读取 shape、元素数和 dtype。
        """

        spec = None
        for layer_idx in self._routed_expert_layers:
            candidate = self.adapter.routed_expert_weight_name(layer_idx, 0)

            # DeepSeek 这类模型可能前面有 dense 层；只用 manifest 中真实存在的 expert 来探测。
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
        """把每层容量转换成连续 buffer 的行范围映射。"""

        layer_slot_map: dict[int, tuple[int, int]] = {}
        offset = 0

        # 按层号排序，保证每次初始化得到稳定的 buffer 布局。
        for layer_idx in sorted(cap):
            count = cap[layer_idx]
            layer_slot_map[layer_idx] = (offset, offset + count)
            offset += count

        return offset, layer_slot_map

    def close(self) -> None:
        """关闭后台调度线程和磁盘读取线程池。"""

        with self._cv:
            self._shutdown = True
            self._cv.notify_all()

        self._scheduler.join(timeout=1.0)
        self._disk_pool.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        """对象析构时尽量释放后台线程资源。"""

        try:
            self.close()
        except Exception:
            pass
