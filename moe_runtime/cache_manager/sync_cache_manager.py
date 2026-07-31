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

# demand 使用独立磁盘 worker，避免已经排队的预取阻塞当前层真实专家。
_DEMAND_DISK_WORKERS = 4

# 预取只占用少量独立 worker，防止低优先级读取耗尽磁盘并发。
_PREFETCH_DISK_WORKERS = 2

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

    # LRU 依据。None 表示尚未被真实 demand 使用，应当优先于正常缓存被覆盖。
    last_used_step: int | None = None

    # LFU 依据。当前 expert 在该 slot 中被命中的次数越少，越容易被覆盖。
    access_count: int = 0

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

    # demand 和预取共用 request；预取只流转到 prefetch_ready_cpu，不进入 GPU。
    stage: str

    # True 表示低优先级预取；真实 block 请求到达后会原地提升为 demand。
    is_prefetch: bool = False

    # CPU/GPU slot 下标，以及发起异步任务时对应的 generation。
    cpu_slot_idx: int | None = None
    gpu_slot_idx: int | None = None
    cpu_generation: int | None = None
    gpu_generation: int | None = None

    # 磁盘到 CPU 的后台线程任务。
    disk_future: Future | None = None

    # 标记磁盘任务由 demand 还是 prefetch 线程池执行，用于准确限制两类并发。
    disk_executor: str | None = None

    # 错误预取已开始执行时不能立刻复用写入 slot，完成后再安全释放。
    cancel_after_load: bool = False

    # CPU 到 GPU 的 CUDA stream 事件；非 CUDA 目标设备时为 None。
    h2d_event: torch.cuda.Event | None = None

    # block release 后记录的计算完成事件，用来保护正在被 GPU 计算使用的 slot。
    compute_event: torch.cuda.Event | None = None

    # 异步任务里的异常会挂到 request 上，由 block 下一次查询时抛出。
    error: BaseException | None = None


class asyncExpertCacheManager:
    """异步 routed expert 缓存管理器。

    设计边界：
    - experts block 调用 demand 接口，逐层 gate 预取模块只调用 prefetch_experts。
    - CacheManager 独立维护 CPU/GPU slot 四状态，并由后台调度线程推进磁盘读取和 H2D 拷贝。
    - GPU 上已经命中的 expert 会直接 protected，然后交给 block 立刻计算。
    - GPU 上未命中的 expert 会排队加载；加载完成后状态变为 protected，保证马上计算前不会被 LRU 覆盖。
    - 预取只加载到 CPU，完成后保留 request，直到目标层真实 gate 统一对账。
    - 多个上游层对同一目标层的预测累积去重，新预测任务排在旧预取任务之前。
    """

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        adapter: Any,
        tensor_loader: ManifestTensorLoader,
        model_config: Any | None = None,
    ) -> None:
        """
        初始化缓存布局、异步调度队列和后台调度线程。

        输入：运行配置、模型 Adapter、张量加载器和模型配置。
        输出：无。
        功能：建立 demand/预取队列、CPU/GPU 连续 buffer、磁盘线程池和 H2D 调度 stream。
        """

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

        # 预取只使用 CPU 队列和加载集合，不提交低优先级 H2D。
        self._prefetch_waiting_cpu: deque[ExpertKey] = deque()
        self._prefetch_loading_cpu: set[ExpertKey] = set()

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
            #pin_memory=self._is_cuda_target
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

        # demand 和预取使用独立线程池，确保当前层真实请求始终有可用 worker。
        self._demand_disk_pool = ThreadPoolExecutor(
            max_workers=_DEMAND_DISK_WORKERS,
            thread_name_prefix="moe-demand-disk",
        )
        self._prefetch_disk_pool = ThreadPoolExecutor(
            max_workers=_PREFETCH_DISK_WORKERS,
            thread_name_prefix="moe-prefetch-disk",
        )

        # H2D 使用独立 CUDA stream，避免阻塞当前计算 stream。
        self._h2d_stream = torch.cuda.Stream() if self._is_cuda_target else None

        # 同一把锁保护 request 表、slot 状态和所有调度队列。
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

        # 后台调度线程只做非阻塞推进；真正等待只发生在 block 的 wait_any_ready。
        self._shutdown = False
        self._closed = False
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

    def request_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
        """
        原子对账并登记当前层全部真实 expert。

        输入：目标 transformer 层编号和当前 gate 实际命中的 expert 编号列表。
        输出：无。
        功能：先取消本层全部未命中预取，再提升命中预取并登记遗漏 demand，保证未来层预取提交前本层请求已完整可见。
        """

        actual_expert_ids = list(dict.fromkeys(int(expert_idx) for expert_idx in expert_ids))
        actual_keys = {
            (layer_idx, expert_idx)
            for expert_idx in actual_expert_ids
        }

        with self._cv:
            # 到达目标层后预测结果已经可以判定，未命中的预取不应继续占用队列或 CPU slot。
            for key, req in list(self._requests.items()):
                if key[0] != layer_idx or not req.is_prefetch or key in actual_keys:
                    continue
                self._cancel_prefetch_request_locked(key, req)

            # 在同一临界区登记全部 demand，预取线程看不到只登记了一部分专家的中间状态。
            for expert_idx in actual_expert_ids:
                self._request_expert_locked((layer_idx, expert_idx))

            self._cv.notify_all()

    def prefetch_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
        """
        提交一组未来层预测 expert 的低优先级预取请求。

        输入：目标 transformer 层编号和预测 expert 编号列表。
        输出：无。
        功能：累积并去重多个上游层的预测，只把新增 CPU 预取插到队首，不因后续预测不同而取消旧预测。
        """

        with self._cv:
            new_keys: list[ExpertKey] = []
            for expert_idx in dict.fromkeys(int(expert_idx) for expert_idx in expert_ids):
                key = (layer_idx, int(expert_idx))

                # demand、先前预取或可读缓存已经覆盖该 key 时，不重复创建磁盘任务。
                if key in self._requests:
                    continue

                gpu_slot_idx = self._find_expert_in_slots(
                    self._gpu_slots,
                    self._gpu_layer_slot_map,
                    layer_idx,
                    int(expert_idx),
                )
                if gpu_slot_idx is not None:
                    continue

                cpu_slot_idx = self._find_expert_in_slots(
                    self._cpu_slots,
                    self._cpu_layer_slot_map,
                    layer_idx,
                    int(expert_idx),
                )
                if cpu_slot_idx is not None:
                    continue

                self._requests[key] = _ExpertRequest(
                    layer_idx=layer_idx,
                    expert_idx=int(expert_idx),
                    stage="prefetch_waiting_cpu",
                    is_prefetch=True,
                )
                new_keys.append(key)

            # appendleft 会反转顺序，因此倒序插入以保持同一批 expert 的原始顺序。
            for key in reversed(new_keys):
                self._prefetch_waiting_cpu.appendleft(key)

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
            self._touch_slot(gpu_slot)
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
        """
        阻塞等待一组 expert 中任意一个变为 ready。

        输入：当前层号和已经批量登记的 pending expert 编号列表。
        输出：最先 ready 的 expert 编号。
        功能：只等待 request_experts 已登记的请求；状态缺失时直接报错，不在等待路径补建 demand。
        """

        if not expert_ids:
            raise ValueError("wait_any_ready requires at least one expert id.")

        keys = [(layer_idx, expert_idx) for expert_idx in expert_ids]
        with self._cv:
            # pending expert 必须已经由 block 通过批量接口原子登记。
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
        """
        执行一次非阻塞调度 tick。

        输入：无；调用方必须已经持有 self._cv。
        输出：至少一个请求或 slot 状态发生推进时返回 True。
        功能：先推进 demand 的计算、H2D 和磁盘读取，再使用独立线程池推进 CPU-only 预取。
        """

        progressed = False

        # 先回收计算完成的 GPU slot，让后续 H2D 有机会拿到可覆盖 slot。
        progressed |= self._complete_finished_compute_locked()

        # 再检查 H2D 完成，把 loading GPU slot 转为 protected ready。
        progressed |= self._complete_finished_h2d_locked()

        # 再检查磁盘读取完成，把 loading CPU slot 转为 protected，并放入 waiting_gpu。
        progressed |= self._complete_finished_disk_reads_locked()

        # 预取完成只生成 CPU used 缓存，不会进入 H2D 或当前 block 可见的 ready 集合。
        progressed |= self._complete_finished_prefetch_disk_reads_locked()

        # 优先调度 CPU -> GPU，让 CPU hit 或刚读完磁盘的 expert 尽快可算。
        progressed |= self._schedule_h2d_locked()

        # 最后补充磁盘回源任务，持续填充 CPU cache。
        progressed |= self._schedule_disk_reads_locked()

        # 预取使用独立受限线程池，不会占用 demand 的磁盘 worker。
        progressed |= self._schedule_prefetch_disk_reads_locked()

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
            self._reset_slot_usage(gpu_slot)
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
        """
        检查 demand 磁盘读取是否完成。

        输入：无；调用方必须已经持有 self._cv。
        输出：至少一个 demand 磁盘任务完成时返回 True。
        功能：把 CPU slot 从 loading 转为 protected 并排入 H2D，同时清除已完成 Future 以释放线程池并发计数。
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
            self._reset_slot_usage(cpu_slot)
            cpu_slot.status = "protected"
            req.stage = "waiting_gpu"
            req.disk_future = None
            req.disk_executor = None
            self._loading_cpu.discard(key)
            self._waiting_gpu.append(key)
            progressed = True

        return progressed

    def _complete_finished_prefetch_disk_reads_locked(self) -> bool:
        """
        检查低优先级预取磁盘读取是否完成。

        输入：无；调用方必须已经持有 self._cv。
        输出：至少一个预取磁盘任务完成时返回 True。
        功能：把有效预测保留为冷 CPU 缓存；已在目标层判定错误的运行中预取完成后立即释放。
        """

        progressed = False
        for key in list(self._prefetch_loading_cpu):
            req = self._requests.get(key)
            if req is None:
                raise RuntimeError(f"Prefetch loading CPU expert has no request: {key}")

            if not req.is_prefetch or req.stage != "prefetch_loading_cpu":
                raise RuntimeError(
                    f"Prefetch loading CPU expert has invalid request: {key}, "
                    f"stage={req.stage}, is_prefetch={req.is_prefetch}"
                )

            if req.disk_future is None:
                raise RuntimeError(f"Prefetch loading CPU expert has no disk future: {key}")

            if not req.disk_future.done():
                continue

            try:
                req.disk_future.result()
            except BaseException as exc:
                raise RuntimeError(f"Prefetch disk read failed for expert {key}") from exc

            if req.cpu_slot_idx is None:
                raise RuntimeError(f"Prefetch loading CPU expert has no CPU slot: {key}")

            cpu_slot = self._cpu_slots[req.cpu_slot_idx]
            if (
                cpu_slot.generation != req.cpu_generation
                or cpu_slot.status != "loading"
                or cpu_slot.layer_idx != req.layer_idx
                or cpu_slot.expert_idx != req.expert_idx
            ):
                raise RuntimeError(f"Prefetch loading CPU slot state is invalid: {key}")

            if req.cancel_after_load:
                # 运行中的磁盘读取不能强制终止，写入结束后才可安全释放其 slot。
                self._free_slot_locked(cpu_slot)
                self._prefetch_loading_cpu.discard(key)
                self._requests.pop(key, None)
                progressed = True
                continue

            # 预测尚未被真实 demand 使用，保持冷统计并保留 request 等待目标层对账。
            self._mark_prefetched_slot(cpu_slot)
            req.stage = "prefetch_ready_cpu"
            req.disk_future = None
            req.disk_executor = None

            self._prefetch_loading_cpu.discard(key)
            progressed = True

        return progressed

    def _schedule_h2d_locked(self) -> bool:
        """
        为 demand 的 waiting_gpu 队列调度 CPU 到 GPU 拷贝。

        输入：无；调用方必须已经持有 self._cv。
        输出：至少提交一个 H2D 任务时返回 True。
        功能：完整扫描当前队列并跳过暂时没有本层 GPU slot 的请求，避免队首阻塞其他层。
        """

        progressed = False
        scan_count = len(self._waiting_gpu)
        for _ in range(scan_count):
            key = self._waiting_gpu.popleft()
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
                # 当前层没有可覆盖 slot 时把请求移到队尾，继续尝试其他层。
                self._waiting_gpu.append(key)
                continue

            gpu_slot = self._gpu_slots[gpu_slot_idx]
            self._prepare_slot_for_write_locked(self._gpu_slots, gpu_slot_idx)

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
        """
        为 demand 的 waiting_cpu 队列调度磁盘到 CPU 读取。

        输入：无；调用方必须已经持有 self._cv。
        输出：至少提交一个 demand 磁盘任务时返回 True。
        功能：完整扫描当前队列，使用独立 demand 线程池并跳过暂时没有本层 CPU slot 的请求。
        """

        progressed = False
        active_demand_reads = self._count_disk_reads_locked("demand")
        scan_count = len(self._waiting_cpu)
        for _ in range(scan_count):
            if active_demand_reads >= _DEMAND_DISK_WORKERS:
                break

            key = self._waiting_cpu.popleft()
            req = self._requests.get(key)

            if req is None:
                raise RuntimeError(f"Waiting CPU expert has no request: {key}")

            if req.stage != "waiting_cpu":
                raise RuntimeError(f"Waiting CPU expert has invalid stage: {key}, stage={req.stage}")

            if req.cpu_slot_idx is not None:
                raise RuntimeError(f"Waiting CPU expert already has CPU slot: {key}")

            cpu_slot_idx = self._pick_slot(self._cpu_slots, self._cpu_layer_slot_map, req.layer_idx)
            if cpu_slot_idx is None:
                # 某层 slot 被保护时不能阻塞队列中仍有可写 slot 的其他层。
                self._waiting_cpu.append(key)
                continue

            cpu_slot = self._cpu_slots[cpu_slot_idx]
            self._prepare_slot_for_write_locked(self._cpu_slots, cpu_slot_idx)

            # CPU slot 进入 loading 后，后台磁盘线程拥有写入权。
            cpu_slot.layer_idx = req.layer_idx
            cpu_slot.expert_idx = req.expert_idx
            cpu_slot.status = "loading"
            cpu_slot.generation += 1

            req.cpu_slot_idx = cpu_slot_idx
            req.cpu_generation = cpu_slot.generation
            req.stage = "loading_cpu"

            spec = self.adapter.routed_expert_weight_name(req.layer_idx, req.expert_idx)
            req.disk_future = self._demand_disk_pool.submit(
                self.loader.load_tensors_to_buffer,
                tensor_names=list(spec.values()),
                buffer=self._cpu_buffer,
                slot_idx=cpu_slot_idx,
            )
            req.disk_executor = "demand"
            self._loading_cpu.add(key)
            active_demand_reads += 1
            progressed = True

        return progressed

    def _schedule_prefetch_disk_reads_locked(self) -> bool:
        """
        使用独立低优先级线程池提交 CPU 预取读取。

        输入：无；调用方必须已经持有 self._cv。
        输出：至少提交一个预取磁盘任务时返回 True。
        功能：完整扫描预取队列并跳过无可写 slot 的层；新预测已由提交接口插入队首。
        """

        progressed = False
        active_prefetch_reads = self._count_disk_reads_locked("prefetch")
        scan_count = len(self._prefetch_waiting_cpu)
        for _ in range(scan_count):
            if active_prefetch_reads >= _PREFETCH_DISK_WORKERS:
                break

            key = self._prefetch_waiting_cpu.popleft()
            req = self._requests.get(key)
            if req is None:
                raise RuntimeError(f"Prefetch waiting CPU expert has no request: {key}")

            if not req.is_prefetch or req.stage != "prefetch_waiting_cpu":
                raise RuntimeError(
                    f"Prefetch waiting CPU expert has invalid request: {key}, "
                    f"stage={req.stage}, is_prefetch={req.is_prefetch}"
                )

            cpu_slot_idx = self._pick_slot(
                self._cpu_slots,
                self._cpu_layer_slot_map,
                req.layer_idx,
            )
            if cpu_slot_idx is None:
                self._prefetch_waiting_cpu.append(key)
                continue

            cpu_slot = self._cpu_slots[cpu_slot_idx]
            self._prepare_slot_for_write_locked(self._cpu_slots, cpu_slot_idx)
            cpu_slot.layer_idx = req.layer_idx
            cpu_slot.expert_idx = req.expert_idx
            cpu_slot.status = "loading"
            cpu_slot.generation += 1

            req.cpu_slot_idx = cpu_slot_idx
            req.cpu_generation = cpu_slot.generation
            req.stage = "prefetch_loading_cpu"

            spec = self.adapter.routed_expert_weight_name(req.layer_idx, req.expert_idx)
            req.disk_future = self._prefetch_disk_pool.submit(
                self.loader.load_tensors_to_buffer,
                tensor_names=list(spec.values()),
                buffer=self._cpu_buffer,
                slot_idx=cpu_slot_idx,
            )
            req.disk_executor = "prefetch"
            self._prefetch_loading_cpu.add(key)
            active_prefetch_reads += 1
            progressed = True

        return progressed

    def _request_expert_locked(self, key: ExpertKey) -> None:
        """
        在锁内登记一个真实 demand。

        输入：expert 唯一键；调用方必须已经持有 self._cv。
        输出：无。
        功能：检查异步异常，提升同 key 的预取任务，或按 GPU、CPU、磁盘顺序创建 demand。
        """

        self._raise_if_error(key)
        req = self._requests.get(key)
        if req is None:
            self._enqueue_request(key)
        elif req.is_prefetch:
            # 真实路由命中预取时复用已完成或正在执行的磁盘读取。
            self._promote_prefetch_request_locked(key, req)

    def _enqueue_request(self, key: ExpertKey) -> None:
        """
        按 GPU、CPU、磁盘命中顺序把 demand 放入调度系统。

        输入：expert 唯一键；调用方必须已经持有 self._cv。
        输出：无。
        功能：缓存命中时立即保护 slot，未命中时进入高优先级 demand 磁盘队列。
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
            self._touch_slot(gpu_slot)
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
            self._touch_slot(cpu_slot)
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
        self._waiting_cpu.appendleft(key)

    def _promote_prefetch_request_locked(
        self,
        key: ExpertKey,
        req: _ExpertRequest,
    ) -> None:
        """
        将正在等待或加载的预取请求提升为真实 demand。

        输入：expert 唯一键和对应的预取请求对象；调用方必须已经持有 self._cv。
        输出：无。
        功能：复用已完成或运行中的 CPU 预取；尚未执行的任务改投 demand 线程池并获得最高调度优先级。
        """

        if not req.is_prefetch:
            raise RuntimeError(f"Cannot promote a non-prefetch request: {key}")

        if req.stage == "prefetch_waiting_cpu":
            self._prefetch_waiting_cpu.remove(key)
            req.is_prefetch = False
            req.stage = "waiting_cpu"
            self._waiting_cpu.appendleft(key)
            return

        if req.stage == "prefetch_loading_cpu":
            self._prefetch_loading_cpu.discard(key)
            self._loading_cpu.add(key)
            req.is_prefetch = False
            req.stage = "loading_cpu"

            if req.disk_future is None:
                raise RuntimeError(f"Prefetch loading CPU expert has no disk future: {key}")

            # 尚在预取线程池队列中的任务可取消并立即改投 demand 池；已运行任务直接复用。
            if req.disk_future.cancel():
                if req.cpu_slot_idx is None:
                    raise RuntimeError(f"Prefetch loading CPU expert has no CPU slot: {key}")
                spec = self.adapter.routed_expert_weight_name(req.layer_idx, req.expert_idx)
                req.disk_future = self._demand_disk_pool.submit(
                    self.loader.load_tensors_to_buffer,
                    tensor_names=list(spec.values()),
                    buffer=self._cpu_buffer,
                    slot_idx=req.cpu_slot_idx,
                )
                req.disk_executor = "demand"
            return

        if req.stage == "prefetch_ready_cpu":
            if req.cpu_slot_idx is None:
                raise RuntimeError(f"Prefetch ready CPU expert has no CPU slot: {key}")

            cpu_slot = self._cpu_slots[req.cpu_slot_idx]
            if (
                cpu_slot.generation != req.cpu_generation
                or cpu_slot.status != "used"
                or cpu_slot.layer_idx != req.layer_idx
                or cpu_slot.expert_idx != req.expert_idx
            ):
                raise RuntimeError(f"Prefetch ready CPU expert slot state is invalid: {key}")

            # 只有真实 demand 接管时才更新访问统计并保护 H2D 源。
            self._touch_slot(cpu_slot)
            cpu_slot.status = "protected"
            req.is_prefetch = False
            req.stage = "waiting_gpu"
            self._waiting_gpu.appendleft(key)
            return

        raise RuntimeError(
            f"Prefetch request cannot be promoted from stage: {key}, stage={req.stage}"
        )

    def _cancel_prefetch_request_locked(
        self,
        key: ExpertKey,
        req: _ExpertRequest,
    ) -> None:
        """
        取消目标层真实 gate 未命中的预取。

        输入：expert 唯一键和对应预取请求；调用方必须已经持有 self._cv。
        输出：无。
        功能：等待任务立即删除，未运行 Future 取消后释放 slot，运行中任务标记为完成后释放。
        """

        if not req.is_prefetch:
            raise RuntimeError(f"Cannot cancel a non-prefetch request: {key}")

        if req.stage == "prefetch_waiting_cpu":
            self._prefetch_waiting_cpu.remove(key)
            self._requests.pop(key, None)
            return

        if req.stage == "prefetch_loading_cpu":
            if req.disk_future is None or req.cpu_slot_idx is None:
                raise RuntimeError(f"Prefetch loading CPU expert is incomplete: {key}")

            if not req.disk_future.cancel():
                # 后台线程可能正在写入，必须等 Future 完成后才能回收对应 CPU slot。
                req.cancel_after_load = True
                return

            cpu_slot = self._cpu_slots[req.cpu_slot_idx]
            if (
                cpu_slot.generation != req.cpu_generation
                or cpu_slot.status != "loading"
                or cpu_slot.layer_idx != req.layer_idx
                or cpu_slot.expert_idx != req.expert_idx
            ):
                raise RuntimeError(f"Cancelled prefetch CPU slot state is invalid: {key}")

            self._free_slot_locked(cpu_slot)
            self._prefetch_loading_cpu.discard(key)
            self._requests.pop(key, None)
            return

        if req.stage == "prefetch_ready_cpu":
            if req.cpu_slot_idx is None:
                raise RuntimeError(f"Prefetch ready CPU expert has no CPU slot: {key}")

            cpu_slot = self._cpu_slots[req.cpu_slot_idx]
            if (
                cpu_slot.generation != req.cpu_generation
                or cpu_slot.status != "used"
                or cpu_slot.layer_idx != req.layer_idx
                or cpu_slot.expert_idx != req.expert_idx
            ):
                raise RuntimeError(f"Prefetch ready CPU slot state is invalid: {key}")

            self._free_slot_locked(cpu_slot)
            self._requests.pop(key, None)
            return

        raise RuntimeError(
            f"Prefetch request cannot be cancelled from stage: {key}, stage={req.stage}"
        )

    def _count_disk_reads_locked(self, executor: str) -> int:
        """
        统计指定线程池仍由状态机跟踪的磁盘任务数。

        输入：线程池标识 demand 或 prefetch；调用方必须已经持有 self._cv。
        输出：对应线程池中尚未完成状态回收的任务数量。
        功能：分别限制 demand 和预取并发，提升后的运行中预取不会错误占用 demand 配额。
        """

        if executor not in {"demand", "prefetch"}:
            raise ValueError(f"Unsupported disk executor: {executor}")

        return sum(
            1
            for req in self._requests.values()
            if req.disk_executor == executor and req.disk_future is not None
        )

    def _prepare_slot_for_write_locked(
        self,
        slots: list[_SlotInfo],
        slot_idx: int,
    ) -> None:
        """
        在异步写入前清理被选中的 used slot。

        输入：CPU 或 GPU slot 列表及目标下标；调用方必须已经持有 self._cv。
        输出：无。
        功能：移除被淘汰的 CPU 预取 request；真实 demand 缓存没有预取 request，因此只覆盖数据不误删状态。
        """

        slot = slots[slot_idx]
        if slot.status == "free":
            return
        if slot.status != "used":
            raise RuntimeError(f"Cannot overwrite cache slot with status={slot.status}")

        if slots is self._cpu_slots and slot.layer_idx is not None and slot.expert_idx is not None:
            key = (slot.layer_idx, slot.expert_idx)
            req = self._requests.get(key)
            if req is not None:
                if not req.is_prefetch or req.stage != "prefetch_ready_cpu":
                    raise RuntimeError(f"Used CPU slot has an active non-prefetch request: {key}")
                self._requests.pop(key, None)

    @staticmethod
    def _free_slot_locked(slot: _SlotInfo) -> None:
        """
        将不再被异步任务使用的 slot 复位为空槽。

        输入：已确认可以安全释放的 slot。
        输出：无。
        功能：清除 expert 身份和访问统计；generation 保留到下一次写入时递增。
        """

        slot.expert_idx = None
        slot.status = "free"
        slot.last_used_step = None
        slot.access_count = 0

    @staticmethod
    def _mark_prefetched_slot(slot: _SlotInfo) -> None:
        """
        把完成的 CPU 预取标记为冷缓存。

        输入：已完成磁盘写入的 CPU slot。
        输出：无。
        功能：保持 access_count 为零且 last_used_step 为空，只有真实 demand 命中后才产生访问统计。
        """

        slot.status = "used"
        slot.last_used_step = None
        slot.access_count = 0

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

    def _pick_slot(
        self,
        slots: list[_SlotInfo],
        layer_slot_map: dict[int, tuple[int, int]],
        layer_idx: int,
    ) -> int | None:
        """
        在指定层选择一个可写 slot。

        输入：slot 列表、逐层 slot 范围和目标层号。
        输出：优先返回 free slot，否则返回缓存策略选出的 used slot；没有可写 slot 时返回 None。
        功能：protected 和 loading 永不覆盖，未被 demand 使用的冷预取优先成为 victim。
        """

        slot_range = layer_slot_map.get(layer_idx)
        if slot_range is None:
            return None

        free_idx: int | None = None
        used_idx: int | None = None

        for slot_idx in range(slot_range[0], slot_range[1]):
            slot = slots[slot_idx]
            if slot.status == "free":
                free_idx = slot_idx
                break

            if slot.status == "used" and self._is_better_victim(slot, slots[used_idx] if used_idx is not None else None):
                used_idx = slot_idx

        return free_idx if free_idx is not None else used_idx

    def _is_better_victim(self, candidate: _SlotInfo, current: _SlotInfo | None) -> bool:
        """
        判断候选 slot 是否更适合被覆盖。

        输入：候选 used slot 和当前最佳 victim。
        输出：候选更适合覆盖时返回 True。
        功能：LFU 优先比较访问次数，其他情况比较 LRU；没有 demand 访问时间的预取视为最冷。
        """

        if current is None:
            return True
        if self.config.cache_policy == "lfu" and candidate.access_count != current.access_count:
            return candidate.access_count < current.access_count
        candidate_step = -1 if candidate.last_used_step is None else candidate.last_used_step
        current_step = -1 if current.last_used_step is None else current.last_used_step
        return candidate_step < current_step

    def _reset_slot_usage(self, slot: _SlotInfo) -> None:
        """新 expert 写入 slot 后，初始化使用统计。"""

        slot.status = "used"
        slot.last_used_step = self._global_step
        slot.access_count = 1

    def _touch_slot(self, slot: _SlotInfo) -> None:
        """expert 命中时更新使用统计。"""

        slot.last_used_step = self._global_step
        slot.access_count += 1

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
        """
        关闭后台调度资源并释放缓存 buffer。

        输入：无。
        输出：无。
        功能：停止调度线程，等待磁盘/H2D 任务完成，再清空 demand、预取队列和大块缓存引用。
        """

        if getattr(self, "_closed", False):
            return

        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._shutdown = True
            self._cv.notify_all()

        # 调度线程退出后，CacheManager 不会再提交新的磁盘读取或 H2D 拷贝。
        self._scheduler.join()

        # 已经提交到独立 stream 的 H2D 拷贝必须完成后，才能释放 CPU/GPU buffer。
        if self._is_cuda_target and self._h2d_stream is not None:
            self._h2d_stream.synchronize()

        # 等待已经开始的磁盘读取完成；尚未开始的任务会被取消。
        # 两个线程池都结束后，才能安全断开 CPU buffer 引用。
        self._demand_disk_pool.shutdown(wait=True, cancel_futures=True)
        self._prefetch_disk_pool.shutdown(wait=True, cancel_futures=True)

        # 断开大块 buffer 和状态表引用，帮助 Python/torch 尽快归还 pinned/GPU 内存。
        self._cpu_buffer = None
        self._gpu_buffer = None
        self._cpu_slots.clear()
        self._gpu_slots.clear()
        self._requests.clear()
        self._waiting_cpu.clear()
        self._waiting_gpu.clear()
        self._loading_cpu.clear()
        self._loading_gpu.clear()
        self._ready.clear()
        self._compute_pending.clear()
        self._prefetch_waiting_cpu.clear()
        self._prefetch_loading_cpu.clear()

    def __del__(self) -> None:
        """对象析构时尽量释放后台线程资源。"""

        try:
            self.close()
        except Exception:
            pass
