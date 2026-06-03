from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import torch

from moe_runtime.config import RuntimeConfig
from moe_runtime.loader import ManifestTensorLoader


# ======================================================================
# 缓存槽位元信息映射表条目
# ======================================================================

@dataclass
class _SlotInfo:
    """缓存槽位映射表的一条记录。

    整个 GPU / CPU 连续 buffer 等分为 rows 行（每行 = 一个专家），
    每行对应一条 _SlotInfo，记录该行存的是哪个专家、当前状态、多少步未使用。
    """
    layer_idx: int | None = None        # 所属 transformer 层编号
    expert_idx: int | None = None       # 当前存储的专家编号，None = 空闲
    status: str = "free"                # "free" | "used" | "evicted"
    last_used_step: int = 0             # 最后命中时的全局 step


# ======================================================================
# ExpertCacheManager
# ======================================================================

class ExpertCacheManager:
    """MoE 专家二级缓存管理器：GPU 缓存 → CPU 缓存 → 磁盘。

    核心设计：
    - init 时根据 config 每层容量一次性预分配 GPU / CPU 连续 buffer。
      buffer 形状为 [num_slots, expert_elems]，每"行"存一个展平的专家权重。
      同层 slot 在 buffer 中连续排列，保证 forward 时显存访问的连续性。
    - 每行对应一条 _SlotInfo 映射记录，跟踪 layer / expert / status / step。
    - 驱逐为懒操作：仅将映射表状态改为 "evicted"，数据留在原地不搬运。
      需要新 slot 时按 free > evicted > LRU-used 优先级覆盖。
    - 全局 step 计数器追踪每行"多少步没被使用"，作为 LRU 淘汰依据。
    - 通过 adapter 获取张量名，tensor_loader 从磁盘加载权重。
    """

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        adapter: Any,
        tensor_loader: ManifestTensorLoader,
    ) -> None:
        self.config = runtime_config
        self.adapter = adapter
        self.loader = tensor_loader
        self._cpu_device = "cpu"
        self._gpu_device = "cuda"

        self._global_step = 0
        self._expert_elems: int = 0
        self._expert_layout: dict[str,list[int, int]] = {}

        self._gpu_layer_slot_map: dict[int, tuple[int, int]] = {}
        self._cpu_layer_slot_map: dict[int, tuple[int, int]] = {}

        self._gpu_slots: list[_SlotInfo] = []
        self._cpu_slots: list[_SlotInfo] = []


        # ---- 探测单个专家的大小 & 切分布局 & dtype ----
        self._expert_elems, self._expert_layout, self._expert_dtype = self._probe_expert()

        # ---- 按 config 每层容量构建 layer → slot 行范围映射 ----
        self._num_cpu_experts, self._cpu_layer_slot_map = self._build_layer_slot_mapping(self.config.max_cpu_cache_capacity)
        self._num_gpu_experts, self._gpu_layer_slot_map = self._build_layer_slot_mapping(self.config.max_gpu_cache_capacity)

        # ---- 预分配 GPU / CPU 连续 buffer：[num_slots, expert_elems] ----
        if self._num_cpu_experts > 0:
            self._cpu_buffer = torch.empty(
                self._num_cpu_experts,
                self._expert_elems,
                dtype=self._expert_dtype,
                device=self._cpu_device,
            )

        if self._num_gpu_experts > 0:
            self._gpu_buffer = torch.empty(
                self._num_gpu_experts,
                self._expert_elems,
                dtype=self._expert_dtype,
                device=self._gpu_device,
            )


        # ---- 初始化映射表（每行填入所属 layer_idx） ----
        for layer_idx, (start, end) in self._cpu_layer_slot_map.items():
            for _ in range(start, end):
                self._cpu_slots.append(_SlotInfo(layer_idx=layer_idx))

        for layer_idx, (start, end) in self._gpu_layer_slot_map.items():
            for _ in range(start, end):
                self._gpu_slots.append(_SlotInfo(layer_idx=layer_idx))


    # ==================================================================
    # 公开接口
    # ==================================================================

    def step(self) -> None:
        """每个 forward pass 结束后调用，递增全局 step。

        使得未访问的 slot 的 step 差值逐步增大，在淘汰时被优先选中。
        """
        self._global_step += 1

    def get_expert(self, layer_idx: int, expert_idx: int) -> dict[str, torch.Tensor]:
        """获取指定专家的所有权重张量（GPU buffer 视图，零拷贝）。

        查找顺序：GPU → CPU → 磁盘。
        """
        # 查 GPU 映射表
        gpu_slot_idx = self._find_expert_in_slots(self._gpu_slots, self._gpu_layer_slot_map, layer_idx, expert_idx)
        if gpu_slot_idx is not None:
            return self._get_expert_from_gpu(gpu_slot_idx)

        # 查 CPU 映射表
        cpu_slot_idx = self._find_expert_in_slots(self._cpu_slots, self._cpu_layer_slot_map, layer_idx, expert_idx)
        if cpu_slot_idx is not None:
            return self._get_expert_from_cpu(layer_idx, expert_idx, cpu_slot_idx)

        # 磁盘回源
        return self._get_expert_from_disk(layer_idx, expert_idx)

    # ==================================================================
    # 映射表查找
    # ==================================================================

    @staticmethod
    def _find_expert_in_slots(
        slots: list[_SlotInfo],
        layer_slot_map: dict[int, tuple[int, int]],
        layer_idx: int,
        expert_idx: int,
    ) -> int | None:
        """在指定层的 slot 范围内查找专家，返回 slot 索引或 None。

        匹配条件：expert_idx 相同且 status != "free"（used 或 evicted 都算命中）。
        """
        r = layer_slot_map.get(layer_idx)
        if r is None:
            return None
        for i in range(r[0], r[1]):
            s = slots[i]
            if s.expert_idx == expert_idx and s.status != "free":
                return i
        return None

    # ==================================================================
    # 从GPU Buffer 中视图切片：按 layout 还原各角色张量（零拷贝）
    # ==================================================================

    def _get_expert_from_gpu(self, gpu_slot_idx: int) -> dict[str, torch.Tensor]:
        """从 2D buffer 的第 slot_idx 行按 layout 切出各角色张量视图。

        layout 格式：{"gate": [1408,2048], "up": [1408,2048], "down": [2048,1408]}
        """
        self._gpu_slots[gpu_slot_idx].status = "used"
        self._gpu_slots[gpu_slot_idx].last_used_step = self._global_step
        offset = 0
        result: dict[str, torch.Tensor] = {}
        for role, shape in self._expert_layout.items():
            n = prod(shape)
            result[role] = self._gpu_buffer[gpu_slot_idx, offset : offset + n].view(torch.Size(shape))
            offset += n
        return result


    # ==================================================================
    # 从CPU Buffer 中copy 到 GPU Buffer
    # ==================================================================

    def _get_expert_from_cpu(self, layer_idx: int, expert_idx: int, cpu_slot_idx: int) -> dict[str, torch.Tensor]:
        """将 CPU buffer 第 cpu_slot 行整行拷入 GPU buffer 对应行。"""
        
        gpu_slot_idx = self._find_slot_to_overwrite_gpu(layer_idx)

        # 2D buffer：整行拷贝
        self._gpu_buffer[gpu_slot_idx, :].copy_(self._cpu_buffer[cpu_slot_idx, :])

        self._gpu_slots[gpu_slot_idx].expert_idx = expert_idx
        self._gpu_slots[gpu_slot_idx].status = "used"
        self._gpu_slots[gpu_slot_idx].last_used_step = self._global_step

        # CPU 源行标记 evicted（数据残留，后续可覆盖）
        self._cpu_slots[cpu_slot_idx].status = "evicted"

        return self._get_expert_from_gpu(gpu_slot_idx)
    
    # ==================================================================
    # 从磁盘中copy 到 GPU Buffer
    # ==================================================================

    def _get_expert_from_disk(self, layer_idx: int, expert_idx: int) -> dict[str, torch.Tensor]:
        """从磁盘加载专家权重：一次 IO 写入 CPU buffer，再整行拷入 GPU buffer。"""
        spec = self.adapter.routed_expert_weight_name(layer_idx, expert_idx)
        tensor_names = list(spec.values())

        # 从 mmap 直接写入 CPU buffer（零中间分配）
        cpu_slot_idx = self._find_slot_to_overwrite_cpu(layer_idx)
        self.loader.load_tensors_to_buffer(
            tensor_names=tensor_names,
            buffer=self._cpu_buffer,
            slot_idx=cpu_slot_idx,
        )

        self._cpu_slots[cpu_slot_idx].expert_idx = expert_idx
        self._cpu_slots[cpu_slot_idx].status = "used"
        self._cpu_slots[cpu_slot_idx].last_used_step = self._global_step

        # CPU buffer 整行拷入 GPU buffer
        gpu_slot_idx = self._find_slot_to_overwrite_gpu(layer_idx)
        self._gpu_buffer[gpu_slot_idx, :].copy_(self._cpu_buffer[cpu_slot_idx, :])
        self._gpu_slots[gpu_slot_idx].expert_idx = expert_idx
        self._gpu_slots[gpu_slot_idx].status = "used"
        self._gpu_slots[gpu_slot_idx].last_used_step = self._global_step

        return self._get_expert_from_gpu(gpu_slot_idx)

    # ==================================================================
    # 可覆盖 slot 选择：优先 free → evicted → LRU-used
    # ==================================================================

    def _find_slot_to_overwrite_gpu(self, layer_idx: int) -> int:
        """在指定层 GPU slot 范围内选可覆盖 slot。"""
        return self._pick_slot(self._gpu_slots, self._gpu_layer_slot_map, layer_idx)

    def _find_slot_to_overwrite_cpu(self, layer_idx: int) -> int:
        """在指定层 CPU slot 范围内选可覆盖 slot。"""
        return self._pick_slot(self._cpu_slots, self._cpu_layer_slot_map, layer_idx)

    @staticmethod
    def _pick_slot(
        slots: list[_SlotInfo],
        layer_slot_map: dict[int, tuple[int, int]],
        layer_idx: int,
    ) -> int:
        """在指定层 slot 范围内选可覆盖 slot。

        优先级：free > evicted > LRU-used（last_used_step 最小）。
        """
        r = layer_slot_map[layer_idx]
        free_idx: int | None = None
        evicted_idx: int | None = None
        used_idx: int | None = None
        used_step: int = 2 ** 63

        for i in range(r[0], r[1]):
            st = slots[i].status
            if st == "free":
                free_idx = i
                break
            if st == "evicted":
                evicted_idx = i
                break
            if st == "used" and slots[i].last_used_step < used_step:
                used_step = slots[i].last_used_step
                used_idx = i

        return free_idx if free_idx is not None else (evicted_idx if evicted_idx is not None else used_idx)


    def _load_tensors_to_buffer(
        self,
        *,
        buffer: torch.Tensor,
        tensor_names: list[str],
        slot_idx: int,
    ) -> None:
        """从磁盘加载张量，分块拷入 2D buffer 的第 slot_idx 行。

        不做 torch.cat：逐张量 flatten 后 copy_ 到 buffer[slot_idx, start:end]。
        """
        raw = self.loader.load_tensors_by_names(tensor_names=tensor_names, device="cpu")
        offset = 0
        for name in tensor_names:
            t = raw[name]
            n = t.numel()
            buffer[slot_idx, offset : offset + n].copy_(t.flatten())
            offset += n


    # ==================================================================
    # 初始化：
    # ==================================================================

    def _probe_expert(self) -> tuple[int, dict[str, list[int, int], torch.dtype]]:
        """通过 adapter + tensor_loader 探测单个专家的总元素数和张量切分布局。

        以 layer 0, expert 0 采样（所有 MoE 层专家结构一致）。
        adapter 负责提供张量名，tensor_loader 负责提供 shape / dtype。
        """
        spec = self.adapter.routed_expert_weight_name(0, 0)
        role_names = list(spec.keys())
        tensor_names = list(spec.values())
        
        expert_elms = 0
        layout = {}


        for role_name, tensor_name in zip(role_names, tensor_names):
            shape, elms, dtype = self.loader.get_tensor_info(tensor_name=tensor_name)
            expert_elms += elms
            layout[role_name] = shape
            torch_dtype = dtype

        return expert_elms, layout, torch_dtype
        
    def _build_layer_slot_mapping(self, cap: dict[int,int]) -> tuple[int, dict[int, tuple[int, int]]]:
        """
        构建 layer → slot 行范围映射，直接赋值目标属性
        """

        layer_slot_map = {}
        offset = 0
        # 严格遵循原逻辑：按层索引排序遍历
        for layer_idx in sorted(cap):
            count = cap[layer_idx]
            layer_slot_map[layer_idx] = (offset, offset + count)
            offset += count
        # 专家数量 = 层数，直接返回
        return offset, layer_slot_map




