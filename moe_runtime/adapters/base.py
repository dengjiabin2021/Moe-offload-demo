from __future__ import annotations

# abc 用来定义 Adapter 抽象接口，确保不同模型适配器拥有一致能力。
from abc import ABC, abstractmethod
# dataclass 用来描述 expert 权重和 MoE block 引用。
from dataclasses import dataclass
# Path 用于模型目录类型标注。
from pathlib import Path
# Any 用于 transformers config、manifest metadata、runtime block 等类型占位。
from typing import Any, Iterable

# nn.Module 是 transformers 模型和 block 的通用基类。
from torch import nn

# RuntimeConfig 用于根据用户配置决定常驻设备。
from moe_runtime.config import RuntimeConfig


@dataclass(frozen=True)
class ExpertWeightSpec:
    """描述某个 expert 需要从 checkpoint 中读取哪些 tensor。"""

    # Transformer 层编号。
    layer_idx: int

    # 当前层内 expert 编号。
    expert_idx: int

    # 权重角色到 tensor name 的映射。
    # 不同模型可以使用不同角色名，例如 gate/up/down 或 w1/w2/w3。
    tensors: dict[str, str]



class MoeModelAdapter(ABC):
    """MoE 模型适配器基类。

    Adapter 只负责屏蔽不同模型的结构差异，提供“模型说明书”：
    - 如何识别模型
    - 如何构造空模型
    - 哪些 tensor 是 routed expert
    - 哪些 tensor 是 shared expert
    - 某个 expert 需要哪些 tensor


    Adapter 不负责：
    - 调用 tensor loader
    - 遍历 manifest 加载权重
    - 管理缓存或预取
    - 执行完整推理流程
    """

    @abstractmethod
    def match(self, model_config: Any) -> bool:
        """判断当前 Adapter 是否支持这个模型。"""


    @abstractmethod
    def build_empty_model(self, model_config: Any) -> nn.Module:
        """构造空模型结构。

        调用方应在 init_empty_weights 上下文中调用该方法。
        """

    @abstractmethod
    def is_routed_expert_tensor(self, tensor_name: str) -> bool:
        """判断 checkpoint manifest 中的 tensor 是否是 routed expert 权重。"""

    @abstractmethod
    def is_shared_expert_tensor(self, tensor_name: str) -> bool:
        """判断 checkpoint manifest 中的 tensor 是否是 shared expert 权重。"""

    @abstractmethod
    def routed_expert_weight_spec(self, layer_idx: int, expert_idx: int) -> ExpertWeightSpec:
        """返回某层某个 expert 所需的 tensor_name 映射。"""

    @abstractmethod
    def patch_moe_block(self, model: nn.Module) -> nn.Module:
        """将原始 MoE block替换"""

