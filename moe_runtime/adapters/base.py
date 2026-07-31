from __future__ import annotations

# abc 用来定义 Adapter 抽象接口，确保不同模型适配器拥有一致能力。
from abc import ABC, abstractmethod
# dataclass 用来描述 expert 权重和 MoE block 引用。
from dataclasses import dataclass
# Path 用于模型目录类型标注。
from pathlib import Path
# Any 用于 transformers config、manifest metadata、runtime block 等类型占位。
from typing import Any, Iterable

import torch
# nn.Module 是 transformers 模型和 block 的通用基类。
from torch import nn

# RuntimeConfig 用于根据用户配置决定常驻设备。
from moe_runtime.config import RuntimeConfig



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
    def routed_expert_weight_name(self, layer_idx: int, expert_idx: int) -> dict[str, str]:
        """返回某层某个 expert 所需的 tensor_name 映射。"""

    @abstractmethod
    def routed_expert_layers(self, model_config: Any) -> list[int]:
        """返回包含 routed experts 的 transformer layer 索引。"""

    @abstractmethod
    def patch_moe_block(
        self,
        model: nn.Module,
        model_config: Any,
        runtime_config: RuntimeConfig,
        cache_manager: Any | None = None,
    ) -> nn.Module:
        """
        替换模型中的 routed experts。

        输入：空模型、模型配置、运行时配置和 CacheManager。
        输出：完成 routed experts 替换的模型。
        功能：子类只负责按模型结构执行模块替换，不在这里注册运行期 hook。
        """

    @abstractmethod
    def materialize_meta_tensors(self, model: nn.Module, target_device: torch.device) -> None:
        """Materialize 残留的 meta tensor。

        像 RoPE 的 inv_freq 这样 persistent=False 的计算 buffer，
        在 meta device 上构建时没有真实数据，需要在此重新计算并迁移到目标设备。
        默认实现遍历所有模块，用 torch.empty 填充任何还在 meta 上的 buffer / parameter。
        子类可重写以正确处理特定计算 buffer（如 inv_freq 重算）。
        """
