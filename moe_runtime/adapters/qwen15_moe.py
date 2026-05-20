from __future__ import annotations

# re 用于匹配 Qwen checkpoint 中的 expert/shared expert 权重名。
import re
# Any 用于 transformers config 和 manifest metadata 类型标注。
from typing import Any

from pathlib import Path

# nn.Module 用于返回类型标注。
from torch import nn

from accelerate import init_empty_weights
# Qwen1.5-MoE-A2.7B 在 transformers 中复用 Qwen2MoeForCausalLM。
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM 

from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig


# 引入 Adapter 基类和结构化返回类型。
from .base import ExpertWeightSpec, MoeModelAdapter


class Qwen15MoeAdapter(MoeModelAdapter):
    """Qwen1.5-MoE-A2.7B 适配器。"""

    supported_model_types: list[str] = ["qwen2_moe"]

    # checkpoint 中 routed expert 权重名：model.layers.0.mlp.experts.0.gate_proj.weight
    routed_expert_re = re.compile(
        r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
    )

    # checkpoint 中 shared expert 权重名：model.layers.0.mlp.shared_expert.gate_proj.weight && model.layers.0.mlp.shared_expert_gate.weight
    shared_expert_re = re.compile(
        r"^model\.layers\.(?P<layer>\d+)\.mlp\."
        r"(?P<name>shared_expert\.(gate_proj|up_proj|down_proj)\.weight|shared_expert_gate\.weight)$"
    )

    def match(self, model_config: Qwen2MoeConfig) -> bool:
        """判断当前 Adapter 是否支持这个模型。"""

        # 默认通过 config.model_type 匹配；特殊模型可在子类重写。
        return getattr(model_config, "model_type", None) in self.supported_model_types   


    def build_empty_model(self, model_config: Qwen2MoeConfig) -> Qwen2MoeForCausalLM:
        """构造 Qwen2MoeForCausalLM 空模型。"""

        # 内置空模型上下文，无需外部依赖，100% 返回空模型
        with init_empty_weights():
            model = Qwen2MoeForCausalLM(model_config)
        return model

    def is_routed_expert_tensor(self, tensor_name: str) -> bool:
        return self.routed_expert_re.match(tensor_name) is not None

    def is_shared_expert_tensor(self, tensor_name: str) -> bool:
        return self.shared_expert_re.match(tensor_name) is not None

    def routed_expert_weight_spec(self, layer_idx: int, expert_idx: int) -> ExpertWeightSpec:

        # Qwen expert 由 gate_proj/up_proj/down_proj 三个无 bias 线性层组成。
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"

        # 用通用 ExpertWeightSpec 表达，后续 ModelRuntime/ExpertRuntime 不需要硬编码 Qwen 命名。
        return ExpertWeightSpec(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            tensors={
                "gate": f"{prefix}.gate_proj.weight",
                "up": f"{prefix}.up_proj.weight",
                "down": f"{prefix}.down_proj.weight",
            },
        )
    
    def patch_moe_block(self, model: nn.Module) -> nn.Module:
        """将原始 MoE block替换"""
    