from __future__ import annotations

# re 用于匹配 Qwen checkpoint 中的 expert/shared expert 权重名。
import re
# Any 用于 transformers config 和 manifest metadata 类型标注。
from typing import Any

from pathlib import Path

import torch
# nn.Module 用于返回类型标注。
from torch import nn

from accelerate import init_empty_weights
# Qwen1.5-MoE-A2.7B 在 transformers 中复用 Qwen2MoeForCausalLM。
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM 

from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# 引入 Adapter 基类和结构化返回类型。
from .base import MoeModelAdapter
from moe_runtime.blocks import Qwen2OnDemandExperts
from moe_runtime.blocks import asyncQwen2OnDemandExperts
from moe_runtime.blocks import patch_qwen2_prefetch
from moe_runtime.prefetcher import Qwen2GatePrefetcher


class Qwen15MoeAdapter(MoeModelAdapter):
    """Qwen1.5-MoE-A2.7B 模型适配器。

    该类负责 Qwen routed expert 权重命名、专家层识别、空模型构造，
    以及串行/异步 experts 与 Qwen gate 预取 hook 的模型装配。
    """

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
        with torch.device('meta'):
            model = Qwen2MoeForCausalLM(model_config)
        return model

    def is_routed_expert_tensor(self, tensor_name: str) -> bool:
        return self.routed_expert_re.match(tensor_name) is not None

    def is_shared_expert_tensor(self, tensor_name: str) -> bool:
        return self.shared_expert_re.match(tensor_name) is not None

    def routed_expert_weight_name(self, layer_idx: int, expert_idx: int) -> dict[str,str]:

        # Qwen expert 由 gate_proj/up_proj/down_proj 三个无 bias 线性层组成。
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        
        idxs_to_name = { 
                        "gate": f"{prefix}.gate_proj.weight",
                        "up": f"{prefix}.up_proj.weight",
                        "down": f"{prefix}.down_proj.weight"
                        }
        # 用通用 ExpertWeightSpec 表达，后续 ModelRuntime/ExpertRuntime 不需要硬编码 Qwen 命名。
        return idxs_to_name
    
    def routed_expert_layers(self, model_config: Qwen2MoeConfig) -> list[int]:
        """返回 Qwen MoE 中包含 routed experts 的 decoder layer 索引。"""

        sparse_step = getattr(model_config, "decoder_sparse_step", 1)
        return list(range(0, model_config.num_hidden_layers, sparse_step))

    def patch_moe_block(
        self,
        model: nn.Module,
        model_config: Qwen2MoeConfig,
        cache_manager: Any | None = None,
        prefetcher: Any | None = None,
    ) -> nn.Module:
        """
        替换 Qwen routed experts，并按配置注册未来 gate 预取 hook。

        输入：空模型、Qwen 配置、CacheManager 和可选外部 Prefetcher。
        输出：完成 routed experts 替换和可选预取 patch 的模型。
        功能：保留原始 gate/shared expert，只替换 experts；gate 预取仅在异步模式下启用。
        """

        # Qwen2MoeForCausalLM 的 decoder layers 通常位于 model.model.layers。
        layers = model.model.layers

        # 串行/异步体系共用同一个 adapter；根据 CacheManager 暴露的接口选择专家模块。
        use_async_experts = self._use_sync_experts(cache_manager)
        expert_cls = (
            asyncQwen2OnDemandExperts
            if use_async_experts
            else Qwen2OnDemandExperts
        )

        # 遍历每一层 decoder layer，只处理包含 sparse MoE experts 的层。
        for layer_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)

            # 普通 dense MLP 没有 experts 字段，直接跳过。
            if mlp is None or not hasattr(mlp, "experts"):
                continue

            # 保留 gate/shared_expert/shared_expert_gate，只把 routed experts 替换为按需加载版本。
            mlp.experts = expert_cls(
                model_config,
                layer_idx=layer_idx,
                cache_manager=cache_manager,
                prefetcher=prefetcher,
            )

        runtime_config = getattr(cache_manager, "config", None)
        if (
            use_async_experts
            and runtime_config is not None
            and runtime_config.prefetch_policy == "gate"
        ):
            routed_expert_layers = self.routed_expert_layers(model_config)
            moe_blocks = {
                layer_idx: layers[layer_idx].mlp
                for layer_idx in routed_expert_layers
            }
            gate_prefetcher = Qwen2GatePrefetcher(
                cache_manager=cache_manager,
                moe_blocks=moe_blocks,
                routed_expert_layers=routed_expert_layers,
                prefetch_distance=runtime_config.prefetch_distance,
            )

            # Hook 只旁路读取原 MoE 输入，现有异步 experts 的请求和计算逻辑保持不变。
            patch_qwen2_prefetch(
                model=model,
                routed_expert_layers=routed_expert_layers,
                prefetcher=gate_prefetcher,
            )

        return model

    @staticmethod
    def _use_sync_experts(cache_manager: Any | None) -> bool:
        """判断当前 CacheManager 是否使用异步 expert ready/release 接口。"""

        if cache_manager is None:
            return False

        # 不绑定具体类名，后续新的异步 CacheManager 只要实现同一组接口即可复用。
        required_methods = (
            "request_expert",
            "is_expert_ready",
            "wait_any_ready",
            "get_ready_expert",
            "release_expert",
        )
        return all(hasattr(cache_manager, name) for name in required_methods)

    def materialize_meta_tensors(self, model: nn.Module, target_device: torch.device) -> None:
        """Qwen MoE 专用：重算 RoPE inv_freq 并 materialize 残留 meta tensor。"""

        for module in model.modules():
            for buf_name, buf in list(module.named_buffers(recurse=False)):
                if buf.device.type != "meta":
                    continue
                # inv_freq/original_inv_freq 是 persistent=False 的计算 buffer，需要按 Qwen2MoeRotaryEmbedding 自身逻辑重建。
                if buf_name in ("inv_freq", "original_inv_freq") and hasattr(module, "config"):
                    self._materialize_rope_buffer(module, target_device)
                    break

    @staticmethod
    def _materialize_rope_buffer(module: nn.Module, target_device: torch.device) -> None:
        """按 Qwen2MoeRotaryEmbedding.__init__ 的逻辑重建 RoPE buffer。"""

        config = module.config
        rope_type = config.rope_parameters["rope_type"]
        if rope_type == "default":
            base = config.rope_parameters["rope_theta"]
            dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            inv_freq = 1.0 / (
                base ** ( torch.arange(0, dim, 2, dtype=torch.int64).to(device=target_device, dtype=torch.float) / dim)
            )
            attention_scaling = 1.0
        else:
            rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            inv_freq, attention_scaling = rope_init_fn(config, target_device)

        module.attention_scaling = attention_scaling
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
