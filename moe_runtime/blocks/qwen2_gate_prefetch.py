from __future__ import annotations

from typing import Any

import torch
from torch import nn


class Qwen2GatePrefetcher:
    """Qwen2-MoE 的单层 gate 专家预取器。

    该对象由当前 routed experts 模块持有，只预测当前层之后有限个真实
    routed expert 层。它不注册 hook，也不改变当前层的真实路由结果。
    """

    def __init__(
        self,
        *,
        cache_manager: Any,
        model: nn.Module,
        layer_idx: int,
        routed_expert_layers: list[int],
        prefetch_distance: int,
    ) -> None:
        """
        初始化当前 Qwen routed expert 层的未来 gate 预测范围。

        输入：异步 CacheManager、完整模型、当前层号、真实专家层列表和预取距离。
        输出：无。
        功能：从模型中取得截断后的未来 MoE block；模型末尾不会越界预取。
        """

        if layer_idx not in routed_expert_layers:
            raise ValueError(f"Layer {layer_idx} is not a routed expert layer.")
        if prefetch_distance < 1:
            raise ValueError("prefetch_distance must be greater than or equal to 1.")

        current_position = routed_expert_layers.index(layer_idx)
        self.layer_idx = layer_idx
        self.target_layer_indices = list(
            routed_expert_layers[
                current_position + 1 : current_position + 1 + prefetch_distance
            ]
        )
        self.cache_manager = cache_manager
        self.target_moe_blocks = {
            target_layer_idx: model.model.layers[target_layer_idx].mlp
            for target_layer_idx in self.target_layer_indices
        }

    @torch.no_grad()
    def prefetch(self, hidden_states: torch.Tensor) -> None:
        """
        使用当前层 hidden_states 预测后续 Qwen routed expert。

        输入：当前 Qwen experts 收到的 token hidden_states。
        输出：无。
        功能：跳过非单 token 输入，复用目标层 gate 并提交截断后的预取请求。
        """

        # experts 输入已被展平，只在单 token decode 的常用路径执行预测，避免 prefill 污染缓存。
        token_count = hidden_states.numel() // hidden_states.shape[-1]
        if token_count != 1 or not self.target_layer_indices:
            return

        gate_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        # 预取接口会把每批新增任务插到队首，因此从远到近提交可让最近目标层最终位于最前。
        for target_layer_idx in reversed(self.target_layer_indices):
            target_moe = self.target_moe_blocks[target_layer_idx]
            # Qwen2MoeTopKRouter 已完成 softmax、top-k 和可选归一化。
            _, _, selected_experts = target_moe.gate(gate_hidden_states)

            expert_ids = torch.unique(selected_experts).tolist()
            self.cache_manager.prefetch_experts(
                layer_idx=target_layer_idx,
                expert_ids=expert_ids,
            )
