from __future__ import annotations

from typing import Any

import torch
from torch import nn


class DeepseekV2GatePrefetcher:
    """DeepSeek-V2 的跨层 gate 预测预取器。

    该类按 Adapter 给出的真实 MoE 层顺序保存 DeepseekV2Moe 引用，并在 decode
    阶段复用目标层 gate 和 route_tokens_to_experts 预测未来 top-k expert。
    """

    def __init__(
        self,
        *,
        cache_manager: Any,
        moe_blocks: dict[int, nn.Module],
        routed_expert_layers: list[int],
        prefetch_distance: int,
    ) -> None:
        """
        初始化 DeepSeek-V2 gate 预测所需的模型层和调度依赖。

        输入：异步 CacheManager、MoE block 映射、真实专家层列表和预取距离。
        输出：无。
        功能：按 Adapter 提供的专家层顺序建立位置索引，自动跳过普通 MLP 层。
        """

        self.cache_manager = cache_manager
        self.moe_blocks = moe_blocks
        self.routed_expert_layers = list(routed_expert_layers)
        self.prefetch_distance = prefetch_distance
        self._layer_positions = {
            layer_idx: position
            for position, layer_idx in enumerate(self.routed_expert_layers)
        }
        self._is_decode = False

    def set_decode_mode(self, is_decode: bool) -> None:
        """
        更新当前模型 forward 所处的生成阶段。

        输入：当前 forward 是否为 decode 阶段。
        输出：无。
        功能：Prefill 阶段禁止未来 gate 预测，避免多 token 预取造成缓存污染。
        """

        self._is_decode = is_decode

    @torch.no_grad()
    def prefetch(self, current_layer_idx: int, hidden_states: torch.Tensor) -> None:
        """
        使用当前层 hidden_states 预测后续 DeepSeek-V2 MoE 层的专家。

        输入：当前 MoE 层编号，以及该层 gate 将要使用的 hidden_states。
        输出：无。
        功能：Decode 时复用目标层原始路由方法，并把预测 top-k 提交到低优先级预取队列。
        """

        if not self._is_decode:
            return

        current_position = self._layer_positions[current_layer_idx]
        target_layers = self.routed_expert_layers[
            current_position + 1 : current_position + 1 + self.prefetch_distance
        ]

        for target_layer_idx in target_layers:
            target_moe = self.moe_blocks[target_layer_idx]

            # DeepSeek 的路由方法可能包含分组约束，必须复用目标层原始实现。
            router_logits = nn.functional.linear(
                hidden_states.type(torch.float32),
                target_moe.gate.weight.type(torch.float32),
            )
            selected_experts, _ = target_moe.route_tokens_to_experts(router_logits)
            expert_ids = torch.unique(selected_experts).tolist()
            self.cache_manager.prefetch_experts(
                layer_idx=target_layer_idx,
                expert_ids=expert_ids,
            )
