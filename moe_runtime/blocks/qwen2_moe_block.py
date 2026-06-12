from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN


class Qwen2OnDemandExperts(nn.Module):
    """Qwen1.5-MoE 的按需加载专家集合。

    该模块替换 transformers 原始 Qwen2MoeExperts。外层 Qwen2MoeSparseMoeBlock
    仍负责 router 和 top-k 选择；这里仅根据命中的 expert id 向 CacheManager
    提交请求、等待 ready，并执行专家计算。
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        cache_manager: Any | None = None,
        prefetcher: Any | None = None,
    ) -> None:
        super().__init__()

        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]
        self.layer_idx = layer_idx
        self.cache_manager = cache_manager
        self.prefetcher = prefetcher

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """执行 routed experts 计算。

        输入参数与 transformers 官方 Qwen2MoeExperts.forward 保持一致。
        """

        if self.cache_manager is None:
            raise RuntimeError("Qwen2OnDemandExperts.cache_manager 未设置，无法按需获取专家权重。")

        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index,
                num_classes=self.num_experts,
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        active_experts: list[int] = []
        for expert_idx in expert_hit:
            expert_id = int(expert_idx[0].item())
            if expert_id == self.num_experts:
                continue
            active_experts.append(expert_id)

        ready: list[int] = []
        pending: list[int] = []
        for expert_id in active_experts:
            self.cache_manager.request_expert(
                layer_idx=self.layer_idx,
                expert_idx=expert_id,
            )
            if self.cache_manager.is_expert_ready(layer_idx=self.layer_idx, expert_idx=expert_id):
                ready.append(expert_id)
            else:
                pending.append(expert_id)

        while ready or pending:
            while ready:
                expert_id = ready.pop()
                self._compute_ready_expert(
                    expert_id=expert_id,
                    expert_mask=expert_mask,
                    hidden_states=hidden_states,
                    top_k_weights=top_k_weights,
                    final_hidden_states=final_hidden_states,
                )
                self.cache_manager.release_expert(
                    layer_idx=self.layer_idx,
                    expert_idx=expert_id,
                )

            for expert_id in list(pending):
                if self.cache_manager.is_expert_ready(layer_idx=self.layer_idx, expert_idx=expert_id):
                    ready.append(expert_id)
                    pending.remove(expert_id)

            if not ready and pending:
                expert_id = self.cache_manager.wait_any_ready(
                    layer_idx=self.layer_idx,
                    expert_ids=pending,
                )
                ready.append(expert_id)
                pending.remove(expert_id)

        return final_hidden_states

    def _compute_ready_expert(
        self,
        *,
        expert_id: int,
        expert_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        top_k_weights: torch.Tensor,
        final_hidden_states: torch.Tensor,
    ) -> None:
        """计算一个已经由 CacheManager 准备好的 expert。"""

        top_k_pos, token_idx = torch.where(expert_mask[expert_id])
        current_state = hidden_states[token_idx]

        expert_tensor_dict = self.cache_manager.get_ready_expert(
            layer_idx=self.layer_idx,
            expert_idx=expert_id,
        )

        gate = nn.functional.linear(current_state, expert_tensor_dict["gate"])
        up = nn.functional.linear(current_state, expert_tensor_dict["up"])
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = nn.functional.linear(
            current_hidden_states,
            expert_tensor_dict["down"],
        )
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

        final_hidden_states.index_add_(
            0,
            token_idx,
            current_hidden_states.to(final_hidden_states.dtype),
        )
