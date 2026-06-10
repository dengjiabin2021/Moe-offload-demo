from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN


class DeepseekV2OnDemandExperts(nn.Module):
    """DeepSeek-V2 的按需加载专家集合。

    该模块用于替换 transformers 原始的 DeepseekV2Experts。外层 DeepseekV2Moe
    仍然负责 router、top-k 选择和 shared experts 计算；这里仅处理 routed experts。
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        cache_manager: Any | None = None,
        prefetcher: Any | None = None,
    ) -> None:
        super().__init__()

        # DeepSeek-V2 配置中的 routed expert 数量。
        self.num_experts = config.n_routed_experts

        # hidden 维度和专家中间层维度，仅保留为结构信息和调试入口。
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        # 专家激活函数，与 transformers 官方实现保持一致。
        self.act_fn = ACT2FN[config.hidden_act]

        # 当前 experts 所在的 transformer layer 编号。
        self.layer_idx = layer_idx

        # 约定接口：get_expert(layer_idx: int, expert_idx: int) -> dict[str, Tensor]
        self.cache_manager = cache_manager

        # 预取接口暂不在本模块内触发，保留给后续策略接入。
        self.prefetcher = prefetcher

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """执行 routed experts 计算。

        输入参数与 transformers 原始 DeepseekV2Experts.forward 保持一致，因此外层
        DeepseekV2Moe 不需要修改。
        """

        if self.cache_manager is None:
            raise RuntimeError("DeepseekV2OnDemandExperts.cache_manager 未设置，无法按需获取专家权重。")

        # 输出形状与输入 hidden_states 一致，用于累加所有命中专家的贡献。
        final_hidden_states = torch.zeros_like(hidden_states)

        # 构造 expert 命中表，形状为 [num_experts, top_k, num_tokens]。
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index,
                num_classes=self.num_experts,
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        if self.cache_manager is None:
            raise RuntimeError("Qwen2MoeExperts.cache_manager 未设置，无法按需获取专家权重。")
    
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue

            # 找到当前 expert 负责的 token，以及对应的 top-k 槽位。
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # CacheManager 返回 gate/up/down 三个角色的专家权重。
            expert_tensor_dict = self.cache_manager.get_expert(
                layer_idx=self.layer_idx,
                expert_idx=expert_idx,
            )

            gate = nn.functional.linear(current_state, expert_tensor_dict["gate"])
            up = nn.functional.linear(current_state, expert_tensor_dict["up"])
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(
                current_hidden_states,
                expert_tensor_dict["down"],
            )

            # 乘以 router 给出的 top-k 权重。
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

            # 同一个 token 可能经过多个 expert，这里累加所有 expert 的贡献。
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states
