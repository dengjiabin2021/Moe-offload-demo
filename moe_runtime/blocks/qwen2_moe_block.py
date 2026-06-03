from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN


class Qwen2OnDemandExperts(nn.Module):
    """Qwen1.5-MoE 的按需加载专家集合。

    这个模块用于替换 transformers 原版 Qwen2MoeExperts。
    原版会在模块里持有所有 routed expert 的大参数；这里不持有专家参数，
    只在 forward 中根据 router 结果向 cache_manager 请求当前命中的专家权重。
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        cache_manager: Any | None = None,
        prefetcher: Any | None = None,
    ) -> None:
        super().__init__()

        # Qwen 配置中的 routed expert 数量。
        self.num_experts = config.num_experts

        # hidden 维度，例如 Qwen1.5-MoE-A2.7B 是 2048。
        self.hidden_dim = config.hidden_size

        # routed expert 的中间层维度，例如 Qwen1.5-MoE-A2.7B 是 1408。
        self.intermediate_dim = config.moe_intermediate_size

        # 专家激活函数，与 transformers 官方实现保持一致。
        self.act_fn = ACT2FN[config.hidden_act]


        # 当前 experts 所在的 transformer layer 编号，用于定位 layer_id + expert_id。
        self.layer_idx = layer_idx

        # 模型级缓存管理器，后续负责加载、命中、淘汰和显存布局。
        # 约定接口：get_expert(layer_idx: int, expert_idx: int) -> expert
        self.cache_manager = cache_manager

        # 模型级预取器，后续负责根据当前或未来路由结果触发预取。
        # 约定接口：prefetch(layer_idx: int, expert_ids: list[int]) -> None
        self.prefetcher = prefetcher

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor, # [num_tokens, top_k]
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """执行 routed experts 计算。

        输入参数和 transformers 官方 Qwen2MoeExperts.forward 保持一致，
        因此外层 Qwen2MoeSparseMoeBlock 不需要修改。
        """

        # 输出张量与输入 hidden_states 同形状，用于累加 top-k experts 的结果。
        final_hidden_states = torch.zeros_like(hidden_states)

        # 根据 router 选中的 expert id 构造专家命中表，形状变为 [num_experts, top_k, num_tokens]。
        with torch.no_grad():
            # 形状变为 [num_tokens, top_k, nums_experts]
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            # 形状变为 [num_experts, top_k, num_tokens]
            expert_mask = expert_mask.permute(2, 1, 0)
            # [num_active_experts, 1]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()


        # 当前版本只定义 cache_manager 接口，具体加载、缓存和显存管理后续实现。
        if self.cache_manager is None:
            raise RuntimeError("Qwen2MoeExperts.cache_manager 未设置，无法按需获取专家权重。")

        # 逐个执行本次 forward 中实际命中的 expert，逻辑与官方 Qwen2MoeExperts 基本一致。
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue

            # 找到当前 expert 负责的 token，以及对应 top-k 槽位。
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            # 取出当前 expert 需要处理的 token hidden states。
            current_state = hidden_states[token_idx]

            # 向 cache_manager 请求当前层、当前专家的权重或可调用专家对象
            expert_tensor_dict = self.cache_manager.get_expert(layer_idx=self.layer_idx, expert_idx=expert_idx)

            gate = nn.functional.linear(current_state, expert_tensor_dict["gate"])
            up = nn.functional.linear(current_state, expert_tensor_dict["up"])
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, expert_tensor_dict["down"])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))


            # 乘以 router 给出的 top-k 权重。
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

            # 同一个 token 可能经过多个 expert，这里累加所有 expert 的贡献。
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
