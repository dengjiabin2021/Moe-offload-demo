from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.activations import ACT2FN


class asyncDeepseekV2OnDemandExperts(nn.Module):
    """DeepSeek-V2 的异步按需加载专家集合。

    该模块用于替换 transformers 原始 DeepseekV2Experts。外层 DeepseekV2Moe
    仍然负责 router、top-k 选择和 shared experts 计算；这里仅负责 routed experts
    的异步请求、ready 等待、权重读取和专家计算。
    """

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        cache_manager: Any | None = None,
        prefetcher: Any | None = None,
    ) -> None:
        super().__init__()

        # DeepSeek-V2 配置中的 routed expert 数量，用于构造 one_hot 命中表。
        self.num_experts = config.n_routed_experts

        # hidden 维度和专家中间层维度仅作为结构信息保留，真实权重由 CacheManager 提供。
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        # 激活函数必须和 transformers 原始 DeepseekV2Experts 保持一致。
        self.act_fn = ACT2FN[config.hidden_act]

        # 当前 experts 模块所在的 decoder layer 编号，用于定位 layer_idx + expert_idx。
        self.layer_idx = layer_idx

        # 模型级异步缓存管理器，负责 expert 请求、ready 查询、权重 view 获取和 release。
        self.cache_manager = cache_manager

        # 预取器接口暂时保留，后续可接入跨层/跨 token 的预取策略。
        self.prefetcher = prefetcher

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """执行 routed experts 计算。

        输入参数与 transformers 官方 DeepseekV2Experts.forward 保持一致。
        """

        if self.cache_manager is None:
            raise RuntimeError("asyncDeepseekV2OnDemandExperts.cache_manager 未设置，无法按需获取专家权重。")

        # final_hidden_states 与输入同形状，用来累加所有 top-k experts 的输出。
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            # top_k_index: [num_tokens, top_k] -> expert_mask: [num_tokens, top_k, num_experts]。
            expert_mask = torch.nn.functional.one_hot(
                top_k_index,
                num_classes=self.num_experts,
            )

            # 调整为 [num_experts, top_k, num_tokens]，便于按 expert 找到负责的 token。
            expert_mask = expert_mask.permute(2, 1, 0)

            # 找出本次 forward 实际命中的 expert，未命中的 expert 不发起请求也不计算。
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # active_experts 记录当前层本次 forward 需要计算的 expert。
        active_experts: list[int] = []
        for expert_idx in expert_hit:
            expert_id = expert_idx[0]
            if expert_id == self.num_experts:
                continue
            active_experts.append(expert_id)

        # ready 表示 GPU 上已经可直接计算的 expert。
        ready: list[int] = []

        # pending 表示当前还不能计算，需要 CacheManager 后台继续加载的 expert。
        pending: list[int] = []
        for expert_id in active_experts:
            # 如果 expert 已经在 GPU used slot 中，CacheManager 会将其保护为 ready。
            if self.cache_manager.is_expert_ready(layer_idx=self.layer_idx, expert_idx=expert_id):
                ready.append(expert_id)
            else:
                # 不 ready 的 expert 只提交一次请求；后续磁盘/CPU/GPU 调度由 CacheManager 接管。
                self.cache_manager.request_expert(
                    layer_idx=self.layer_idx,
                    expert_idx=expert_id,
                )
                pending.append(expert_id)

        # 只要还有可计算或待加载的 expert，就持续推进本层 routed experts 计算。
        while ready or pending:
            # 优先计算已经 ready 的 expert，让 GPU 计算和后台加载 pending experts 尽量重叠。
            while ready:
                expert_id = ready.pop()
                self._compute_ready_expert(
                    expert_id=expert_id,
                    expert_mask=expert_mask,
                    hidden_states=hidden_states,
                    top_k_weights=top_k_weights,
                    final_hidden_states=final_hidden_states,
                )
                # 计算 kernel 已提交后立即 release；CacheManager 通过 CUDA event 判断何时真正可覆盖 slot。
                self.cache_manager.release_expert(
                    layer_idx=self.layer_idx,
                    expert_idx=expert_id,
                )

            # 每轮 ready 计算结束后，重新检查 pending 中哪些 expert 已经由后台加载完成。
            for expert_id in list(pending):
                if self.cache_manager.is_expert_ready(layer_idx=self.layer_idx, expert_idx=expert_id):
                    ready.append(expert_id)
                    pending.remove(expert_id)

            # 如果当前没有任何 ready expert，但仍有 pending，就阻塞等待任意一个 expert ready。
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

        # 找出当前 expert 负责的 token 下标，以及这些 token 在 top-k 中的位置。
        top_k_pos, token_idx = torch.where(expert_mask[expert_id])

        # 只取当前 expert 实际需要处理的 token hidden states。
        current_state = hidden_states[token_idx]

        # 从 CacheManager 获取当前 expert 的 GPU 权重 view；此时对应 slot 处于 protected 状态。
        expert_tensor_dict = self.cache_manager.get_ready_expert(
            layer_idx=self.layer_idx,
            expert_idx=expert_id,
        )

        # DeepSeek-V2 routed expert 是 SwiGLU 结构：act(gate_proj(x)) * up_proj(x)。
        gate = nn.functional.linear(current_state, expert_tensor_dict["gate"])
        up = nn.functional.linear(current_state, expert_tensor_dict["up"])
        current_hidden_states = self.act_fn(gate) * up

        # down_proj 将 expert 中间维度投回 hidden 维度。
        current_hidden_states = nn.functional.linear(
            current_hidden_states,
            expert_tensor_dict["down"],
        )

        # 乘以 router 给出的 top-k 权重，保持与 transformers 原始实现一致。
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

        # 同一个 token 可能命中多个 expert，这里把当前 expert 的贡献累加回最终输出。
        final_hidden_states.index_add_(
            0,
            token_idx,
            current_hidden_states.to(final_hidden_states.dtype),
        )
