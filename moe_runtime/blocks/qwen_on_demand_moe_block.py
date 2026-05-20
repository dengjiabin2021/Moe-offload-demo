from __future__ import annotations

# Any 用于 expert_manager 的接口占位；后续 CacheManager/PrefetchManager 定好后再收紧类型。
from typing import Any

# torch 用于张量 reshape、topk、softmax 等 MoE forward 基础操作。
import torch
# F 提供 softmax/sigmoid 等函数，保持和 transformers Qwen 实现一致。
import torch.nn.functional as F
# nn.Module 是所有 PyTorch 模块基类。
from torch import nn


class QwenOnDemandMoeBlock(nn.Module):
    """Qwen1.5-MoE 的按需专家 MoE block。

    这个模块用于替换 transformers 原始的 Qwen sparse MLP。
    为了尽量复用 transformers 的 model.forward / generate 调用链，本模块必须保持：
    - forward(hidden_states) 的外部签名不变
    - gate / experts / shared_expert / shared_expert_gate 这些关键子模块名尽量保留
    - shared expert 和 router 仍然作为常驻模块参与计算
    - routed experts 的执行通过 expert_manager 委托出去
    """

    def __init__(
        self,
        *,
        layer_idx: int,
        original_mlp: nn.Module | None = None,
        expert_manager: Any | None = None,
        gate: nn.Module | None = None,
        experts: nn.Module | None = None,
        shared_expert: nn.Module | None = None,
        shared_expert_gate: nn.Module | None = None,
        top_k: int | None = None,
        norm_topk_prob: bool | None = None,
    ) -> None:
        super().__init__()

        # 当前 MoE block 所在 transformer layer 编号。
        self.layer_idx = layer_idx

        # expert_manager 是模型级专家权重管理/执行入口。
        # 它后续负责根据 layer_idx/expert_idx 返回权重，或直接执行 routed expert。
        self.expert_manager = expert_manager

        # 如果传入原始 mlp，就从原始 mlp 上取出 Qwen 需要保留的子模块。
        # 这样替换 layer.mlp 后，router/shared expert 的模块结构仍然完整。
        if original_mlp is not None:
            gate = getattr(original_mlp, "gate", gate)
            experts = getattr(original_mlp, "experts", experts)
            shared_expert = getattr(original_mlp, "shared_expert", shared_expert)
            shared_expert_gate = getattr(original_mlp, "shared_expert_gate", shared_expert_gate)
            top_k = getattr(original_mlp, "top_k", top_k)
            norm_topk_prob = getattr(original_mlp, "norm_topk_prob", norm_topk_prob)

        # router / gate 模块，负责计算 token 到 experts 的路由。
        self.gate = gate

        # 保留原始 experts 属性名，避免模型结构检查或命名逻辑失效。
        # forward 中不会直接使用它，真正 routed expert 由 expert_manager 处理。
        self.experts = experts

        # Qwen 的 shared expert 常驻 GPU，继续复用原始模块。
        self.shared_expert = shared_expert

        # Qwen 的 shared expert gate 常驻 GPU，继续复用原始模块。
        self.shared_expert_gate = shared_expert_gate

        # top-k 路由数量；Qwen1.5-MoE-A2.7B 通常是 4。
        self.top_k = top_k

        # 是否对 top-k 概率重新归一化。
        self.norm_topk_prob = bool(norm_topk_prob) if norm_topk_prob is not None else False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """执行 Qwen MoE block forward。

        这个签名必须和 transformers 原始 Qwen sparse MLP 兼容，
        这样上层 DecoderLayer / model.generate 才能继续复用。
        """

        # Qwen MoE block 输入是 [batch, seq, hidden_dim]。
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # router 和 expert 计算按 token 展平处理。
        hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)

        # 先计算 shared expert。shared expert 是常驻权重，不走按需加载。
        shared_expert_output = self._run_shared_expert(hidden_states_reshaped)

        # router 选择 routed experts。
        _router_logits, routing_weights, selected_experts = self._run_router(hidden_states_reshaped)

        # routed expert 输出由 expert_manager 负责。
        expert_output = self._run_routed_experts(
            hidden_states=hidden_states_reshaped,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
        )

        # routed expert 和 shared expert 输出相加，再还原形状。
        output = expert_output + shared_expert_output
        return output.reshape(batch_size, sequence_length, hidden_dim)

    def _run_router(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """运行 Qwen router，并兼容不同 transformers 版本的 gate 返回格式。"""

        # gate 是必须保留的原始 router 模块。
        if self.gate is None:
            raise RuntimeError("QwenOnDemandMoeBlock.gate 未初始化。")

        # transformers 5.x 的 Qwen gate 通常返回 (_, routing_weights, selected_experts)。
        gate_output = self.gate(hidden_states)
        if isinstance(gate_output, tuple):
            router_logits, routing_weights, selected_experts = gate_output
            return router_logits, routing_weights, selected_experts

        # 旧版 gate 可能只返回 router logits，此时在这里做 softmax + topk。
        router_logits = gate_output
        if self.top_k is None:
            raise RuntimeError("gate 只返回 logits 时，QwenOnDemandMoeBlock.top_k 不能为空。")

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(hidden_states.dtype)
        return router_logits, routing_weights, selected_experts

    def _run_shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """运行 Qwen shared expert。"""

        # shared expert 和 shared expert gate 都应在常驻权重加载阶段完成加载。
        if self.shared_expert is None:
            raise RuntimeError("QwenOnDemandMoeBlock.shared_expert 未初始化。")
        if self.shared_expert_gate is None:
            raise RuntimeError("QwenOnDemandMoeBlock.shared_expert_gate 未初始化。")

        shared_expert_output = self.shared_expert(hidden_states)
        shared_gate = F.sigmoid(self.shared_expert_gate(hidden_states))
        return shared_gate * shared_expert_output

    def _run_routed_experts(
        self,
        *,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """运行 routed experts。

        当前 block 不直接加载权重，也不直接管理缓存。
        它只把 layer_idx、hidden_states、selected_experts、routing_weights 交给 expert_manager。
        """

        if self.expert_manager is None:
            raise RuntimeError("QwenOnDemandMoeBlock.expert_manager 未初始化，无法执行 routed experts。")

        # 第一优先级：expert_manager 直接提供 run_experts 接口。
        if hasattr(self.expert_manager, "run_experts"):
            return self.expert_manager.run_experts(
                layer_idx=self.layer_idx,
                hidden_states=hidden_states,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
            )

        # 第二优先级：如果后续命名为 run_layer_experts，也做兼容。
        if hasattr(self.expert_manager, "run_layer_experts"):
            return self.expert_manager.run_layer_experts(
                layer_idx=self.layer_idx,
                hidden_states=hidden_states,
                selected_experts=selected_experts,
                routing_weights=routing_weights,
            )

        raise TypeError("expert_manager 必须实现 run_experts(...) 或 run_layer_experts(...)。")

    def set_expert_manager(self, expert_manager: Any) -> None:
        """注入模型级 expert manager。"""

        self.expert_manager = expert_manager

    def extra_repr(self) -> str:
        """打印模块调试信息时显示层编号和路由参数。"""

        return (
            f"layer_idx={self.layer_idx}, "
            f"top_k={self.top_k}, "
            f"norm_topk_prob={self.norm_topk_prob}"
        )
