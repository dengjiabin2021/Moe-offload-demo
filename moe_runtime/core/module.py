from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .expert import ExpertId


class OffloadedExperts(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        act_fn,
        manager,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.act_fn = act_fn
        self.manager = manager

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        expert_ids = [ExpertId(self.layer_idx, int(item[0])) for item in expert_hit]

        for expert_id in expert_ids:
            top_k_pos, token_idx = torch.where(expert_mask[expert_id.expert])
            current_state = hidden_states[token_idx]
            weights = self.manager.get(expert_id)

            gate = F.linear(current_state, weights.gate_proj)
            up = F.linear(current_state, weights.up_proj)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, weights.down_proj)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
