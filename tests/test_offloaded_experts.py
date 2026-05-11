import torch
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts

from moe_runtime.core.expert import ExpertId, ExpertWeights
from moe_runtime.core.module import OffloadedExperts


class StaticManager:
    def __init__(self, weights):
        self.weights = weights
        self.prefetched = []

    def prefetch(self, expert_ids):
        self.prefetched.extend(expert_ids)

    def get(self, expert_id):
        return self.weights[expert_id]


def test_offloaded_experts_match_transformers_qwen2_experts():
    torch.manual_seed(7)
    config = Qwen2MoeConfig(
        hidden_size=8,
        moe_intermediate_size=4,
        num_experts=3,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    reference = Qwen2MoeExperts(config)
    reference.gate_up_proj.data.normal_()
    reference.down_proj.data.normal_()

    weights = {}
    for expert_idx in range(config.num_experts):
        gate, up = reference.gate_up_proj[expert_idx].chunk(2, dim=0)
        weights[ExpertId(0, expert_idx)] = ExpertWeights(
            gate_proj=gate.detach().clone(),
            up_proj=up.detach().clone(),
            down_proj=reference.down_proj[expert_idx].detach().clone(),
        )
    manager = StaticManager(weights)
    offloaded = OffloadedExperts(
        layer_idx=0,
        num_experts=config.num_experts,
        hidden_dim=config.hidden_size,
        intermediate_dim=config.moe_intermediate_size,
        act_fn=reference.act_fn,
        manager=manager,
    )

    hidden_states = torch.randn(5, config.hidden_size)
    top_k_index = torch.tensor([[0, 1], [1, 2], [0, 2], [2, 1], [0, 1]])
    top_k_weights = torch.softmax(torch.randn(5, 2), dim=-1)

    expected = reference(hidden_states, top_k_index, top_k_weights)
    actual = offloaded(hidden_states, top_k_index, top_k_weights)

    torch.testing.assert_close(actual, expected)
