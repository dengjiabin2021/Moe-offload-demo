from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from moe_runtime.core.adapter import MoEAdapter
from moe_runtime.core.expert import ExpertId, ExpertWeights
from moe_runtime.core.index import TensorIndex

_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>w1|w2|w3)\.weight$"
)


class PhiMoeAdapter(MoEAdapter):
    def is_expert_tensor(self, tensor_name: str) -> bool:
        return _EXPERT_RE.match(tensor_name) is not None

    def expert_id_for_tensor(self, tensor_name: str) -> ExpertId:
        match = _EXPERT_RE.match(tensor_name)
        if match is None:
            raise ValueError(f"Not a Phi-MoE expert tensor: {tensor_name}")
        return ExpertId(layer=int(match.group("layer")), expert=int(match.group("expert")))

    def expert_tensor_names(self, expert_id: ExpertId) -> dict[str, str]:
        prefix = f"model.layers.{expert_id.layer}.block_sparse_moe.experts.{expert_id.expert}"
        return {
            "gate_proj": f"{prefix}.w1.weight",
            "up_proj": f"{prefix}.w3.weight",
            "down_proj": f"{prefix}.w2.weight",
        }

    def discover_experts(self, index: TensorIndex) -> set[ExpertId]:
        return {
            self.expert_id_for_tensor(name)
            for name in index.names()
            if self.is_expert_tensor(name)
        }

    def load_expert_weights(self, index: TensorIndex, expert_id: ExpertId) -> ExpertWeights:
        names = self.expert_tensor_names(expert_id)
        return ExpertWeights(
            gate_proj=index.load_tensor(names["gate_proj"], device="cpu"),
            up_proj=index.load_tensor(names["up_proj"], device="cpu"),
            down_proj=index.load_tensor(names["down_proj"], device="cpu"),
        )

    def replace_experts(self, model, manager) -> None:
        for layer_idx, layer in enumerate(model.model.layers):
            block = getattr(layer, "block_sparse_moe", None)
            if block is None or not hasattr(block, "experts"):
                continue
            block.act_fn = block.experts[0].act_fn
            block.layer_idx = layer_idx
            block._moe_offload_manager = manager
            block.experts = nn.ModuleList()
            block.forward = _phimoe_sparse_moe_forward_with_offload.__get__(block, block.__class__)

    def non_expert_tensor_names(self, index: TensorIndex) -> Iterable[str]:
        return (name for name in index.names() if not self.is_expert_tensor(name))

    @classmethod
    def supports(cls, model_path: str | Path) -> bool:
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return False
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model_type") == "phimoe" or "PhiMoEForCausalLM" in config.get("architectures", [])


def _phimoe_sparse_moe_forward_with_offload(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.input_jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise)

    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)

    sparsemixer = self.__class__.forward.__globals__["sparsemixer"]
    routing_weights, selected_experts = sparsemixer(
        router_logits,
        top_k=self.top_k,
        jitter_eps=self.router_jitter_noise,
        training=self.training,
    )

    expert_ids = {
        ExpertId(self.layer_idx, int(expert))
        for expert in selected_experts.detach().reshape(-1).cpu().tolist()
    }
    future = self._moe_offload_manager.prefetch_async(expert_ids)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

    if future is not None:
        future.result()

    for expert_idx in range(self.num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        if top_x.shape[0] == 0:
            continue

        current_state = hidden_states[top_x]
        weights = self._moe_offload_manager.get(ExpertId(self.layer_idx, expert_idx))
        current_hidden_states = F.linear(current_state, weights.gate_proj)
        current_hidden_states = self.act_fn(current_hidden_states) * F.linear(current_state, weights.up_proj)
        current_hidden_states = F.linear(current_hidden_states, weights.down_proj)
        current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits
