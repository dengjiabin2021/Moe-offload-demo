from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F

from moe_runtime.core.adapter import MoEAdapter
from moe_runtime.core.expert import ExpertId, ExpertWeights
from moe_runtime.core.index import TensorIndex
from moe_runtime.core.module import OffloadedExperts

_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


class Qwen2MoeAdapter(MoEAdapter):
    def is_expert_tensor(self, tensor_name: str) -> bool:
        return _EXPERT_RE.match(tensor_name) is not None

    def expert_id_for_tensor(self, tensor_name: str) -> ExpertId:
        match = _EXPERT_RE.match(tensor_name)
        if match is None:
            raise ValueError(f"Not a Qwen2-MoE expert tensor: {tensor_name}")
        return ExpertId(layer=int(match.group("layer")), expert=int(match.group("expert")))

    def expert_tensor_names(self, expert_id: ExpertId) -> dict[str, str]:
        prefix = f"model.layers.{expert_id.layer}.mlp.experts.{expert_id.expert}"
        return {
            "gate_proj": f"{prefix}.gate_proj.weight",
            "up_proj": f"{prefix}.up_proj.weight",
            "down_proj": f"{prefix}.down_proj.weight",
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
        from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

        for layer_idx, layer in enumerate(model.model.layers):
            mlp = getattr(layer, "mlp", None)
            if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
                continue
            mlp.layer_idx = layer_idx
            mlp._moe_offload_manager = manager
            mlp.forward = _qwen2_sparse_moe_forward_with_prefetch.__get__(mlp, mlp.__class__)
            old = mlp.experts
            mlp.experts = OffloadedExperts(
                layer_idx=layer_idx,
                num_experts=old.num_experts,
                hidden_dim=old.hidden_dim,
                intermediate_dim=old.intermediate_dim,
                act_fn=old.act_fn,
                manager=manager,
            )

    def non_expert_tensor_names(self, index: TensorIndex) -> Iterable[str]:
        return (name for name in index.names() if not self.is_expert_tensor(name))

    @classmethod
    def supports(cls, model_path: str | Path) -> bool:
        config_path = Path(model_path) / "config.json"
        if not config_path.exists():
            return False
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("model_type") == "qwen2_moe" or "Qwen2MoeForCausalLM" in config.get("architectures", [])


def _qwen2_sparse_moe_forward_with_prefetch(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
    _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

    expert_ids = {
        ExpertId(self.layer_idx, int(expert))
        for expert in selected_experts.detach().reshape(-1).cpu().tolist()
    }
    future = self._moe_offload_manager.prefetch_async(expert_ids)

    shared_expert_output = self.shared_expert(hidden_states_reshaped)
    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

    if future is not None:
        future.result()
    expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
    expert_output = expert_output + shared_expert_output
    return expert_output.reshape(batch_size, sequence_length, hidden_dim)
