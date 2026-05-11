from moe_runtime.adapters.qwen2_moe import Qwen2MoeAdapter
from moe_runtime.core.expert import ExpertId


def test_qwen2_adapter_parses_expert_tensor_names():
    adapter = Qwen2MoeAdapter()
    name = "model.layers.3.mlp.experts.42.up_proj.weight"

    assert adapter.is_expert_tensor(name)
    assert adapter.expert_id_for_tensor(name) == ExpertId(layer=3, expert=42)
    assert adapter.expert_tensor_names(ExpertId(3, 42)) == {
        "gate_proj": "model.layers.3.mlp.experts.42.gate_proj.weight",
        "up_proj": "model.layers.3.mlp.experts.42.up_proj.weight",
        "down_proj": "model.layers.3.mlp.experts.42.down_proj.weight",
    }


def test_shared_expert_is_not_offloaded_expert():
    adapter = Qwen2MoeAdapter()
    assert not adapter.is_expert_tensor("model.layers.0.mlp.shared_expert.up_proj.weight")
