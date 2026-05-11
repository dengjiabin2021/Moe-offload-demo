from pathlib import Path

import pytest

from moe_runtime.adapters.qwen2_moe import Qwen2MoeAdapter
from moe_runtime.core.index import TensorIndex


def test_local_qwen15_moe_index_shape_if_available():
    model_path = Path("models/Qwen1.5-MoE-A2.7B")
    if not (model_path / "model.safetensors.index.json").exists():
        pytest.skip("Local Qwen1.5-MoE-A2.7B checkpoint is not available.")

    index = TensorIndex.from_safetensors_index(model_path)
    adapter = Qwen2MoeAdapter()
    experts = adapter.discover_experts(index)

    assert len(experts) == 24 * 60
    assert len([name for name in index.names() if adapter.is_expert_tensor(name)]) == 24 * 60 * 3
    assert "model.layers.0.mlp.shared_expert.up_proj.weight" in set(adapter.non_expert_tensor_names(index))
