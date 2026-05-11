from pathlib import Path

import pytest
import torch

from moe_runtime import OffloadConfig, load_offloaded_model
from moe_runtime.adapters.phimoe import PhiMoeAdapter
from moe_runtime.core.expert import ExpertId
from moe_runtime.core.index import TensorIndex


def test_phimoe_adapter_parses_expert_tensor_names():
    adapter = PhiMoeAdapter()
    name = "model.layers.3.block_sparse_moe.experts.12.w3.weight"

    assert adapter.is_expert_tensor(name)
    assert adapter.expert_id_for_tensor(name) == ExpertId(layer=3, expert=12)
    assert adapter.expert_tensor_names(ExpertId(3, 12)) == {
        "gate_proj": "model.layers.3.block_sparse_moe.experts.12.w1.weight",
        "up_proj": "model.layers.3.block_sparse_moe.experts.12.w3.weight",
        "down_proj": "model.layers.3.block_sparse_moe.experts.12.w2.weight",
    }


def test_local_phimoe_index_shape_if_available():
    model_path = Path("models/Phi-tiny-MoE-instruct")
    if not (model_path / "model.safetensors.index.json").exists():
        pytest.skip("Local Phi-tiny-MoE checkpoint is not available.")

    index = TensorIndex.from_safetensors_index(model_path)
    adapter = PhiMoeAdapter()
    experts = adapter.discover_experts(index)

    assert adapter.supports(model_path)
    assert len(experts) == 32 * 16
    assert len([name for name in index.names() if adapter.is_expert_tensor(name)]) == 32 * 16 * 3
    assert "model.layers.0.block_sparse_moe.gate.weight" in set(adapter.non_expert_tensor_names(index))


def test_local_phimoe_loader_replaces_experts_if_available():
    model_path = Path("models/Phi-tiny-MoE-instruct")
    if not (model_path / "model.safetensors.index.json").exists():
        pytest.skip("Local Phi-tiny-MoE checkpoint is not available.")

    config = OffloadConfig(
        model_path=model_path,
        device="cpu",
        dtype=torch.float32,
        cpu_cache_bytes="1MiB",
        gpu_cache_bytes="1MiB",
        pin_memory=False,
    )
    model = load_offloaded_model(model_path, config, attn_implementation="sdpa")

    block = model.model.layers[0].block_sparse_moe
    assert len(block.experts) == 0
    assert hasattr(block, "_moe_offload_manager")
    assert not any(buffer.is_meta for buffer in model.buffers())
