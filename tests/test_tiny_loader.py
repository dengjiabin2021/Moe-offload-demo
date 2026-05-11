import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM

from moe_runtime import OffloadConfig, load_offloaded_model
from moe_runtime.core.module import OffloadedExperts


def _write_tiny_checkpoint(path: Path) -> None:
    config = Qwen2MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        max_position_embeddings=16,
        tie_word_embeddings=False,
        torch_dtype="float32",
    )
    path.mkdir()
    config.save_pretrained(path)

    model = Qwen2MoeForCausalLM(config)
    state = model.state_dict()
    checkpoint = {}
    for name, tensor in state.items():
        if name == "model.layers.0.mlp.experts.gate_up_proj":
            for expert_idx in range(config.num_experts):
                gate, up = tensor[expert_idx].chunk(2, dim=0)
                checkpoint[f"model.layers.0.mlp.experts.{expert_idx}.gate_proj.weight"] = gate.contiguous()
                checkpoint[f"model.layers.0.mlp.experts.{expert_idx}.up_proj.weight"] = up.contiguous()
        elif name == "model.layers.0.mlp.experts.down_proj":
            for expert_idx in range(config.num_experts):
                checkpoint[f"model.layers.0.mlp.experts.{expert_idx}.down_proj.weight"] = tensor[expert_idx].contiguous()
        else:
            checkpoint[name] = tensor.contiguous()

    shard = "model-00001-of-00001.safetensors"
    save_file(checkpoint, path / shard)
    with (path / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {"total_size": sum(t.nbytes for t in checkpoint.values())},
                "weight_map": {name: shard for name in checkpoint},
            },
            f,
        )


def test_tiny_qwen2_moe_loader_runs_forward(tmp_path):
    model_path = tmp_path / "tiny-qwen2-moe"
    _write_tiny_checkpoint(model_path)

    config = OffloadConfig(
        model_path=model_path,
        device="cpu",
        dtype="float32",
        cpu_cache_bytes="16MiB",
        gpu_cache_bytes="16MiB",
        pin_memory=False,
    )
    model = load_offloaded_model(model_path, config)

    assert isinstance(model.model.layers[0].mlp.experts, OffloadedExperts)
    assert not any(parameter.is_meta for parameter in model.parameters())

    input_ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        output = model(input_ids=input_ids)

    assert output.logits.shape == (1, 3, 32)
    stats = model._moe_offload_manager.stats()
    assert stats["disk_loads"] > 0
    assert stats["prefetches"] > 0
