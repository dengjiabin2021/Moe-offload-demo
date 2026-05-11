from __future__ import annotations

import os
from pathlib import Path
import sys
import types
from importlib.machinery import ModuleSpec
from typing import Any

os.environ.setdefault("HF_MODULES_CACHE", str(Path.cwd() / ".cache" / "huggingface" / "modules"))

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from moe_runtime.adapters import PhiMoeAdapter, Qwen2MoeAdapter
from moe_runtime.core.adapter import MoEAdapter
from moe_runtime.core.config import OffloadConfig
from moe_runtime.core.manager import ExpertManager
from moe_runtime.core.index import TensorIndex


def resolve_adapter(model_path: str | Path) -> MoEAdapter:
    adapters = [Qwen2MoeAdapter, PhiMoeAdapter]
    for adapter_cls in adapters:
        if adapter_cls.supports(model_path):
            return adapter_cls()
    raise ValueError(f"No MoE adapter registered for model at {model_path}")


def load_offloaded_model(
    model_path: str | Path,
    config: OffloadConfig | None = None,
    **from_config_kwargs: Any,
):
    offload_config = config or OffloadConfig(model_path=model_path)
    model_path = Path(model_path)
    adapter = resolve_adapter(model_path)
    _install_optional_remote_code_shims(adapter)
    index = TensorIndex.from_safetensors_index(model_path)

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    _patch_remote_config_compat(hf_config, adapter)
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(
            hf_config,
            torch_dtype=offload_config.dtype,
            trust_remote_code=True,
            **from_config_kwargs,
        )

    manager = ExpertManager(index=index, adapter=adapter, config=offload_config)
    if offload_config.offload_experts:
        adapter.replace_experts(model, manager)

    _load_non_expert_weights(model, index, adapter, offload_config)
    _materialize_non_persistent_buffers(model, hf_config, adapter, offload_config, from_config_kwargs)
    model.eval()
    model._moe_offload_manager = manager
    model._moe_offload_config = offload_config
    return model


def _load_non_expert_weights(
    model: nn.Module,
    index: TensorIndex,
    adapter: MoEAdapter,
    config: OffloadConfig,
) -> None:
    names = set(adapter.non_expert_tensor_names(index))
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(index.require(name).filename, []).append(name)

    state_dict = model.state_dict()
    missing_in_model = sorted(name for name in names if name not in state_dict)
    if missing_in_model:
        preview = ", ".join(missing_in_model[:5])
        raise KeyError(f"Checkpoint tensors missing from model state dict: {preview}")

    for filename, tensor_names in by_file.items():
        with safe_open(index.model_path / filename, framework="pt", device="cpu") as shard:
            for name in tensor_names:
                loaded = shard.get_tensor(name)
                tensor = loaded.to(
                    device=config.device,
                    dtype=config.dtype if torch.is_floating_point(loaded) else None,
                )
                _assign_tensor(model, name, tensor)


def _materialize_non_persistent_buffers(
    model: nn.Module,
    hf_config,
    adapter: MoEAdapter,
    config: OffloadConfig,
    from_config_kwargs: dict[str, Any],
) -> None:
    if isinstance(adapter, PhiMoeAdapter):
        _materialize_meta_buffers_from_reference(model, hf_config, config, from_config_kwargs)

    for module in model.modules():
        if not hasattr(module, "compute_default_rope_parameters"):
            continue
        for name, buffer in list(module.named_buffers(recurse=False)):
            if not getattr(buffer, "is_meta", False):
                continue
            if name not in {"inv_freq", "original_inv_freq"}:
                continue
            inv_freq, attention_scaling = module.compute_default_rope_parameters(
                module.config,
                device=config.device,
            )
            inv_freq = inv_freq.to(device=config.device, dtype=torch.float32)
            module.attention_scaling = attention_scaling
            module._buffers["inv_freq"] = inv_freq
            module._buffers["original_inv_freq"] = inv_freq.clone()


def _materialize_meta_buffers_from_reference(
    model: nn.Module,
    hf_config,
    config: OffloadConfig,
    from_config_kwargs: dict[str, Any],
) -> None:
    meta_buffers = {name for name, buffer in model.named_buffers() if getattr(buffer, "is_meta", False)}
    if not meta_buffers:
        return

    reference = AutoModelForCausalLM.from_config(
        hf_config,
        torch_dtype=config.dtype,
        trust_remote_code=True,
        **from_config_kwargs,
    )
    reference_buffers = dict(reference.named_buffers())
    for name in meta_buffers:
        source = reference_buffers[name].to(device=config.device)
        _assign_tensor(model, name, source)
    del reference


def _assign_tensor(model: nn.Module, name: str, tensor: torch.Tensor) -> None:
    module_name, _, leaf_name = name.rpartition(".")
    module = model.get_submodule(module_name) if module_name else model
    current = getattr(module, leaf_name)

    if isinstance(current, nn.Parameter):
        setattr(module, leaf_name, nn.Parameter(tensor, requires_grad=False))
    elif isinstance(current, torch.Tensor):
        module._buffers[leaf_name] = tensor
    else:
        raise TypeError(f"Cannot assign tensor to {name}: unsupported target {type(current)!r}")


def _install_optional_remote_code_shims(adapter: MoEAdapter) -> None:
    if not isinstance(adapter, PhiMoeAdapter):
        return
    try:
        import transformers.utils as transformers_utils
        from transformers.utils import import_utils

        import_utils.PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn", ["flash_attn"])
        if not hasattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
            transformers_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
        if not hasattr(import_utils, "is_torch_fx_available"):
            import_utils.is_torch_fx_available = lambda: True
    except Exception:
        pass
    _ensure_einops_shim()
    _ensure_flash_attn_shim()


def _patch_remote_config_compat(hf_config, adapter: MoEAdapter) -> None:
    if not isinstance(adapter, PhiMoeAdapter):
        return
    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "default":
        hf_config.rope_scaling = None
        return
    if isinstance(rope_scaling, dict) and "type" not in rope_scaling and "rope_type" in rope_scaling:
        rope_scaling["type"] = rope_scaling["rope_type"]


def _missing_optional_dependency(name: str):
    def _raise(*args, **kwargs):
        raise ImportError(f"{name} is not installed. Use eager/sdpa attention or install the optional package.")

    return _raise


def _ensure_einops_shim() -> None:
    if "einops" in sys.modules:
        return
    module = types.ModuleType("einops")
    module.__spec__ = ModuleSpec("einops", loader=None)
    module.rearrange = _missing_optional_dependency("einops")
    sys.modules["einops"] = module


def _ensure_flash_attn_shim() -> None:
    if "flash_attn" in sys.modules:
        return
    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__path__ = []
    flash_attn.__spec__ = ModuleSpec("flash_attn", loader=None, is_package=True)
    flash_attn.flash_attn_func = _missing_optional_dependency("flash_attn")
    flash_attn.flash_attn_varlen_func = _missing_optional_dependency("flash_attn")

    layers = types.ModuleType("flash_attn.layers")
    layers.__path__ = []
    layers.__spec__ = ModuleSpec("flash_attn.layers", loader=None, is_package=True)
    rotary = types.ModuleType("flash_attn.layers.rotary")
    rotary.__spec__ = ModuleSpec("flash_attn.layers.rotary", loader=None)
    rotary.RotaryEmbedding = type(
        "RotaryEmbedding",
        (),
        {"__init__": _missing_optional_dependency("flash_attn")},
    )
    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding.index_first_axis = _missing_optional_dependency("flash_attn")
    bert_padding.pad_input = _missing_optional_dependency("flash_attn")
    bert_padding.unpad_input = _missing_optional_dependency("flash_attn")

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.layers"] = layers
    sys.modules["flash_attn.layers.rotary"] = rotary
    sys.modules["flash_attn.bert_padding"] = bert_padding
