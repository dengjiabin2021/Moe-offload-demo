from __future__ import annotations

# Path 用于模型目录类型标注。
from pathlib import Path
# Any 用于 transformers config 类型标注。
from typing import Any

# Adapter 基类。
from .base import MoeModelAdapter
# 当前第一版只注册 Qwen1.5-MoE Adapter。
from .qwen15_moe import Qwen15MoeAdapter


# Adapter 注册表。后续新增 PhiMoE/Mixtral/DeepSeek 时，把对应 Adapter 类加入这里。
ADAPTER_REGISTRY: list[type[MoeModelAdapter]] = [
    Qwen15MoeAdapter,
]


def auto_select_adapter(model_dir: Path, model_config: Any) -> MoeModelAdapter:
    """根据模型 config 自动选择 Adapter。"""

    # 每个 Adapter 自己判断是否支持当前模型。
    for adapter_cls in ADAPTER_REGISTRY:
        adapter = adapter_cls()
        if adapter.match(model_config):
            return adapter

    # 没有适配器匹配时，提示当前 model_type。
    model_type = getattr(model_config, "model_type", None)
    raise ValueError(f"No MoE adapter found for model_type={model_type!r}, model_dir={model_dir}")

