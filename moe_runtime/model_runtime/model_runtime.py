from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from transformers import AutoConfig

from moe_runtime.config import RuntimeConfig
from moe_runtime.loader import ManifestTensorLoader


class ModelRuntime:
    """通用 MoE 模型运行时。

    ModelRuntime 是 RuntimeEngine 内部真正执行模型装配的核心组件：
    - 它不关心具体模型结构差异，模型差异由 adapter 屏蔽。
    - 它负责构造空模型、替换 MoE 专家模块、加载常驻权重。
    - 它持有 cache_manager 和 prefetcher，并把它们注入到 adapter 替换后的模块里。
    """

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        adapter: Any,
        tensor_loader: ManifestTensorLoader,
        model_config: Any | None = None,
        cache_manager: Any | None = None,
        prefetcher: Any | None = None,
    ) -> None:
        # 顶层运行时配置，包含模型目录、manifest 路径、设备策略等。
        self.runtime_config = runtime_config

        # 模型适配器，负责屏蔽 Qwen/Phi/Mixtral 等模型结构差异。
        self.adapter = adapter

        # 已实现的 tensor_name 精确加载器。
        self.tensor_loader = tensor_loader

        # transformers 模型 config；如果上层不传，load_model 时会从本地模型目录读取。
        self.model_config = model_config

        # 模型级缓存管理器，后续负责 routed expert 的加载、命中、淘汰和显存布局。
        self.cache_manager = cache_manager

        # 模型级预取器，后续负责根据路由或策略触发专家预取。
        self.prefetcher = prefetcher

        # 当前已经装配好的 transformers 模型实例。
        self.model: nn.Module | None = None

        # 标记模型常驻权重是否已经加载完成。
        self.is_loaded = False

    def load_model(self) -> nn.Module:
        """构造模型结构，替换 routed experts，并加载常驻权重。"""

        # 如果已经加载过，直接返回同一个模型实例，避免重复占用显存。
        if self.is_loaded and self.model is not None:
            return self.model

        # 确保存在模型 config。
        self.model_config = self._load_model_config()

        # 由 adapter 构造对应模型的空结构。
        model = self.adapter.build_empty_model(self.model_config)

        # 先替换 routed experts，避免后续为原始 experts 分配或加载常驻参数。
        model = self._patch_model(model)

        # 从 manifest 中逐个加载非 routed expert 权重。
        self._load_resident_weights(model)

        # 推理场景默认关闭训练行为。
        model.eval()

        # 保存模型实例并更新状态。
        self.model = model
        self.is_loaded = True
        return model


    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """调用底层模型 generate。"""

        # generate 前必须先完成 load_model。
        model = self._require_model()
        if not hasattr(model, "generate"):
            raise AttributeError("当前模型没有 generate 方法。")
        return model.generate(*args, **kwargs)

