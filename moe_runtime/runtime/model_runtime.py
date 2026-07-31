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
    - 它持有 cache_manager，并把它注入到 adapter 替换后的模块里。
    """

    def __init__(
        self,
        *,
        runtime_config: RuntimeConfig,
        adapter: Any,
        tensor_loader: ManifestTensorLoader,
        model_config: Any | None = None,
        cache_manager: Any | None = None,
    ) -> None:
        """
        初始化模型运行时依赖。

        输入：运行配置、模型 Adapter、张量加载器、模型配置和 CacheManager。
        输出：无。
        功能：保存模型装配与常驻权重加载依赖，具体 gate 预取模块由 Adapter 按层替换。
        """

        # 顶层运行时配置，包含模型目录、manifest 路径、设备策略等。
        self.runtime_config = runtime_config

        # transformers 模型 config；
        self.model_config = model_config

        # 模型适配器，负责屏蔽 Qwen/Phi/Mixtral 等模型结构差异。
        self.adapter = adapter

        # 已实现的 tensor_name 精确加载器。
        self.tensor_loader = tensor_loader

        # 模型级缓存管理器，后续负责 routed expert 的加载、命中、淘汰和显存布局。
        self.cache_manager = cache_manager

        # 当前已经装配好的 transformers 模型实例。
        self.model: nn.Module | None = None

        # 标记模型常驻权重是否已经加载完成。
        self.is_loaded = False

    def load_model(self):
        """构造模型结构，替换 routed experts，并加载常驻权重。"""

        # 如果已经加载过，直接返回同一个模型实例，避免重复占用显存。
        if self.is_loaded and self.model is not None:
            return

        # 由 adapter 构造对应模型的空结构。
        self.model = self.adapter.build_empty_model(self.model_config)

        # 先替换 routed experts，避免后续为原始 experts 分配或加载常驻参数。
        self.model = self.adapter.patch_moe_block(
            model=self.model,
            model_config=self.model_config,
            runtime_config=self.runtime_config,
            cache_manager=self.cache_manager,
        )

        # 从 manifest 中逐个加载非 routed expert 权重。
        self._load_resident_weights()

        # 推理场景默认关闭训练行为。
        self.model.eval()
        self.model.requires_grad_(False)

        # 注册 forward hook：每次 forward 推进 cache step，驱动 LRU 老化
        if self.cache_manager is not None:
            def _step_hook(_module, _args, output):
                self.cache_manager.step()
                return output
            self.model.register_forward_hook(_step_hook)

        self.is_loaded = True


    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """调用底层模型 generate。"""

        if not hasattr(self.model, "generate"):
            raise AttributeError("当前模型没有 generate 方法。")
        return self.model.generate(*args, **kwargs)

    def _load_resident_weights(self):
        """将非 routed expert 的常驻权重加载到模型对应设备。

        这里的原则是：
        - routed expert 不加载，留给后续 cache_manager 按需处理
        - shared expert 按 shared_expert_device 加载
        - 其余常驻权重按 non_expert_device 加载
        - 直接把 tensor 写回模型参数/缓冲区，避免再次整模型搬运
        """

        if self.model is None:
            raise RuntimeError("ModelRuntime.model 为空，无法加载常驻权重。")
        
        # ---- 1. 加载常驻 param ----
        # 只从当前模型结构中收集参数名，不在这里读取 manifest/json。
        resident_param_names: list[str] = []

        # named_parameters 会返回当前模型真实存在的参数；routed expert 参数已经由 adapter 替换掉。
        for name, param in self.model.named_parameters():

            # routed expert 权重不在初始化阶段加载，后续交给 cache_manager 按需加载。
            if self.adapter.is_routed_expert_tensor(name):
                continue

            # 其余参数都属于 resident weights，需要加载到 GPU。
            resident_param_names.append(name)

        # 权重读取完全交给 tensor_loader，它内部已经持有 manifest 信息。
        resident_tensors = self.tensor_loader.load_tensors_by_names(
            tensor_names=resident_param_names,
            device=self.runtime_config.non_expert_device,
        )

        # 将真实 tensor 写回模型对应参数位置。
        for name in resident_param_names:
            tensor = resident_tensors[name]

            # 根据 named_parameters 的名字定位父模块和参数名。
            if "." in name:
                parent_name, param_name = name.rsplit(".", 1)
                parent_module = self.model.get_submodule(parent_name)
            else:
                parent_module = self.model
                param_name = name

            # 用 PyTorch 标准属性赋值替换参数，同时保持模块属性和参数注册表一致。
            setattr(parent_module, param_name, nn.Parameter(tensor, requires_grad=False))

        # ---- 2. 加载常驻 Buffer（仅限 manifest 中存在的） ----
        resident_buffer_names: list[str] = []

        for name, _buf in self.model.named_buffers():
            if self.adapter.is_routed_expert_tensor(name):
                continue
            if not self.tensor_loader.has_tensor(tensor_name=name):
                continue
            resident_buffer_names.append(name)

        if resident_buffer_names:
            buffer_tensors = self.tensor_loader.load_tensors_by_names(
                tensor_names=resident_buffer_names,
                device=self.runtime_config.non_expert_device,
            )
            for name in resident_buffer_names:
                tensor = buffer_tensors[name]
                if "." in name:
                    parent_name, buf_name = name.rsplit(".", 1)
                    parent_module = self.model.get_submodule(parent_name)
                else:
                    parent_module = self.model
                    buf_name = name
                # buffer 不包 nn.Parameter，直接赋值 Tensor
                setattr(parent_module, buf_name, tensor)

        # ---- 3. Materialize 残留的 meta tensor（由 adapter 处理模型差异） ----
        self.adapter.materialize_meta_tensors(self.model, torch.device(self.runtime_config.non_expert_device))
