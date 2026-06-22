from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig

from moe_runtime.adapters import auto_select_adapter
from moe_runtime.cache_manager import ExpertCacheManager, asyncExpertCacheManager
from moe_runtime.config import RuntimeConfig
from moe_runtime.loader import ManifestTensorLoader
from moe_runtime.runtime.model_runtime import ModelRuntime


class MoeRuntimeEngine:
    """MoE Runtime 的顶层抽象。

    设计目标：
    - 对上层用户只暴露 RuntimeEngine。
    - 内部组合 config、模型适配器、tensor loader、cache、prefetch、expert runtime 等组件。
    """

    def __init__(self, config: RuntimeConfig) -> None:
        """创建 RuntimeEngine 实例。"""

        # 顶层运行时配置，包含模型目录、manifest 路径、设备、缓存策略、预取策略等。
        self.config: RuntimeConfig = config

        # transformers 模型配置对象，例如 Qwen2MoeConfig；由模型适配器加载。
        self.model_config: Any | None = None

        # 模型适配器，例如 Qwen1.5-MoE adapter。
        # 它只负责识别模型并创建具体 ModelRuntime。
        self.adapter: Any | None = None

        # 具体模型运行时，例如 ModelRuntime。
        # 模型加载、结构替换、推理适配都放在这里。
        self.model_runtime: Any | None = None

        # 张量加载器。
        # 它只负责通过 tensor_name 从磁盘加载 tensor。
        self.tensor_loader: ManifestTensorLoader | None = None

        # 缓存管理器。
        self.cache_manager: Any | None = None

        # 预取器。
        self.prefetcher: Any | None = None

    def build_components(self) -> None:
        """根据 RuntimeConfig 创建内部组件。"""

        # 1. 创建 tensor loader，负责通过 tensor_name 从 safetensors 分片加载权重。
        self.tensor_loader = ManifestTensorLoader(
            model_dir=self.config.model_dir,
            manifest_path=self.config.manifest_path,
        )

        # 2. 由 transformers 加载模型 config，adapter 的 match / build_empty_model 依赖它。
        self.model_config = AutoConfig.from_pretrained(self.config.model_dir)

        # 3. 根据 model_type 自动选择对应 Adapter。
        self.adapter = auto_select_adapter(self.config.model_dir, self.model_config)

        # 4. 创建 ExpertCacheManager。串行/异步由 RuntimeConfig.cache_manager_mode 统一决定。
        cache_manager_cls = self._select_cache_manager_cls()
        self.cache_manager = cache_manager_cls(
            runtime_config=self.config,
            adapter=self.adapter,
            tensor_loader=self.tensor_loader,
            model_config=self.model_config,
        )

        self.model_runtime = ModelRuntime(
            runtime_config=self.config,
            adapter=self.adapter,
            tensor_loader=self.tensor_loader,
            model_config=self.model_config,
            cache_manager=self.cache_manager,
            prefetcher=self.prefetcher,
        )


    def load_model(self) -> None:
        """构造 ModelRuntime 并加载常驻权重。"""

        self.model_runtime.load_model()

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """执行模型生成（forward hook 自动推进 cache step）。"""

        if self.model_runtime is None:
            raise RuntimeError("请先调用 load_model() 加载模型。")

        return self.model_runtime.generate(*args, **kwargs)

    def _select_cache_manager_cls(self) -> type:
        """根据 RuntimeConfig.cache_manager_mode 选择串行或异步 CacheManager。"""

        cache_manager_mode = self.config.cache_manager_mode

        if cache_manager_mode == "serial":
            return ExpertCacheManager

        if cache_manager_mode == 'async':
            return asyncExpertCacheManager

        raise ValueError(
            f"Unsupported cache_manager_mode: {cache_manager_mode}. "
            "Expected 'serial', 'async'."
        )
