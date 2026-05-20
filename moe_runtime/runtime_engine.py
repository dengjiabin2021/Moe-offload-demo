from __future__ import annotations

# Path 用于 from_pretrained 的模型路径参数类型标注。
from pathlib import Path
# Any 用于尚未实现的模型、适配器、缓存、预取器等内部组件类型占位。
from typing import Any

# RuntimeConfig 是 RuntimeEngine 的统一配置入口。
from moe_runtime.config import RuntimeConfig
# Adapter 注册表用于按配置自动选择具体模型 Adapter。
from moe_runtime.adapters import auto_select_adapter
# 复用你已经实现好的张量加载器：通过 tensor_name 从 manifest 中加载 tensor。
from moe_runtime.loader import ManifestTensorLoader



class MoeRuntimeEngine:
    """MoE Runtime 的顶层抽象。

    设计目标：
    - 对上层用户只暴露 RuntimeEngine。
    - 内部组合 config、模型适配器、tensor loader、cache、prefetch、expert runtime 等组件。
    """

    def __init__(self, config: RuntimeConfig) -> None:
        """创建 RuntimeEngine 实例。
        """

        # 顶层运行时配置，包含模型目录、manifest 路径、设备、缓存策略、预取策略等。
        self.config: RuntimeConfig = config

        # transformers 模型配置对象，例如 Qwen2MoeConfig；由模型适配器加载。
        self.model_config: Any | None = None

        # 模型适配器，例如 Qwen1.5-MoE adapter。
        # 它只负责识别模型并创建具体 ModelRuntime。
        self.adapter: Any | None = None

        # 具体模型运行时，例如 Qwen15MoeRuntime。
        # 模型加载、结构替换、推理适配都放在这里。
        self.model_runtime: Any | None = None

        # 张量加载器。
        # 它只负责通过 tensor_name 从磁盘加载 tensor。
        self.tensor_loader: ManifestTensorLoader | None = None

        # 缓存管理器。
        # 当前策略可以是 none；未来扩展 CPU cache、GPU cache、混合 cache。
        self.cache_manager: Any | None = None

        # 预取器。
        # 当前策略可以是 none；未来根据 router 输出或 token/layer 调度预取。
        self.prefetcher: Any | None = None


    def build_components(self) -> None:
        """根据 RuntimeConfig 创建内部组件。

        """

        # 创建 tensor loader。它是通用服务，不属于某个具体模型。
        self.tensor_loader = ManifestTensorLoader(
            model_dir=self.config.model_dir,
            manifest_path=self.config.manifest_path,
        )


    def load_model(self):



        return
    
    def generate(self):


        return
