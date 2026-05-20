from __future__ import annotations

# dataclass 用来定义轻量配置对象，避免 RuntimeEngine 构造函数参数越来越长。
from dataclasses import dataclass
# Path 用来规范化模型目录和 manifest 路径。
from pathlib import Path


@dataclass
class RuntimeConfig:
    """MoE Runtime 的顶层配置。

    """
    # 本地模型目录，例如 models/Qwen1.5-Moe-A2.7B。
    model_dir: str | Path

    # 权重 manifest 路径；如果不传，默认使用 <model_dir>/weight_manifest.json。
    manifest_path: str | Path | None = None

    # 模型适配器名称；None 表示让 RuntimeEngine 后续根据 config 自动选择。
    adapter_name: str | None = None

    # 非专家权重所在设备。
    non_expert_device: str = "cuda"

    # 共享专家权重所在设备。
    shared_expert_device: str = "cuda"
    
    # 专家权重按需加载后的目标设备。
    routed_expert_device: str = "cuda"

    # 缓存策略。
    cache_policy: str = "none"

    # 预取策略。
    prefetch_policy: str = "none"

    # CPU cache 的容量上限。
    max_cpu_cache_bytes: int | None = None

    # GPU cache 的容量上限。
    max_gpu_cache_bytes: int | None = None

    def __post_init__(self) -> None:
        """规范化路径字段，保证 RuntimeEngine 内部拿到的是 Path。"""

        # model_dir 是必填项，统一转换为绝对路径。
        self.model_dir = Path(self.model_dir).resolve()

        # manifest_path 不传时默认放在模型目录下。
        if self.manifest_path is None:
            self.manifest_path = self.model_dir / "weight_manifest.json"
        else:
            self.manifest_path = Path(self.manifest_path).resolve()

    @classmethod
    def from_pretrained_args(
        cls,
        model_dir: str | Path,
        **kwargs,
    ) -> "RuntimeConfig":
        """从 from_pretrained 风格参数构造 RuntimeConfig。"""

        # 这个方法让 RuntimeEngine.from_pretrained 可以保持简洁。
        return cls(model_dir=model_dir, **kwargs)
