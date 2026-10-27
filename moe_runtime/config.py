from __future__ import annotations

# dataclass 用来定义轻量配置对象，避免 RuntimeEngine 构造函数参数越来越长。
from dataclasses import dataclass
# Path 用来规范化模型目录和 manifest 路径。
from pathlib import Path


# demand 使用独立磁盘 worker，避免已经排队的预取阻塞当前层真实专家。
_DEMAND_DISK_WORKERS = 4

# 预取只占用少量独立 worker，防止低优先级读取耗尽磁盘并发。
_PREFETCH_DISK_WORKERS = 2

# 后台调度线程没有可推进任务时的短等待时间，单位秒。
_SCHEDULER_WAIT_SECONDS = 0.0005

# block 阻塞等待任意 expert ready 时的轮询等待时间，单位秒。
_WAIT_READY_SECONDS = 0.001


@dataclass
class RuntimeConfig:
    """MoE Runtime 的顶层配置。

    该配置集中描述模型路径、设备、缓存容量、缓存策略和预取策略，
    由 RuntimeEngine、CacheManager 和 Adapter 在组件装配阶段共同读取。
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

    # 缓存淘汰策略：lru 表示覆盖最久未使用的专家；lfu 表示覆盖访问频率最低的专家。
    cache_policy: str = "lru"

    # CacheManager 实现模式：serial 使用串行加载；async 使用异步加载。
    cache_manager_mode: str = "async"

    # 预取策略。
    prefetch_policy: str = "none"

    # Gate 预取向后预测的 MoE 层数量；普通 MLP 层不计入距离。
    prefetch_distance: int = 1

    # 是否为 expert CPU cache 使用 pinned memory；仅在 routed expert 目标设备为可用 CUDA 时生效。
    pin_memory: bool = False

    # CPU 缓存容量上限
    max_cpu_cache_capacity: dict[int, int] = None

    # GPU 缓存容量上限
    max_gpu_cache_capacity: dict[int, int] = None

    def __post_init__(self) -> None:
        """
        校验运行配置并规范化模型路径。

        输入：当前 RuntimeConfig 实例中的配置字段。
        输出：无。
        功能：校验缓存和预取模式，并把模型目录及 manifest 路径统一转换为绝对 Path。
        """

        if self.cache_policy not in {"lru", "lfu"}:
            raise ValueError(f"Unsupported cache_policy: {self.cache_policy}. Expected 'lru' or 'lfu'.")

        if self.prefetch_policy not in {"none", "gate"}:
            raise ValueError(
                f"Unsupported prefetch_policy: {self.prefetch_policy}. "
                "Expected 'none' or 'gate'."
            )

        if self.prefetch_distance < 1:
            raise ValueError("prefetch_distance must be greater than or equal to 1.")

        if self.prefetch_policy == "gate" and self.cache_manager_mode != "async":
            raise ValueError("Gate prefetch requires cache_manager_mode='async'.")

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
