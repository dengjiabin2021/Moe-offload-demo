"""MoE gate 预测预取器。"""

from .deepseekv2_gate_prefetcher import DeepseekV2GatePrefetcher
from .qwen2_gate_prefetcher import Qwen2GatePrefetcher

__all__ = [
    "DeepseekV2GatePrefetcher",
    "Qwen2GatePrefetcher",
]
