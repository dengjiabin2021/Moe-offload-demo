"""MoE runtime blocks。"""

from .deepseekv2_moe_block import DeepseekV2OnDemandExperts
from .qwen2_moe_block import Qwen2OnDemandExperts
from .sync_deepseekv2_moe_block import asyncDeepseekV2OnDemandExperts
from .sync_qwen2_moe_block import asyncQwen2OnDemandExperts

__all__ = [
    "DeepseekV2OnDemandExperts",
    "Qwen2OnDemandExperts",
    "asyncDeepseekV2OnDemandExperts",
    "asyncQwen2OnDemandExperts",
]
