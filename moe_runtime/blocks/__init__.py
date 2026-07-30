"""MoE runtime blocks。"""

from .deepseekv2_moe_block import DeepseekV2OnDemandExperts
from .qwen2_moe_block import Qwen2OnDemandExperts
from .async_deepseekv2_moe_block import asyncDeepseekV2OnDemandExperts
from .async_qwen2_moe_block import asyncQwen2OnDemandExperts
from .deepseekv2_prefetch_patch import patch_deepseekv2_prefetch
from .qwen2_prefetch_patch import patch_qwen2_prefetch

__all__ = [
    "DeepseekV2OnDemandExperts",
    "Qwen2OnDemandExperts",
    "asyncDeepseekV2OnDemandExperts",
    "asyncQwen2OnDemandExperts",
    "patch_deepseekv2_prefetch",
    "patch_qwen2_prefetch",
]
