from __future__ import annotations

import argparse
import gc
import json
import re
import types
import inspect
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeForCausalLM

# 当前文件路径配置
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# -------------------- 【升级】分设备详细统计函数 --------------------
def get_model_device_stats(model: nn.Module) -> dict:
    """
    严格按设备统计：Meta / CPU / GPU(CUDA)
    返回：每个设备的 总参数数量、占用MB
    """
    stats = {
        "meta": {"params": 0, "mb": 0.0},
        "cpu": {"params": 0, "mb": 0.0},
        "cuda": {"params": 0, "mb": 0.0},
    }

    for param in model.parameters():
        numel = param.numel()
        device_type = param.device.type

        # 累加参数数量（所有设备都统计）
        if device_type == "meta":
            stats["meta"]["params"] += numel
        elif device_type == "cpu":
            stats["cpu"]["params"] += numel
        elif device_type == "cuda":
            stats["cuda"]["params"] += numel

        # 仅真实设备（CPU/GPU）计算内存，Meta 不占内存
        if device_type in ["cpu", "cuda"]:
            bytes_size = numel * param.element_size()
            mb_size = bytes_size / (1024 ** 2)
            stats[device_type]["mb"] += mb_size

    # 保留2位小数
    for k in stats:
        stats[k]["mb"] = round(stats[k]["mb"], 2)
    return stats

# -------------------- 你的原有代码（无任何改动） --------------------
from scripts.load_tensor_by_name import to_torch_dtype, validate_tensor_size
from moe_runtime import load_qwen2_moe_with_ondemand_experts

# -------------------- 测试主函数 --------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三种MoE加载方式对比测试")
    parser.add_argument("--model_dir", default="models/Qwen1.5-Moe-A2.7B", help="模型路径")
    parser.add_argument("--manifest", default="models/Qwen1.5-Moe-A2.7B/weight_manifest.json", help="manifest路径")
    parser.add_argument("--device", default="cuda", help="运行设备")
    return parser.parse_args()

# 打印分设备统计结果
def print_device_stats(stats, title):
    print(f"\n📌 {title}")
    print(f"Meta设备 参数: {stats['meta']['params']:,} | 占用: {stats['meta']['mb']} MB")
    print(f"CPU设备  参数: {stats['cpu']['params']:,} | 占用: {stats['cpu']['mb']} MB")
    print(f"GPU设备  参数: {stats['cuda']['params']:,} | 占用: {stats['cuda']['mb']} MB")
    total_params = stats['meta']['params'] + stats['cpu']['params'] + stats['cuda']['params']
    total_mb = stats['meta']['mb'] + stats['cpu']['mb'] + stats['cuda']['mb']
    print(f"✅ 总计参数: {total_params:,} | 总计占用: {round(total_mb, 2)} MB")

def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    device = args.device
    config = AutoConfig.from_pretrained(model_dir)

    # ============== 测试1：Meta空初始化 ==============
    print("="*80)
    print("🔥 测试1：Meta空初始化（无参数，仅结构）")
    with init_empty_weights():
        model1 = Qwen2MoeForCausalLM(config)
    stats1 = get_model_device_stats(model1)
    print_device_stats(stats1, "Meta初始化 设备分布")
    del model1
    gc.collect()

    # ============== 测试2：按需加载 ==============
    print("\n" + "="*80)
    print("🔥 测试2：自定义按需加载（常驻权重+磁盘专家）")
    model2 = load_qwen2_moe_with_ondemand_experts(
        model_dir=args.model_dir,
        manifest_path=args.manifest,
        device=device,
        empty_cache_after_expert=True,
    )
    stats2 = get_model_device_stats(model2)
    print_device_stats(stats2, "按需加载 设备分布")
    del model2
    gc.collect()

    # ============== 测试3：普通随机加载 ==============
    print("\n" + "="*80)
    print("🔥 测试3：普通随机初始化加载（默认PyTorch方式）")
    try:
        model3 = Qwen2MoeForCausalLM(config)
        stats3 = get_model_device_stats(model3)
        print_device_stats(stats3, "普通加载 设备分布")
        del model3
        gc.collect()
    except:
        print("❌ 普通加载内存不足，程序崩溃（正常现象）")

if __name__ == "__main__":
    main()