from __future__ import annotations

# argparse 用来解析命令行参数，例如 tensor 名、模型目录、目标设备等。
import argparse
# json 用来读取 build_safetensors_manifest.py 生成的 weight_manifest.json。
import json
# math.prod 用来计算 shape 中所有维度的乘积，也就是 tensor 元素个数。
from math import prod
# Path 用来安全地拼接模型目录、manifest 路径和分片文件路径。
from pathlib import Path
# Any 用在类型标注中，表示 JSON 里可能有字符串、数字、列表、字典等类型。
from typing import Any

import sys
# 当前文件路径配置
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moe_runtime.loader import ManifestTensorLoader


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    # 这个脚本会真正把某个权重读成 torch.Tensor，因此运行时需要安装 torch。
    parser = argparse.ArgumentParser(
        description="根据 tensor_name 从本地 safetensors 分片中精确读取一个权重 tensor。"
    )

    # tensor 名必须显式传入，例如 model.layers.0.mlp.experts.0.gate_proj.weight。
    parser.add_argument(
        "--tensor_name",
        default=None,
        help="要读取的完整 tensor 名称。",
    )

    # 模型目录中需要包含 weight_manifest.json 和对应的 safetensors 分片。
    parser.add_argument(
        "--model_dir",
        default=None,
        help="模型目录",
    )

    # manifest 路径可选；不传时默认使用模型目录下的 weight_manifest.json。
    parser.add_argument(
        "--manifest_path",
        default=None,
        help="manifest JSON 路径。",
    )

    # 默认先读到 CPU；后续如果要直接放 GPU，可以传 --device cuda。
    parser.add_argument(
        "--device",
        default="cuda",
        help="加载后的目标设备，例如 cpu、cuda、cuda:0。",
    )

    # 默认只打印 tensor 元信息，不打印值，避免误把大权重刷满终端。
    parser.add_argument(
        "--print-values",
        default = True,
        help="打印 tensor 内容。大 tensor 不建议开启。",
    )

    return parser.parse_args()


def main() -> None:
    """命令行入口：读取一个 tensor，并打印基本信息。"""

    # 解析用户命令行输入。
    args = parse_args()

    model_dir = Path(args.model_dir).resolve() if args.model_dir else Path("./models/Qwen1.5-Moe-A2.7B").resolve()

    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else model_dir / "weight_manifest.json"

    device = args.device

    # 通过 tensor_name 从 manifest 定位、读取并还原成 torch.Tensor。

    tensorloader = ManifestTensorLoader(model_dir = model_dir, manifest_path = manifest_path)

    #tensor_name = args.tensor_name if args.tensor_name else "lm_head.weight"
    #tensor = tensorloader.load_tensor_by_name(tensor_name=tensor_name, device=device)

    tensor_names = ["lm_head.weight","model.embed_tokens.weight"]
    tensors = tensorloader.load_tensors_by_names(tensor_names=tensor_names, device=device)
    # ===================== 新增/优化打印 =====================
    for tensor_name in tensor_names:
        print(f"✅ 成功加载张量: {tensor_name}")
        print(f"📦 Tensor 形状: {tensors[tensor_name].shape}")       # 你要求的形状打印
        print(f"🔣 Tensor 类型: {tensors[tensor_name].dtype}")
        print(f"📱 Tensor 设备: {tensors[tensor_name].device}")
        # ========================================================

        # 如果用户明确要求打印值，才输出 tensor 内容。
        if args.print_values:
            print(tensors[tensor_name])


# 只有直接执行该脚本时才运行 main()；被其他代码 import 时只提供 load_tensor_by_name 函数。
if __name__ == "__main__":
    main()
