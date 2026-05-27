from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers.models.qwen2_moe.configuration_qwen2_moe import Qwen2MoeConfig


# 将项目根目录加入 sys.path，保证直接运行本脚本时可以导入 moe_runtime。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moe_runtime.adapters.qwen15_moe_adapter import Qwen15MoeAdapter
from moe_runtime.config import RuntimeConfig
from moe_runtime.loader import ManifestTensorLoader
from moe_runtime.runtime.model_runtime import ModelRuntime


def parse_args() -> argparse.Namespace:
    """解析测试脚本参数。"""

    parser = argparse.ArgumentParser(description="测试 ModelRuntime 加载 resident weights。")
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "models" / "Qwen1.5-Moe-A2.7B"),
        help="本地模型目录。",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="resident weights 加载目标设备，默认 cuda。",
    )
    return parser.parse_args()


def main() -> None:
    """测试入口：加载常驻权重，并打印关键检查结果。"""

    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    manifest_path = model_dir / "weight_manifest.json"

    # 本测试目标是加载到 GPU，如果环境没有 CUDA，直接给出明确错误。
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA，无法测试 resident weights 加载到 GPU。")

    # 将 cuda 规范化为 cuda:当前设备，避免 cuda 和 cuda:0 字符串比较误报。
    expected_device = torch.device(args.device)
    if expected_device.type == "cuda" and expected_device.index is None:
        expected_device = torch.device("cuda", torch.cuda.current_device())

    # RuntimeConfig 只负责提供路径和设备配置。
    runtime_config = RuntimeConfig(
        model_dir=model_dir,
        manifest_path=manifest_path,
        non_expert_device=args.device,
        shared_expert_device=args.device,
        routed_expert_device=args.device,
    )

    # Qwen1.5-MoE 使用 transformers 的 Qwen2MoeConfig。
    model_config = Qwen2MoeConfig.from_pretrained(model_dir)

    # adapter 负责 Qwen 模型结构适配。
    adapter = Qwen15MoeAdapter()

    # tensor_loader 负责从 manifest 和 safetensors 分片中按 tensor_name 加载权重。
    tensor_loader = ManifestTensorLoader(
        model_dir=runtime_config.model_dir,
        manifest_path=runtime_config.manifest_path,
    )

    # 按要求，cache_manager 和 prefetcher 都先设置成 None。
    model_runtime = ModelRuntime(
        runtime_config=runtime_config,
        adapter=adapter,
        tensor_loader=tensor_loader,
        model_config=model_config,
        cache_manager=None,
        prefetcher=None,
    )

    print("开始加载 resident weights...")
    start_time = time.perf_counter()
    model_runtime.load_model()
    load_seconds = time.perf_counter() - start_time

    if model_runtime.model is None:
        raise RuntimeError("模型加载后仍然为空。")

    resident_count = 0
    routed_count = 0
    meta_count = 0
    device_count: dict[str, int] = {}

    # 检查当前模型中所有参数的设备状态。
    for name, param in model_runtime.model.named_parameters():
        if adapter.is_routed_expert_tensor(name):
            routed_count += 1
            continue

        resident_count += 1
        device_count[str(param.device)] = device_count.get(str(param.device), 0) + 1

        if param.is_meta:
            meta_count += 1

        if param.device != expected_device:
            raise RuntimeError(f"resident 参数设备错误: {name}, device={param.device}, expected={expected_device}")

        if param.requires_grad:
            raise RuntimeError(f"resident 参数 requires_grad 应为 False: {name}")

    print("resident weights 加载完成。")
    print(f"resident weights 加载耗时: {load_seconds:.2f} 秒")
    print(f"resident 参数数量: {resident_count}")
    print(f"模型中剩余 routed expert 参数数量: {routed_count}")
    print(f"resident meta 参数数量: {meta_count}")
    print(f"参数设备分布: {device_count}")

    # 按已加载 resident 参数估算吞吐，便于后续比较不同 loader/cache 实现。
    loaded_bytes = sum(
        param.numel() * param.element_size()
        for name, param in model_runtime.model.named_parameters()
        if not adapter.is_routed_expert_tensor(name)
    )
    print(f"resident 参数体积: {loaded_bytes / 1024**3:.2f} GB")
    print(f"resident 加载吞吐: {loaded_bytes / load_seconds / 1024**3:.2f} GB/s")

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print(f"CUDA 已分配显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"CUDA 已保留显存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    if meta_count != 0:
        raise RuntimeError("仍有 resident 参数停留在 meta device。")

    print("测试通过：resident weights 已加载，prefetcher/cache_manager 均为 None。")


if __name__ == "__main__":
    main()
