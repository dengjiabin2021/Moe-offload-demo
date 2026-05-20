from __future__ import annotations

# argparse 用来提供一个最小命令行入口，方便后续直接运行脚本验证流程。
import argparse
# Path 用来处理项目路径和模型目录。
from pathlib import Path
# sys 用来把项目根目录加入 Python import path。
import sys


# 当前脚本位于 <repo>/scripts/run_model.py，因此 parents[1] 是项目根目录。
REPO_ROOT = Path(__file__).resolve().parents[1]

# 直接用 python scripts/run_model.py 启动时，确保可以导入本仓库的 moe_runtime 包。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# RuntimeConfig 是 RuntimeEngine 的配置入口。
from moe_runtime import MoeRuntimeEngine, RuntimeConfig


def parse_args() -> argparse.Namespace:
    """解析运行脚本参数。"""

    # 这个脚本目前只是 RuntimeEngine 调用流程示例，不要求现在完整跑通推理。
    parser = argparse.ArgumentParser(description="MoE RuntimeEngine 基本调用流程示例。")

    # 本地模型目录，默认指向当前项目里的 Qwen1.5-MoE-A2.7B。
    parser.add_argument(
        "--model-dir",
        default="models/Qwen1.5-Moe-A2.7B",
        help="本地模型目录。",
    )

    # manifest 路径可选；不传时 RuntimeConfig 会默认使用 <model_dir>/weight_manifest.json。
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="weight_manifest.json 路径。",
    )

    # 非专家权重设备，当前规划是常驻 GPU。
    parser.add_argument(
        "--non-expert-device",
        default="cuda",
        help="非专家权重常驻设备。",
    )

    # shared expert 权重设备，当前规划是常驻 GPU。
    parser.add_argument(
        "--shared-expert-device",
        default="cuda",
        help="共享专家权重常驻设备。",
    )

    # routed expert 按需加载后的目标设备。
    parser.add_argument(
        "--routed-expert-device",
        default="cuda",
        help="专家权重按需加载后的目标设备。",
    )

    # 缓存策略现在先保留 none，后续可扩展 cpu/gpu/hybrid。
    parser.add_argument(
        "--cache-policy",
        default="none",
        help="专家缓存策略。",
    )

    # 预取策略现在先保留 none，后续可扩展具体策略。
    parser.add_argument(
        "--prefetch-policy",
        default="none",
        help="专家预取策略。",
    )

    # 输入 prompt 目前只作为未来推理流程占位。
    parser.add_argument(
        "--prompt",
        default="用一句话解释 MoE 模型。",
        help="未来推理接口使用的输入文本。",
    )

    return parser.parse_args()


def main() -> None:
    """RuntimeEngine 基本调用流程。"""

    # 读取命令行参数。
    args = parse_args()

    # 1. 构造 RuntimeConfig。
    # 这里只把运行时需要的策略集中进 config，不直接加载模型。
    config = RuntimeConfig(
        model_dir=args.model_dir,
        manifest_path=args.manifest_path,
        non_expert_device=args.non_expert_device,
        shared_expert_device=args.shared_expert_device,
        routed_expert_device=args.routed_expert_device,
        cache_policy=args.cache_policy,
        prefetch_policy=args.prefetch_policy,
    )

    # 2. 创建 RuntimeEngine。
    # 上层用户未来主要只需要接触这个对象。
    engine = MoeRuntimeEngine(config)



if __name__ == "__main__":
    main()
