from __future__ import annotations

# argparse 用来解析 demo 的命令行参数。
import argparse
# sys 用来把项目根目录加入 import path，保证从 scripts 目录脚本启动时也能导入 moe_runtime。
import sys
# Path 用来计算当前脚本所在位置和项目根目录。
from pathlib import Path

# 当前文件位于 <repo>/scripts/qwen_moe_ondemand_demo.py，所以 parents[1] 是项目根目录。
REPO_ROOT = Path(__file__).resolve().parents[1]

# 把项目根目录放到 sys.path 最前面，确保可以导入本仓库里的 moe_runtime 包。
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    """解析 demo 命令行参数。"""

    parser = argparse.ArgumentParser(description="Qwen1.5-MoE routed experts 按需加载 demo。")

    # 模型目录需要包含 config/tokenizer/safetensors/weight_manifest.json。
    parser.add_argument(
        "--model-dir",
        default="models/Qwen1.5-Moe-A2.7B",
        help="本地 Qwen1.5-MoE-A2.7B 模型目录。",
    )

    # manifest 默认在模型目录下，也可以手动指定其他路径。
    parser.add_argument(
        "--manifest",
        default="models/Qwen1.5-Moe-A2.7B/weight_manifest.json",
        help="weight_manifest.json 路径，默认读取 <model-dir>/weight_manifest.json。",
    )

    # prompt 用来做最小生成验证。
    parser.add_argument(
        "--prompt",
        default="用一句话解释 MoE 模型。",
        help="输入提示词。",
    )

    # 限制输出长度，避免 demo 第一次运行时间太久。
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="最多生成的新 token 数。",
    )

    # 目标设备。这个 demo 的设计目标是 cuda；cpu 也能用于结构调试，但会非常慢。
    parser.add_argument(
        "--device",
        default="cuda",
        help="模型常驻权重和按需专家权重加载到的设备，例如 cuda、cuda:0、cpu。",
    )

    # 默认每个专家用完后 empty_cache；如果只想验证功能、不想频繁清 allocator cache，可以打开这个选项。
    parser.add_argument(
        "--keep-pytorch-cache",
        action="store_true",
        help="专家计算后不调用 torch.cuda.empty_cache()。",
    )

    return parser.parse_args()


def main() -> None:
    """加载模型并执行一次文本生成。"""

    # 读取命令行参数。
    args = parse_args()

    # torch/transformers 只在真正运行 demo 时才需要导入，这样 --help 不会因为没装依赖而失败。
    try:
        import torch
        from transformers import AutoTokenizer
        from moe_runtime import load_qwen2_moe_with_ondemand_experts
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("运行 demo 需要先安装依赖：pip install -r requirements.txt") from exc

    # tokenizer 没有专家权重问题，直接用 transformers 官方加载。
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=False)

    # 加载模型：非 routed expert 权重常驻 GPU，routed expert 权重 forward 时按需读取。
    model = load_qwen2_moe_with_ondemand_experts(
        model_dir=args.model_dir,
        manifest_path=args.manifest,
        device=args.device,
        empty_cache_after_expert=not args.keep_pytorch_cache,
    )


    # 关闭梯度，减少推理时的额外内存。
    with torch.inference_mode():
        # 把输入 prompt 转成 tensor，并移动到同一设备。
        inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)

        # 先用贪心生成做最小验证，避免采样引入额外随机性。
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    # 解码并打印完整文本。
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


# 只有直接运行该文件时才启动 demo。
if __name__ == "__main__":
    main()
