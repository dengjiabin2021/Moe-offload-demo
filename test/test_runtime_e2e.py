from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# 将项目根目录加入 sys.path，保证直接运行本脚本时可以导入 moe_runtime。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moe_runtime.config import RuntimeConfig
from moe_runtime.runtime_engine import MoeRuntimeEngine


def parse_args() -> argparse.Namespace:
    """解析测试脚本参数。"""
    parser = argparse.ArgumentParser(description="端到端测试 MoeRuntimeEngine 模型推理。")
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "models" / "Qwen1.5-Moe-A2.7B"),
        help="本地模型目录。",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="默认设备，默认 cuda。",
    )
    parser.add_argument(
        "--prompt",
        default="问：你好 答：你好！有什么我可以帮你的吗? 问: 1+1=? 答：",
        help="测试用 prompt。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="最大生成 token 数。",
    )
    parser.add_argument(
        "--gpu-cache-per-layer",
        type=int,
        default=8,
        help="GPU 每层缓存专家数，默认 8。",
    )
    parser.add_argument(
        "--cpu-cache-per-layer",
        type=int,
        default=12,
        help="CPU 每层缓存专家数，默认 12。",
    )
    return parser.parse_args()


def build_cache_configs(
    args: argparse.Namespace,
    num_layers: int,
) -> tuple[dict[int, int] | None, dict[int, int] | None]:
    """根据命令行参数和模型层数，构建 GPU / CPU 每层缓存容量配置。"""
    gpu_cache_cap: dict[int, int] | None = None
    cpu_cache_cap: dict[int, int] | None = None

    if args.gpu_cache_per_layer > 0:
        gpu_cache_cap = {
            layer_idx: args.gpu_cache_per_layer
            for layer_idx in range(num_layers)
        }

    if args.cpu_cache_per_layer > 0:
        cpu_cache_cap = {
            layer_idx: args.cpu_cache_per_layer
            for layer_idx in range(num_layers)
        }

    return gpu_cache_cap, cpu_cache_cap


def main() -> None:
    """端到端测试入口。"""
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    manifest_path = model_dir / "weight_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")

    # 1. 构造 RuntimeConfig
    print("=" * 60)
    print("Step 1: 构造 RuntimeConfig")
    print("=" * 60)

    # 先加载 config 拿层数，再决定缓存配比
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_dir)
    num_layers = model_config.num_hidden_layers
    print(f"  模型层数: {num_layers}")

    gpu_cache_cap, cpu_cache_cap = build_cache_configs(args, num_layers)

    runtime_config = RuntimeConfig(
        model_dir=model_dir,
        manifest_path=manifest_path,
        non_expert_device=args.device,
        shared_expert_device=args.device,
        routed_expert_device=args.device,
        max_gpu_cache_capacity=gpu_cache_cap,
        max_cpu_cache_capacity=cpu_cache_cap,
    )

    # 2. 创建引擎
    print("\nStep 2: 创建 MoeRuntimeEngine")
    engine = MoeRuntimeEngine(runtime_config)
    print(f"  引擎已创建，adapter={engine.adapter}, tensor_loader={engine.tensor_loader}")

    # 3. 构建组件
    print("\nStep 3: 调用 build_components()")
    start = time.perf_counter()
    engine.build_components()
    build_sec = time.perf_counter() - start
    print(f"  构建完成，耗时 {build_sec:.2f}s")
    print(f"  adapter = {type(engine.adapter).__name__}")
    print(f"  model_config.model_type = {engine.model_config.model_type}")
    print(f"  tensor_loader = {type(engine.tensor_loader).__name__}")
    print(f"  cache_manager = {type(engine.cache_manager).__name__ if engine.cache_manager else None}")

    # 4. 加载模型
    print("\nStep 4: 调用 load_model()")
    start = time.perf_counter()
    engine.load_model()
    load_sec = time.perf_counter() - start
    print(f"  模型加载完成，耗时 {load_sec:.2f}s")

    # 5. 检查模型参数
    print("\nStep 5: 检查模型参数分布")
    resident_count = 0
    routed_count = 0
    device_dist: dict[str, int] = {}
    expected_device = args.device

    for name, param in engine.model_runtime.model.named_parameters():
        if engine.adapter.is_routed_expert_tensor(name):
            routed_count += 1
        else:
            resident_count += 1
        device_dist[str(param.device)] = device_dist.get(str(param.device), 0) + 1

    print(f"  resident 参数: {resident_count}")
    print(f"  routed expert 参数（仍在 meta）: {routed_count}")
    print(f"  参数设备分布: {device_dist}")

    # 6. 构建 tokenizer
    print("\nStep 6: 加载 tokenizer 并编码 prompt")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if expected_device.startswith("cuda"):
        input_ids = input_ids.to(expected_device)

    print(f"  prompt: {args.prompt!r}")
    print(f"  input_ids shape: {input_ids.shape}")

    # 7. 执行 generate
    print("\nStep 7: 调用 engine.generate()")
    if torch.cuda.is_available() and expected_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = engine.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    gen_sec = time.perf_counter() - start
    generated_tokens = output_ids.shape[1] - input_ids.shape[1]
    print(f"  生成完成，耗时 {gen_sec:.2f}s, tokens={generated_tokens}")
    print(f"  生成速度: {generated_tokens / gen_sec:.2f} tokens/s")

    # 8. 解码输出
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n  === 输出 ===")
    print(f"  {output_text}")
    print(f"  ============")

    # 9. 显存统计
    if torch.cuda.is_available() and expected_device.startswith("cuda"):
        print(f"\nStep 8: GPU 显存统计")
        print(f"  显卡总显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        print(f"  当前已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  当前已保留: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  峰值已分配: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


    print("\n" + "=" * 60)
    print("测试通过：MoeRuntimeEngine 端到端推理成功。")
    print("=" * 60)


if __name__ == "__main__":
    main()
from transformers import DeepseekV2ForCausalLM