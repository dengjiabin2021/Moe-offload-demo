from __future__ import annotations

"""异步 MoE Runtime profile 运行脚本。

这个脚本只在测试侧启用 profiler。Profiler 通过 monkeypatch 包装运行时对象的实例方法，
不会修改 ``moe_runtime`` 的逻辑代码文件。
"""

import argparse
import gc
from pathlib import Path
import sys
import time
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moe_profiler import Profiler
from moe_runtime import MoeRuntimeEngine, RuntimeConfig


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 Qwen 异步 MoE Runtime profiler。")
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "Qwen1.5-Moe-A2.7B"),
        help="本地模型目录。",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="运行设备，默认 cuda。",
    )
    parser.add_argument(
        "--prompt",
        default="问：你好 答：你好！有什么我可以帮你的吗? 问: 1+1=? 答：",
        help="生成 prompt。",
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
        default=6,
        help="每个 routed expert 层的 GPU expert cache 数量。",
    )
    parser.add_argument(
        "--cpu-cache-per-layer",
        type=int,
        default=10,
        help="每个 routed expert 层的 CPU expert cache 数量。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "async_profile"),
        help="profile 报告输出目录。",
    )
    parser.add_argument(
        "--memory-sample-interval",
        type=float,
        default=0.1,
        help="后台内存采样间隔，单位秒。",
    )
    return parser.parse_args()


def build_cache_config(model_dir: str, gpu_per_layer: int, cpu_per_layer: int) -> tuple[dict[int, int], dict[int, int]]:
    """为所有模型层生成容量配置；非 expert 层由 CacheManager 自己过滤。"""

    model_config = AutoConfig.from_pretrained(model_dir)
    all_layers = range(model_config.num_hidden_layers)
    gpu_cache = {layer_idx: gpu_per_layer for layer_idx in all_layers}
    cpu_cache = {layer_idx: cpu_per_layer for layer_idx in all_layers}
    return gpu_cache, cpu_cache


def cleanup_engine(engine: MoeRuntimeEngine | None) -> None:
    """释放一次 profile 运行产生的模型、cache 和 CUDA allocator 缓存。"""

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if engine is not None and engine.cache_manager is not None:
        close_fn = getattr(engine.cache_manager, "close", None)
        if close_fn is not None:
            close_fn()

    if engine is not None:
        if engine.model_runtime is not None:
            engine.model_runtime.model = None
            engine.model_runtime.cache_manager = None
        engine.model_runtime = None
        engine.cache_manager = None
        engine.tensor_loader = None
        engine.adapter = None
        engine.model_config = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main() -> None:
    """运行异步 profile 并保存报告。"""

    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_cache, cpu_cache = build_cache_config(
        str(model_dir),
        args.gpu_cache_per_layer,
        args.cpu_cache_per_layer,
    )

    config = RuntimeConfig(
        model_dir=model_dir,
        cache_manager_mode="async",
        non_expert_device=args.device,
        shared_expert_device=args.device,
        routed_expert_device=args.device,
        max_gpu_cache_capacity=gpu_cache,
        max_cpu_cache_capacity=cpu_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    engine: MoeRuntimeEngine | None = None
    report = None

    try:
        engine = MoeRuntimeEngine(config)
        profiler = Profiler(memory_sample_interval_s=args.memory_sample_interval)
        profiler.wrap(engine, tokenizer=tokenizer)

        print("=" * 72)
        print("  异步 MoE Runtime Profiler")
        print("=" * 72)
        print(f"  model_dir: {model_dir}")
        print(f"  max_new_tokens: {args.max_new_tokens}")
        print(f"  GPU cache/layer: {args.gpu_cache_per_layer}")
        print(f"  CPU cache/layer: {args.cpu_cache_per_layer}")

        print("\n[1/4] 构建组件 ...")
        engine.build_components()
        print(f"  cache_manager: {type(engine.cache_manager).__name__}")

        print("\n[2/4] 加载模型 ...")
        engine.load_model()

        print("\n[3/4] 执行 generate ...")
        inputs = tokenizer(args.prompt, return_tensors="pt")
        inputs = {name: tensor.to(args.device) for name, tensor in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started_at = time.perf_counter()
        with torch.inference_mode():
            output_ids = engine.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started_at
        new_tokens = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])
        print(f"  generate 耗时: {elapsed_s:.2f}s")
        print(f"  生成 token 数: {new_tokens}")
        if elapsed_s > 0:
            print(f"  生成速度: {new_tokens / elapsed_s:.3f} tokens/s")

        print("\n[4/4] 生成 profile 报告 ...")
        report = profiler.report()
        report.print_summary()

        json_path = output_dir / "async_profile_report.json"
        csv_path = output_dir / "async_memory_timeline.csv"
        report.save_json(json_path)
        report.save_memory_csv(csv_path)
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
    finally:
        # report 已经停止 memory sampler；这里只负责释放运行时资源。
        cleanup_engine(engine)


if __name__ == "__main__":
    main()
