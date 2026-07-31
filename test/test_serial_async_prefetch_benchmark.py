from __future__ import annotations

"""比较串行、异步和异步 Gate 预取三种 expert 加载模式的推理速度。"""

import argparse
import gc
from pathlib import Path
from statistics import mean
import sys
import time
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moe_runtime import MoeRuntimeEngine, RuntimeConfig


BENCHMARK_CASES = (
    ("serial", "serial", "none"),
    ("async", "async", "none"),
    ("async+prefetch", "async", "gate"),
)


def parse_args() -> argparse.Namespace:
    """
    解析三模式 benchmark 参数。

    输入：命令行参数。
    输出：解析后的 argparse Namespace。
    功能：配置模型、生成长度、缓存容量、预取距离、重复次数和运行间隔。
    """

    parser = argparse.ArgumentParser(
        description="比较串行、异步和异步 Gate 预取三种 expert 加载模式。",
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "DeepSeek-V2-Lite"),
        help="本地模型目录；也可以传入 Qwen1.5-Moe-A2.7B。",
    )
    parser.add_argument("--device", default="cuda", help="模型运行设备。")
    parser.add_argument(
        "--prompt",
        default="问：你好 答：你好！有什么我可以帮你的吗? 问: 1+1=? 答：",
        help="三种模式共同使用的生成 prompt。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="每次 generate 最多生成的 token 数。",
    )
    parser.add_argument(
        "--gpu-cache-per-layer",
        type=int,
        default=6,
        help="每个 routed expert 层的 GPU cache 容量。",
    )
    parser.add_argument(
        "--cpu-cache-per-layer",
        type=int,
        default=24,
        help="每个 routed expert 层的 CPU cache 容量。",
    )
    parser.add_argument(
        "--cache-policy",
        choices=("lru", "lfu"),
        default="lru",
        help="三种模式共同使用的缓存淘汰策略。",
    )
    parser.add_argument(
        "--prefetch-distance",
        type=int,
        default=1,
        help="异步 Gate 预取向后预测的 routed expert 层数量。",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="每种模式的运行次数；三次以上可以覆盖三种运行顺序。",
    )
    parser.add_argument(
        "--cleanup-wait-seconds",
        type=float,
        default=10.0,
        help="每次释放模型和缓存后等待的秒数。",
    )
    return parser.parse_args()


def sync_cuda() -> None:
    """
    同步当前 CUDA 设备。

    输入：无。
    输出：无。
    功能：在计时边界结清异步 kernel，确保 wall time 包含真实 GPU 执行时间。
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_cache_config(
    model_dir: Path,
    gpu_per_layer: int,
    cpu_per_layer: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    为模型所有 decoder 层生成缓存容量。

    输入：模型目录和每层 GPU、CPU expert 容量。
    输出：GPU 与 CPU 的逐层容量字典。
    功能：覆盖所有 decoder 层，非 routed expert 层由 CacheManager 自行过滤。
    """

    model_config = AutoConfig.from_pretrained(model_dir)
    all_layers = range(model_config.num_hidden_layers)
    gpu_cache = {layer_idx: gpu_per_layer for layer_idx in all_layers}
    cpu_cache = {layer_idx: cpu_per_layer for layer_idx in all_layers}
    return gpu_cache, cpu_cache


def register_phase_timer(
    engine: MoeRuntimeEngine,
) -> tuple[dict[str, float | int], list[Any]]:
    """
    注册拆分 prefill 和 decode 的模型 forward hook。

    输入：已经完成模型加载的 RuntimeEngine。
    输出：共享计时状态和可移除的 hook handles。
    功能：首次 forward 计为 prefill，第二次 forward 开始前同步 CUDA 并切换到 decode。
    """

    if engine.model_runtime is None or engine.model_runtime.model is None:
        raise RuntimeError("模型尚未加载，无法注册阶段计时 hook。")

    timer: dict[str, float | int] = {
        "forward_count": 0,
        "prefill_start": 0.0,
        "prefill_end": 0.0,
        "decode_start": 0.0,
    }

    def pre_hook(_module: Any, _args: tuple[Any, ...]) -> None:
        """
        在首次和第二次模型 forward 前记录阶段边界。

        输入：PyTorch hook 传入的模型和位置参数。
        输出：无。
        功能：首次 forward 开始 prefill 计时，第二次 forward 前结清 prefill 并开始 decode。
        """

        forward_count = int(timer["forward_count"])
        if forward_count == 0:
            sync_cuda()
            timer["prefill_start"] = time.perf_counter()
            return

        if forward_count == 1 and float(timer["prefill_end"]) == 0.0:
            # 只在阶段边界同步一次，避免逐 token 同步破坏异步加载和计算重叠。
            sync_cuda()
            boundary = time.perf_counter()
            timer["prefill_end"] = boundary
            timer["decode_start"] = boundary

    def post_hook(
        _module: Any,
        _args: tuple[Any, ...],
        _output: Any,
    ) -> None:
        """
        记录已经进入 generate 的模型 forward 次数。

        输入：PyTorch hook 传入的模型、位置参数和输出。
        输出：无。
        功能：用 forward 次数区分首次 prefill 和后续单 token decode。
        """

        timer["forward_count"] = int(timer["forward_count"]) + 1

    model = engine.model_runtime.model
    handles = [
        model.register_forward_pre_hook(pre_hook),
        model.register_forward_hook(post_hook),
    ]
    return timer, handles


def remove_hooks(handles: list[Any]) -> None:
    """
    移除当前 case 注册的计时 hook。

    输入：register_phase_timer 返回的 hook handles。
    输出：无。
    功能：防止计时逻辑影响后续 benchmark case。
    """

    for handle in handles:
        handle.remove()


def cleanup_engine(
    engine: MoeRuntimeEngine | None,
    wait_seconds: float,
) -> None:
    """
    释放一次 benchmark 使用的模型和缓存。

    输入：可能未完整初始化的 RuntimeEngine 和清理后的等待秒数。
    输出：无。
    功能：关闭 CacheManager、断开大对象引用并清理 CUDA allocator，避免下一模式复用旧 cache。
    """

    sync_cuda()
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
        torch.cuda.reset_peak_memory_stats()

    if wait_seconds > 0:
        time.sleep(wait_seconds)


def run_case(
    *,
    case_name: str,
    cache_manager_mode: str,
    prefetch_policy: str,
    repetition: int,
    tokenizer: Any,
    args: argparse.Namespace,
    model_dir: Path,
    gpu_cache: dict[int, int],
    cpu_cache: dict[int, int],
) -> dict[str, Any]:
    """
    运行一次指定 expert 加载模式。

    输入：模式名称、CacheManager 模式、预取策略、重复序号、Tokenizer、运行参数和缓存配置。
    输出：当前 case 的总耗时、阶段耗时、吞吐、生成文本和装配信息。
    功能：为每次运行创建全新 Engine，计时只覆盖 generate，不包含模型加载和资源清理。
    """

    config = RuntimeConfig(
        model_dir=model_dir,
        cache_manager_mode=cache_manager_mode,
        prefetch_policy=prefetch_policy,
        prefetch_distance=args.prefetch_distance,
        non_expert_device=args.device,
        shared_expert_device=args.device,
        routed_expert_device=args.device,
        cache_policy=args.cache_policy,
        max_gpu_cache_capacity=gpu_cache,
        max_cpu_cache_capacity=cpu_cache,
    )

    engine: MoeRuntimeEngine | None = None
    hook_handles: list[Any] = []
    try:
        engine = MoeRuntimeEngine(config)
        engine.build_components()
        engine.load_model()

        inputs = tokenizer(args.prompt, return_tensors="pt")
        inputs = {
            name: tensor.to(args.device)
            for name, tensor in inputs.items()
        }
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens <= 1:
            raise ValueError("benchmark prompt 至少需要两个 token，才能区分 prefill 和 decode。")

        phase_timer, hook_handles = register_phase_timer(engine)
        sync_cuda()
        total_start = time.perf_counter()
        with torch.inference_mode():
            output_ids = engine.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        sync_cuda()
        total_end = time.perf_counter()

        prefill_start = float(phase_timer["prefill_start"])
        prefill_end = float(phase_timer["prefill_end"])
        decode_start = float(phase_timer["decode_start"])
        forward_count = int(phase_timer["forward_count"])
        if prefill_start == 0.0:
            raise RuntimeError("模型 forward hook 未执行，无法拆分 prefill 和 decode。")
        if prefill_end == 0.0:
            # 只发生一次 forward 时，整段 generate 都属于 prefill。
            prefill_end = total_end
            decode_start = total_end

        total_seconds = max(0.0, total_end - total_start)
        prefill_seconds = max(0.0, prefill_end - prefill_start)
        decode_seconds = max(0.0, total_end - decode_start)
        generated_tokens = int(output_ids.shape[-1] - input_tokens)
        decode_steps = max(0, forward_count - 1)

        return {
            "name": case_name,
            "repetition": repetition,
            "cache_manager_mode": cache_manager_mode,
            "prefetch_policy": prefetch_policy,
            "cache_manager": type(engine.cache_manager).__name__,
            "total_seconds": total_seconds,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "generated_tokens": generated_tokens,
            "decode_steps": decode_steps,
            "generated_tokens_per_second": (
                generated_tokens / total_seconds
                if generated_tokens > 0 and total_seconds > 0
                else 0.0
            ),
            "decode_steps_per_second": (
                decode_steps / decode_seconds
                if decode_steps > 0 and decode_seconds > 0
                else 0.0
            ),
            "text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
        }
    finally:
        remove_hooks(hook_handles)
        cleanup_engine(engine, args.cleanup_wait_seconds)


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    汇总同一模式的重复运行结果。

    输入：同一模式的逐次运行结果。
    输出：各阶段平均耗时、整体吞吐和基础装配信息。
    功能：以运行级平均耗时计算加速比，并按累计 token 或 decode step 计算吞吐。
    """

    if not results:
        raise ValueError("aggregate_results 至少需要一次运行结果。")

    total_seconds = sum(float(result["total_seconds"]) for result in results)
    decode_seconds = sum(float(result["decode_seconds"]) for result in results)
    generated_tokens = sum(int(result["generated_tokens"]) for result in results)
    decode_steps = sum(int(result["decode_steps"]) for result in results)
    return {
        "name": results[0]["name"],
        "cache_manager": results[0]["cache_manager"],
        "runs": len(results),
        "total_seconds": mean(float(result["total_seconds"]) for result in results),
        "prefill_seconds": mean(float(result["prefill_seconds"]) for result in results),
        "decode_seconds": mean(float(result["decode_seconds"]) for result in results),
        "generated_tokens_per_second": (
            generated_tokens / total_seconds
            if generated_tokens > 0 and total_seconds > 0
            else 0.0
        ),
        "decode_steps_per_second": (
            decode_steps / decode_seconds
            if decode_steps > 0 and decode_seconds > 0
            else 0.0
        ),
    }


def safe_speedup(baseline_seconds: float, case_seconds: float) -> float:
    """
    计算指定模式相对串行基线的加速比。

    输入：串行基线耗时和待比较模式耗时。
    输出：基线耗时除以待比较耗时；待比较耗时为零时返回零。
    功能：统一 total、prefill 和 decode 三个阶段的加速比口径。
    """

    if case_seconds <= 0:
        return 0.0
    return baseline_seconds / case_seconds


def print_report(
    summaries: dict[str, dict[str, Any]],
    outputs_same: bool,
) -> None:
    """
    打印三种模式的平均推理速度。

    输入：按模式汇总的 benchmark 结果和生成输出一致性。
    输出：无。
    功能：以串行为基线展示 total、prefill、decode、吞吐和三阶段加速比。
    """

    serial = summaries["serial"]
    print("\n" + "=" * 126)
    print("串行 / 异步 / 异步 Gate 预取推理速度对比")
    print("=" * 126)
    print(
        "mode                total(s)  prefill(s)  decode(s)  "
        "gen tok/s  decode step/s  total x  prefill x  decode x"
    )
    print("-" * 126)
    for case_name, _, _ in BENCHMARK_CASES:
        result = summaries[case_name]
        print(
            f"{case_name:<20}"
            f"{result['total_seconds']:>9.3f}  "
            f"{result['prefill_seconds']:>10.3f}  "
            f"{result['decode_seconds']:>9.3f}  "
            f"{result['generated_tokens_per_second']:>9.3f}  "
            f"{result['decode_steps_per_second']:>13.3f}  "
            f"{safe_speedup(serial['total_seconds'], result['total_seconds']):>7.2f}x  "
            f"{safe_speedup(serial['prefill_seconds'], result['prefill_seconds']):>9.2f}x  "
            f"{safe_speedup(serial['decode_seconds'], result['decode_seconds']):>8.2f}x"
        )

    print("-" * 126)
    print(f"生成输出一致: {outputs_same}")
    for case_name, _, _ in BENCHMARK_CASES:
        result = summaries[case_name]
        print(
            f"{case_name}: cache={result['cache_manager']}, "
            f"runs={result['runs']}"
        )


def main() -> None:
    """
    运行并比较三种 expert 加载模式。

    输入：命令行参数。
    输出：无。
    功能：循环移动三种模式的运行顺序，汇总平均耗时并检查全部生成文本是否一致。
    """

    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于或等于 1。")
    if args.repeats < 1:
        raise ValueError("--repeats 必须大于或等于 1。")
    if args.repeats < 3:
        print("提示：建议使用 --repeats 3，使三种模式分别覆盖三个运行顺序位置。")

    model_dir = Path(args.model_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    gpu_cache, cpu_cache = build_cache_config(
        model_dir,
        args.gpu_cache_per_layer,
        args.cpu_cache_per_layer,
    )

    print("=" * 72)
    print("串行 / 异步 / 异步 Gate 预取推理速度 benchmark")
    print("=" * 72)
    print(f"model_dir: {model_dir}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"cache_policy: {args.cache_policy}")
    print(f"prefetch_distance: {args.prefetch_distance}")
    print(f"repeats: {args.repeats}")

    results_by_case: dict[str, list[dict[str, Any]]] = {
        case_name: []
        for case_name, _, _ in BENCHMARK_CASES
    }
    all_texts: list[str] = []
    for repetition in range(args.repeats):
        shift = repetition % len(BENCHMARK_CASES)
        order = BENCHMARK_CASES[shift:] + BENCHMARK_CASES[:shift]
        for case_name, cache_manager_mode, prefetch_policy in order:
            print(
                f"\n运行 {repetition + 1}/{args.repeats}: "
                f"mode={case_name}"
            )
            result = run_case(
                case_name=case_name,
                cache_manager_mode=cache_manager_mode,
                prefetch_policy=prefetch_policy,
                repetition=repetition + 1,
                tokenizer=tokenizer,
                args=args,
                model_dir=model_dir,
                gpu_cache=gpu_cache,
                cpu_cache=cpu_cache,
            )
            results_by_case[case_name].append(result)
            all_texts.append(result["text"])
            print(
                f"  total={result['total_seconds']:.3f}s, "
                f"prefill={result['prefill_seconds']:.3f}s, "
                f"decode={result['decode_seconds']:.3f}s"
            )

    summaries = {
        case_name: aggregate_results(results)
        for case_name, results in results_by_case.items()
    }
    print_report(
        summaries=summaries,
        outputs_same=len(set(all_texts)) == 1,
    )


if __name__ == "__main__":
    main()
