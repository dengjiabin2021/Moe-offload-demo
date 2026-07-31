from __future__ import annotations

"""对比关闭和启用 Gate 预取时 prefill、decode 阶段耗时。"""

import argparse
from dataclasses import asdict, dataclass
import gc
import json
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


@dataclass
class BenchmarkRun:
    """保存一次预取策略 benchmark 的阶段结果。

    每次运行使用全新 Engine 和 expert cache。计时字段只覆盖 generate，模型加载和
    资源清理不计入；text 用于验证两种策略没有改变生成结果。
    """

    policy: str
    repetition: int
    total_seconds: float
    prefill_seconds: float
    decode_seconds: float
    input_tokens: int
    generated_tokens: int
    forward_count: int
    decode_steps: int
    decode_steps_per_second: float | None
    text: str


def parse_args() -> argparse.Namespace:
    """
    解析预取加速比 benchmark 参数。

    输入：命令行参数。
    输出：解析后的 argparse Namespace。
    功能：配置模型、生成长度、缓存容量、预取距离、重复次数和结果目录。
    """

    parser = argparse.ArgumentParser(
        description="对比关闭和启用 Gate 预取时的 prefill/decode 耗时。",
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "DeepSeek-V2-Lite"),
        help="本地模型目录；也可以传入 Qwen1.5-Moe-A2.7B。",
    )
    parser.add_argument("--device", default="cuda", help="运行设备。")
    parser.add_argument(
        "--prompt",
        default="问：你好 答：你好！有什么我可以帮你的吗? 问: 1+1=? 答：",
        help="benchmark prompt；应至少包含两个 token，确保首次 forward 属于 prefill。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="每次运行最大生成 token 数。",
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
        help="两种预取策略共同使用的缓存淘汰策略。",
    )
    parser.add_argument(
        "--prefetch-distance",
        type=int,
        default=3,
        help="启用 Gate 预取时向后预测的 routed expert 层数量。",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="每种策略的运行次数；设置为 2 或更大时会交替执行顺序以减小顺序偏差。",
    )
    parser.add_argument(
        "--cleanup-wait-seconds",
        type=float,
        default=5.0,
        help="两次运行之间完成资源回收后等待的秒数。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "prefetch_speedup"),
        help="benchmark JSON 输出目录。",
    )
    return parser.parse_args()


def sync_cuda() -> None:
    """
    同步当前 CUDA 设备。

    输入：无。
    输出：无。
    功能：在阶段边界结清异步 kernel，保证 wall time 包含实际 GPU 执行。
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

    输入：已经加载模型的 RuntimeEngine。
    输出：共享计时状态和两个可移除 hook handle。
    功能：首次 forward 计入 prefill，从第二次 forward 开始累计 decode，不逐 token 强制同步。
    """

    if engine.model_runtime is None or engine.model_runtime.model is None:
        raise RuntimeError("模型尚未加载，无法注册阶段计时 hook。")

    model = engine.model_runtime.model
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
        功能：首次 forward 开始 prefill 计时；第二次 forward 前同步并切换到 decode。
        """

        forward_count = int(timer["forward_count"])
        if forward_count == 0:
            # 首次 forward 前结清模型加载阶段可能遗留的 CUDA 工作。
            sync_cuda()
            timer["prefill_start"] = time.perf_counter()
            return

        if forward_count == 1 and float(timer["prefill_end"]) == 0.0:
            # 只在 prefill/decode 边界同步一次，避免逐 token 同步破坏异步加载行为。
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
        记录已经完成 Python forward 调用的次数。

        输入：PyTorch hook 传入的模型、位置参数和输出。
        输出：无。
        功能：用 forward 次数区分首次 prefill 和后续单 token decode。
        """

        timer["forward_count"] = int(timer["forward_count"]) + 1

    handles = [
        model.register_forward_pre_hook(pre_hook),
        model.register_forward_hook(post_hook),
    ]
    return timer, handles


def remove_hooks(handles: list[Any]) -> None:
    """
    移除阶段计时 hook。

    输入：register_phase_timer 返回的 hook handles。
    输出：无。
    功能：确保计时 patch 只影响当前一次 generate。
    """

    for handle in handles:
        handle.remove()


def cleanup_engine(
    engine: MoeRuntimeEngine | None,
    wait_seconds: float,
) -> None:
    """
    释放一次 benchmark 运行的模型和缓存。

    输入：可能未完整初始化的 RuntimeEngine 和清理后等待秒数。
    输出：无。
    功能：关闭调度线程、断开大对象引用并清理 CUDA allocator，保证下一策略使用新 cache。
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
    policy: str,
    repetition: int,
    tokenizer: Any,
    args: argparse.Namespace,
    model_dir: Path,
    gpu_cache: dict[int, int],
    cpu_cache: dict[int, int],
) -> BenchmarkRun:
    """
    运行一次关闭或启用 Gate 预取的 benchmark。

    输入：预取策略、重复序号、Tokenizer、运行参数、模型目录和缓存容量。
    输出：一次 generate 的 prefill、decode 和总耗时结果。
    功能：每次创建全新异步 Engine，唯一实验变量是 prefetch_policy。
    """

    config = RuntimeConfig(
        model_dir=model_dir,
        cache_manager_mode="async",
        prefetch_policy=policy,
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
            raise ValueError(
                "benchmark prompt 至少需要两个 token，否则 Gate 预取会把首次 forward 当作 decode。"
            )

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
            # 只发生一次 forward 时，全部 generate 时间属于 prefill。
            prefill_end = total_end
            decode_start = total_end

        generated_tokens = int(output_ids.shape[-1] - input_tokens)
        decode_steps = max(0, forward_count - 1)
        decode_seconds = max(0.0, total_end - decode_start)
        decode_steps_per_second = (
            decode_steps / decode_seconds
            if decode_steps > 0 and decode_seconds > 0
            else None
        )
        return BenchmarkRun(
            policy=policy,
            repetition=repetition,
            total_seconds=max(0.0, total_end - total_start),
            prefill_seconds=max(0.0, prefill_end - prefill_start),
            decode_seconds=decode_seconds,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            forward_count=forward_count,
            decode_steps=decode_steps,
            decode_steps_per_second=decode_steps_per_second,
            text=tokenizer.decode(output_ids[0], skip_special_tokens=True),
        )
    finally:
        remove_hooks(hook_handles)
        cleanup_engine(engine, args.cleanup_wait_seconds)


def aggregate_runs(runs: list[BenchmarkRun]) -> dict[str, Any]:
    """
    汇总同一预取策略的重复运行。

    输入：同一 policy 的 BenchmarkRun 列表。
    输出：各阶段平均耗时、吞吐和样本数量。
    功能：使用运行级均值形成加速比分母或分子，同时保留生成 token 和 decode step 信息。
    """

    if not runs:
        raise ValueError("aggregate_runs 至少需要一次 benchmark 结果。")

    total_decode_steps = sum(run.decode_steps for run in runs)
    total_decode_seconds = sum(run.decode_seconds for run in runs)
    return {
        "policy": runs[0].policy,
        "runs": len(runs),
        "total_seconds": mean(run.total_seconds for run in runs),
        "prefill_seconds": mean(run.prefill_seconds for run in runs),
        "decode_seconds": mean(run.decode_seconds for run in runs),
        "decode_steps_per_second": (
            total_decode_steps / total_decode_seconds
            if total_decode_steps > 0 and total_decode_seconds > 0
            else None
        ),
        "input_tokens": runs[0].input_tokens,
        "generated_tokens": mean(run.generated_tokens for run in runs),
        "decode_steps": mean(run.decode_steps for run in runs),
    }


def safe_speedup(
    baseline_seconds: float,
    optimized_seconds: float,
) -> float | None:
    """
    计算启用预取相对关闭预取的加速比。

    输入：无预取基线耗时和启用预取耗时。
    输出：基线除以优化后耗时；优化后耗时为零时返回 None。
    功能：统一 total、prefill 和 decode 三个阶段的速度比较口径。
    """

    if optimized_seconds <= 0:
        return None
    return baseline_seconds / optimized_seconds


def print_report(
    baseline: dict[str, Any],
    prefetched: dict[str, Any],
    speedups: dict[str, float | None],
    outputs_same: bool,
) -> None:
    """
    打印两种策略的阶段耗时和加速比。

    输入：无预取汇总、有预取汇总、三项加速比和输出一致性。
    输出：无。
    功能：以表格展示 total、prefill、decode 时间及 decode steps/s。
    """

    print("\n" + "=" * 86)
    print("Gate 预取阶段加速比")
    print("=" * 86)
    print("模式             total(s)   prefill(s)   decode(s)   decode steps/s")
    print("-" * 86)
    for name, result in (
        ("无预取", baseline),
        ("Gate 预取", prefetched),
    ):
        throughput = result["decode_steps_per_second"]
        throughput_text = "N/A" if throughput is None else f"{throughput:.3f}"
        print(
            f"{name:<16} "
            f"{result['total_seconds']:>9.3f}   "
            f"{result['prefill_seconds']:>10.3f}   "
            f"{result['decode_seconds']:>9.3f}   "
            f"{throughput_text:>14}"
        )

    def display_speedup(value: float | None) -> str:
        """
        格式化一项阶段加速比。

        输入：加速比或 None。
        输出：带 x 后缀的两位小数字符串，或 N/A。
        功能：明确区分无法计算和没有加速。
        """

        return "N/A" if value is None else f"{value:.2f}x"

    print("-" * 86)
    print(
        "加速比（无预取 / Gate 预取）: "
        f"total={display_speedup(speedups['total'])}, "
        f"prefill={display_speedup(speedups['prefill'])}, "
        f"decode={display_speedup(speedups['decode'])}"
    )
    print(
        f"输入 tokens={baseline['input_tokens']}, "
        f"平均生成 tokens={baseline['generated_tokens']:.1f}, "
        f"平均 decode steps={baseline['decode_steps']:.1f}"
    )
    print(f"生成输出一致: {outputs_same}")


def save_report(
    report: dict[str, Any],
    output_dir: Path,
) -> Path:
    """
    保存 benchmark 完整结果。

    输入：包含配置、逐次运行、均值和加速比的报告，以及输出目录。
    输出：生成的 JSON 文件路径。
    功能：保留原始样本，避免只根据终端均值判断加速效果。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "prefetch_speedup_benchmark.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    """
    交替运行无预取和 Gate 预取并计算阶段加速比。

    输入：命令行参数。
    输出：无。
    功能：每轮重建模型与 cache，输出逐次结果、平均耗时、加速比和 JSON 报告。
    """

    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats 必须大于或等于 1。")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens 必须大于或等于 1。")
    if args.repeats == 1:
        print(
            "提示：当前每种策略只运行一次；如需减小 OS page cache 和运行顺序偏差，"
            "建议使用 --repeats 2。"
        )

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    gpu_cache, cpu_cache = build_cache_config(
        model_dir,
        args.gpu_cache_per_layer,
        args.cpu_cache_per_layer,
    )

    runs_by_policy: dict[str, list[BenchmarkRun]] = {
        "none": [],
        "gate": [],
    }
    all_runs: list[BenchmarkRun] = []
    for repetition in range(args.repeats):
        # 重复运行时交替先后顺序，减小 OS page cache 和温度变化造成的单向偏差。
        order = (
            ("none", "gate")
            if repetition % 2 == 0
            else ("gate", "none")
        )
        for policy in order:
            print(
                f"\n运行 {repetition + 1}/{args.repeats}: "
                f"prefetch_policy={policy}"
            )
            result = run_case(
                policy=policy,
                repetition=repetition + 1,
                tokenizer=tokenizer,
                args=args,
                model_dir=model_dir,
                gpu_cache=gpu_cache,
                cpu_cache=cpu_cache,
            )
            runs_by_policy[policy].append(result)
            all_runs.append(result)
            print(
                f"  total={result.total_seconds:.3f}s, "
                f"prefill={result.prefill_seconds:.3f}s, "
                f"decode={result.decode_seconds:.3f}s"
            )

    baseline = aggregate_runs(runs_by_policy["none"])
    prefetched = aggregate_runs(runs_by_policy["gate"])
    speedups = {
        "total": safe_speedup(
            baseline["total_seconds"],
            prefetched["total_seconds"],
        ),
        "prefill": safe_speedup(
            baseline["prefill_seconds"],
            prefetched["prefill_seconds"],
        ),
        "decode": safe_speedup(
            baseline["decode_seconds"],
            prefetched["decode_seconds"],
        ),
    }
    baseline_texts = [run.text for run in runs_by_policy["none"]]
    prefetched_texts = [run.text for run in runs_by_policy["gate"]]
    outputs_same = (
        len(set(baseline_texts)) == 1
        and len(set(prefetched_texts)) == 1
        and baseline_texts[0] == prefetched_texts[0]
    )

    print_report(baseline, prefetched, speedups, outputs_same)
    report = {
        "definitions": {
            "prefill": "第一次模型 forward；在第二次 forward 开始前同步 CUDA 并结束计时。",
            "decode": "从第二次模型 forward 开始到 generate 返回并同步 CUDA 的总时间。",
            "decode_steps": "模型首次 forward 产生第一个新 token，后续 forward 次数计为 decode steps。",
            "speedup": "无预取平均耗时除以 Gate 预取平均耗时；大于 1 表示预取加速。",
            "scope": "模型加载、Engine 构建、资源清理和两次运行之间的等待不计入耗时。",
        },
        "config": {
            "model_dir": str(model_dir),
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "gpu_cache_per_layer": args.gpu_cache_per_layer,
            "cpu_cache_per_layer": args.cpu_cache_per_layer,
            "cache_policy": args.cache_policy,
            "prefetch_distance": args.prefetch_distance,
            "repeats": args.repeats,
        },
        "runs": [asdict(run) for run in all_runs],
        "mean": {
            "no_prefetch": baseline,
            "gate_prefetch": prefetched,
        },
        "speedup": speedups,
        "outputs_same": outputs_same,
    }
    output_path = save_report(report, output_dir)
    print(f"JSON: {output_path}")


if __name__ == "__main__":
    main()
