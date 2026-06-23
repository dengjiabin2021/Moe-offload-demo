from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoConfig, AutoTokenizer

# 将项目根目录加入 sys.path，保证直接运行本脚本时可以导入 moe_runtime。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import moe_runtime.adapters.qwen15_moe_adapter as qwen_adapter_module
import moe_runtime.runtime_engine as runtime_engine_module
from moe_runtime.blocks.qwen2_moe_block import Qwen2OnDemandExperts as SerialQwen2OnDemandExperts
from moe_runtime.blocks.async_qwen2_moe_block import Qwen2OnDemandExperts as AsyncQwen2OnDemandExperts
from moe_runtime.cache_manager.cache_manager import ExpertCacheManager as SerialExpertCacheManager
from moe_runtime.cache_manager.sync_cache_manager import ExpertCacheManager as AsyncExpertCacheManager
from moe_runtime.config import RuntimeConfig


@dataclass
class RuntimeCase:
    """一次 benchmark 的运行模式配置。"""

    name: str
    title: str
    cache_manager_cls: type
    block_cls: type
    counter_methods: tuple[str, ...]


@dataclass
class BenchmarkResult:
    """一次 benchmark 的主要结果。"""

    name: str
    build_time_s: float
    load_time_s: float
    generate_time_s: float
    generated_tokens: int
    tokens_per_second: float
    cache_manager_name: str
    expert_module_count: int
    counters: dict[str, int]
    peak_gpu_gb: float | None
    output_text: str


def parse_args() -> argparse.Namespace:
    """解析对比测试脚本参数。"""

    parser = argparse.ArgumentParser(description="对比串行和异步 MoE expert 加载路径的端到端耗时。")
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "models" / "Qwen1.5-Moe-A2.7B"),
        help="本地 Qwen1.5-MoE / qwen2_moe 模型目录。",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="默认设备，默认 cuda。串行旧 CacheManager 当前只支持 cuda。",
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
        help="最大生成 token 数，默认 32。",
    )
    parser.add_argument(
        "--gpu-cache-per-layer",
        type=int,
        default=4,
        help="GPU 每层缓存 expert 数，两种模式使用同一配置。",
    )
    parser.add_argument(
        "--cpu-cache-per-layer",
        type=int,
        default=8,
        help="CPU 每层缓存 expert 数，两种模式使用同一配置。",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=5.0,
        help="异步模式推理结束后等待后台状态清空的最长秒数。",
    )
    parser.add_argument(
        "--order",
        choices=("serial-first", "async-first"),
        default="serial-first",
        help="运行顺序。默认先跑串行，再跑异步。",
    )
    return parser.parse_args()


def build_cache_configs(
    args: argparse.Namespace,
    num_layers: int,
) -> tuple[dict[int, int] | None, dict[int, int] | None]:
    """根据命令行参数和模型层数构建每层 CPU/GPU cache 容量。"""

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


def build_runtime_config(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    gpu_cache_cap: dict[int, int] | None,
    cpu_cache_cap: dict[int, int] | None,
) -> RuntimeConfig:
    """为两种运行模式构建完全相同的 RuntimeConfig。"""

    return RuntimeConfig(
        model_dir=args.model_dir,
        manifest_path=manifest_path,
        non_expert_device=args.device,
        shared_expert_device=args.device,
        routed_expert_device=args.device,
        max_gpu_cache_capacity=gpu_cache_cap,
        max_cpu_cache_capacity=cpu_cache_cap,
    )


def make_cases(order: str) -> list[RuntimeCase]:
    """生成 benchmark 运行顺序。"""

    serial_case = RuntimeCase(
        name="serial",
        title="串行加载",
        cache_manager_cls=SerialExpertCacheManager,
        block_cls=SerialQwen2OnDemandExperts,
        counter_methods=("get_expert",),
    )
    async_case = RuntimeCase(
        name="async",
        title="异步加载",
        cache_manager_cls=AsyncExpertCacheManager,
        block_cls=AsyncQwen2OnDemandExperts,
        counter_methods=(
            "request_expert",
            "is_expert_ready",
            "wait_any_ready",
            "get_ready_expert",
            "release_expert",
        ),
    )

    if order == "async-first":
        return [async_case, serial_case]
    return [serial_case, async_case]


def wrap_cache_counters(cache_manager: Any, method_names: tuple[str, ...]) -> dict[str, int]:
    """给 CacheManager 指定接口加调用计数器。"""

    counters = {method_name: 0 for method_name in method_names}
    for method_name in method_names:
        original = getattr(cache_manager, method_name)
        setattr(cache_manager, method_name, _counting_wrapper(counters, method_name, original))
    return counters


def _counting_wrapper(
    counters: dict[str, int],
    method_name: str,
    method: Callable[..., Any],
) -> Callable[..., Any]:
    """构造单个方法的计数包装器。"""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        counters[method_name] += 1
        return method(*args, **kwargs)

    return wrapper


def patch_runtime_for_case(case: RuntimeCase) -> tuple[type, type]:
    """临时切换 RuntimeEngine 和 Qwen adapter 使用的 cache/block 实现。"""

    old_cache_cls = runtime_engine_module.ExpertCacheManager
    old_block_cls = qwen_adapter_module.Qwen2OnDemandExperts

    runtime_engine_module.ExpertCacheManager = case.cache_manager_cls
    qwen_adapter_module.Qwen2OnDemandExperts = case.block_cls

    return old_cache_cls, old_block_cls


def restore_runtime_patch(old_cache_cls: type, old_block_cls: type) -> None:
    """恢复 benchmark 前的全局 class 绑定。"""

    runtime_engine_module.ExpertCacheManager = old_cache_cls
    qwen_adapter_module.Qwen2OnDemandExperts = old_block_cls


def move_inputs_to_device(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """把 tokenizer 输出移动到目标设备。"""

    if not device.startswith("cuda"):
        return inputs

    return {
        name: value.to(device)
        for name, value in inputs.items()
    }


def count_expert_modules(engine: Any, block_cls: type) -> int:
    """统计模型中被替换成当前 case block 的专家模块数。"""

    return sum(
        1
        for module in engine.model_runtime.model.modules()
        if isinstance(module, block_cls)
    )


def synchronize_if_needed(device: str) -> None:
    """CUDA 模式下同步设备，保证耗时统计包含真实 GPU 工作。"""

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_engine(engine: Any | None) -> None:
    """释放一次 benchmark 创建的模型和 cache，降低下一次运行的显存干扰。"""

    if engine is not None and getattr(engine, "cache_manager", None) is not None:
        close_fn = getattr(engine.cache_manager, "close", None)
        if close_fn is not None:
            close_fn()

    if engine is not None:
        # 显式断开 RuntimeEngine -> ModelRuntime -> model/cache 的引用链。
        # 仅在 cleanup_engine 内 del 形参不够，因为 run_case 外层局部变量仍然持有 engine。
        if getattr(engine, "model_runtime", None) is not None:
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
        torch.cuda.synchronize()


def prepare_cuda_for_case(device: str) -> None:
    """在单个 case 开始前清理 CUDA allocator，并重置峰值统计。"""

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def cuda_peak_allocated_gb(device: str) -> float | None:
    """读取当前 case 内 PyTorch CUDA allocator 的峰值 allocated 显存。"""

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**3


def cuda_total_memory_gb(device: str) -> float | None:
    """读取当前 CUDA 设备的物理显存大小。"""

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None

    device_index = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device_index).total_memory / 1024**3


def wait_async_cache_idle(cache_manager: Any, timeout_s: float) -> bool:
    """等待异步 CacheManager 后台队列清空。"""

    if not hasattr(cache_manager, "_cv"):
        return True

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with cache_manager._cv:
            idle = (
                not cache_manager._requests
                and not cache_manager._loading_cpu
                and not cache_manager._loading_gpu
                and not cache_manager._compute_pending
            )
            if idle:
                return True

        time.sleep(0.01)

    return False


def run_case(
    *,
    case: RuntimeCase,
    args: argparse.Namespace,
    manifest_path: Path,
    gpu_cache_cap: dict[int, int] | None,
    cpu_cache_cap: dict[int, int] | None,
    tokenizer: Any,
) -> BenchmarkResult:
    """运行单个模式，并返回耗时与调用统计。"""

    print("\n" + "=" * 64)
    print(f"  运行模式: {case.title} ({case.name})")
    print("=" * 64)

    old_cache_cls, old_block_cls = patch_runtime_for_case(case)
    engine = None
    encoded = None
    input_ids = None
    attention_mask = None
    output_ids = None
    try:
        prepare_cuda_for_case(args.device)

        runtime_config = build_runtime_config(
            args=args,
            manifest_path=manifest_path,
            gpu_cache_cap=gpu_cache_cap,
            cpu_cache_cap=cpu_cache_cap,
        )

        print("\n[1/5] 构建组件 ...")
        start = time.perf_counter()
        engine = runtime_engine_module.MoeRuntimeEngine(runtime_config)
        engine.build_components()
        build_time_s = time.perf_counter() - start
        print(f"  构建完成: {build_time_s:.2f}s")
        print(f"  adapter: {type(engine.adapter).__name__}")
        print(f"  cache_manager: {type(engine.cache_manager).__name__}")

        print("\n[2/5] 加载模型并替换 routed experts ...")
        start = time.perf_counter()
        engine.load_model()
        load_time_s = time.perf_counter() - start
        expert_module_count = count_expert_modules(engine, case.block_cls)
        if expert_module_count <= 0:
            raise AssertionError(f"{case.title} 未注入预期 expert block: {case.block_cls.__name__}")
        print(f"  加载完成: {load_time_s:.2f}s")
        print(f"  expert block 数量: {expert_module_count}")

        counters = wrap_cache_counters(engine.cache_manager, case.counter_methods)

        print("\n[3/5] 编码输入 ...")
        encoded = tokenizer(args.prompt, return_tensors="pt")
        encoded = move_inputs_to_device(encoded, args.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        print(f"  input_ids shape: {tuple(input_ids.shape)}")
        print(f"  max_new_tokens: {args.max_new_tokens}")

        print("\n[4/5] 执行 generate ...")
        synchronize_if_needed(args.device)

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = engine.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        synchronize_if_needed(args.device)
        generate_time_s = time.perf_counter() - start

        generated_tokens = int(output_ids.shape[1] - input_ids.shape[1])
        tokens_per_second = generated_tokens / generate_time_s if generate_time_s > 0 else 0.0
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"  生成完成: {generate_time_s:.2f}s")
        print(f"  生成 token 数: {generated_tokens}")
        print(f"  生成速度: {tokens_per_second:.2f} tokens/s")

        print("\n[5/5] 检查 CacheManager 调用 ...")
        print(f"  调用计数: {counters}")
        for method_name, count in counters.items():
            if count <= 0:
                raise AssertionError(f"{case.title} 的 {method_name} 未被调用。")

        if case.name == "async" and not wait_async_cache_idle(engine.cache_manager, args.idle_timeout):
            raise AssertionError("异步 CacheManager 未在超时时间内清空后台请求/加载/计算状态。")

        peak_gpu_gb = cuda_peak_allocated_gb(args.device)
        total_gpu_gb = cuda_total_memory_gb(args.device)
        if peak_gpu_gb is not None:
            print(f"  峰值显存: {peak_gpu_gb:.2f} GB")
            if total_gpu_gb is not None:
                print(f"  物理显存: {total_gpu_gb:.2f} GB")
                if peak_gpu_gb > total_gpu_gb + 0.01:
                    raise AssertionError(
                        f"峰值显存统计异常: peak={peak_gpu_gb:.2f}GB > total={total_gpu_gb:.2f}GB"
                    )

        result = BenchmarkResult(
            name=case.name,
            build_time_s=build_time_s,
            load_time_s=load_time_s,
            generate_time_s=generate_time_s,
            generated_tokens=generated_tokens,
            tokens_per_second=tokens_per_second,
            cache_manager_name=type(engine.cache_manager).__name__,
            expert_module_count=expert_module_count,
            counters=dict(counters),
            peak_gpu_gb=peak_gpu_gb,
            output_text=output_text,
        )

        # 这些局部变量可能持有 CUDA tensor。必须在下一轮 case 开始前显式释放，
        # 否则上一轮 serial 的输入/输出/模型会污染 async 的 peak memory。
        encoded = None
        input_ids = None
        attention_mask = None
        output_ids = None
        cleanup_engine(engine)
        engine = None

        return result

    finally:
        restore_runtime_patch(old_cache_cls, old_block_cls)
        if engine is not None:
            encoded = None
            input_ids = None
            attention_mask = None
            output_ids = None
            cleanup_engine(engine)


def print_summary(results: list[BenchmarkResult]) -> None:
    """打印串行和异步模式的对比结果。"""

    print("\n" + "=" * 64)
    print("  串行 vs 异步 对比结果")
    print("=" * 64)
    print(
        f"{'模式':<10}"
        f"{'生成耗时(s)':>14}"
        f"{'tokens/s':>12}"
        f"{'加载耗时(s)':>14}"
        f"{'峰值显存(GB)':>14}"
    )

    for result in results:
        peak = "-" if result.peak_gpu_gb is None else f"{result.peak_gpu_gb:.2f}"
        print(
            f"{result.name:<10}"
            f"{result.generate_time_s:>14.2f}"
            f"{result.tokens_per_second:>12.2f}"
            f"{result.load_time_s:>14.2f}"
            f"{peak:>14}"
        )

    result_map = {result.name: result for result in results}
    if "serial" in result_map and "async" in result_map:
        serial = result_map["serial"]
        async_result = result_map["async"]
        speedup = serial.generate_time_s / async_result.generate_time_s
        delta = serial.generate_time_s - async_result.generate_time_s
        print("\n  生成阶段对比:")
        print(f"    串行耗时: {serial.generate_time_s:.2f}s")
        print(f"    异步耗时: {async_result.generate_time_s:.2f}s")
        print(f"    节省时间: {delta:.2f}s")
        print(f"    加速比:   {speedup:.2f}x")

        same_output = serial.output_text == async_result.output_text
        print(f"    输出文本一致: {same_output}")


def main() -> None:
    """对比串行和异步 expert 加载路径的入口。"""

    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    manifest_path = model_dir / "weight_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if args.device != "cuda":
        raise RuntimeError("当前对比脚本需要 --device cuda，因为旧串行 CacheManager 内部固定使用 cuda。")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，无法公平对比旧串行 CacheManager 和新异步 CacheManager。")

    print("=" * 64)
    print("  MoE Runtime 串行/异步 expert 加载对比测试")
    print("=" * 64)

    model_config = AutoConfig.from_pretrained(model_dir)
    if model_config.model_type != "qwen2_moe":
        raise RuntimeError("当前对比脚本只验证 Qwen async/serial block 路径。")

    num_layers = model_config.num_hidden_layers
    gpu_cache_cap, cpu_cache_cap = build_cache_configs(args, num_layers)

    print(f"  模型目录: {model_dir}")
    print(f"  模型类型: {model_config.model_type}")
    print(f"  模型层数: {num_layers}")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  GPU cache/layer: {args.gpu_cache_per_layer}")
    print(f"  CPU cache/layer: {args.cpu_cache_per_layer}")
    print(f"  运行顺序: {args.order}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    results: list[BenchmarkResult] = []
    for case in make_cases(args.order):
        results.append(
            run_case(
                case=case,
                args=args,
                manifest_path=manifest_path,
                gpu_cache_cap=gpu_cache_cap,
                cpu_cache_cap=cpu_cache_cap,
                tokenizer=tokenizer,
            )
        )

    print_summary(results)


if __name__ == "__main__":
    main()
