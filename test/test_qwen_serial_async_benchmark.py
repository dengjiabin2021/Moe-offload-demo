from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moe_runtime import MoeRuntimeEngine, RuntimeConfig


MODEL_DIR = PROJECT_ROOT / "models" / "DeepSeek-V2-Lite"
DEVICE = "cuda"
MAX_NEW_TOKENS = 32
GPU_CACHE_PER_LAYER = 6
CPU_CACHE_PER_LAYER = 24
PROMPT = "问：你好 答：你好！有什么我可以帮你的吗? 问: 1+1=? 答："
CLEANUP_WAIT_SECONDS = 5
TRACE_ROUTER = False


def parse_args() -> argparse.Namespace:
    """解析 benchmark 运行参数。"""

    parser = argparse.ArgumentParser(description="对比 Qwen 串行/异步 expert 加载耗时。")
    parser.add_argument(
        "--cache-policy",
        choices=("lru", "lfu"),
        default="lru",
        help="缓存淘汰策略：lru 覆盖最久未使用的专家；lfu 覆盖访问频率最低的专家。",
    )
    return parser.parse_args()


def sync_cuda() -> None:
    """同步 CUDA，保证计时统计的是实际执行耗时。"""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup(engine: MoeRuntimeEngine | None) -> None:
    """释放上一次运行留下的模型和缓存。"""

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

    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

    print(f"清理完成，等待 {CLEANUP_WAIT_SECONDS}s ...")
    time.sleep(CLEANUP_WAIT_SECONDS)


def build_cache_config() -> tuple[dict[int, int], dict[int, int]]:
    """给所有模型层配置缓存容量，非 routed expert 层由 CacheManager 自己过滤。"""

    model_config = AutoConfig.from_pretrained(MODEL_DIR)
    all_layers = range(model_config.num_hidden_layers)
    gpu_cache = {layer_idx: GPU_CACHE_PER_LAYER for layer_idx in all_layers}
    cpu_cache = {layer_idx: CPU_CACHE_PER_LAYER for layer_idx in all_layers}
    return gpu_cache, cpu_cache


def wrap_cache_manager(cache_manager: Any, mode: str) -> dict[str, int]:
    """给 CacheManager 加调用计数器。"""

    if mode == "serial":
        method_names = ("get_expert",)
    else:
        method_names = (
            "request_expert",
            "is_expert_ready",
            "wait_any_ready",
            "get_ready_expert",
            "release_expert",
        )

    counters = {name: 0 for name in method_names}

    for name in method_names:
        original = getattr(cache_manager, name)
        setattr(cache_manager, name, make_counter(counters, name, original))

    return counters


def make_counter(
    counters: dict[str, int],
    name: str,
    method: Callable[..., Any],
) -> Callable[..., Any]:
    """构造单个 CacheManager 方法包装器。"""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        counters[name] += 1
        return method(*args, **kwargs)

    return wrapper


def wrap_expert_forward_trace(engine: MoeRuntimeEngine) -> list[list[object]]:
    """记录每次 routed expert forward 的 token 级 top-k expert。"""

    trace: list[list[object]] = []
    model = engine.model_runtime.model

    for module in model.modules():
        if not all(hasattr(module, name) for name in ("layer_idx", "num_experts", "cache_manager")):
            continue

        original_forward = module.forward
        layer_idx = int(module.layer_idx)

        def make_forward_wrapper(original: Callable[..., Any], current_layer_idx: int) -> Callable[..., Any]:
            def wrapper(hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> Any:
                top_k_experts = top_k_index.detach().cpu().tolist()
                trace.append([len(trace), current_layer_idx, top_k_experts])
                return original(hidden_states, top_k_index, top_k_weights)

            return wrapper

        module.forward = make_forward_wrapper(original_forward, layer_idx)

    return trace


def register_phase_timer(engine: MoeRuntimeEngine) -> tuple[dict[str, float | int], list[Any]]:
    """注册 generate 阶段计时 hook，用于拆分 prefill 和 decode。"""

    model = engine.model_runtime.model
    timer: dict[str, float | int] = {
        "forward_count": 0,
        "prefill_start": 0.0,
        "prefill_end": 0.0,
        "decode_start": 0.0,
    }

    def pre_hook(module: Any, args: tuple[Any, ...]) -> None:
        forward_count = int(timer["forward_count"])

        if forward_count == 0:
            # 第一次模型 forward 是 prefill；先同步，避免把前置 CUDA 工作算进去。
            sync_cuda()
            timer["prefill_start"] = time.perf_counter()
            return

        if forward_count == 1 and float(timer["prefill_end"]) == 0.0:
            # 第二次模型 forward 开始前，第一次 forward 的 CUDA 工作应全部结清。
            # 这里作为 prefill/decode 的边界，后续时间归入 decode。
            sync_cuda()
            now = time.perf_counter()
            timer["prefill_end"] = now
            timer["decode_start"] = now

    def post_hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
        timer["forward_count"] = int(timer["forward_count"]) + 1

    handles = [
        model.register_forward_pre_hook(pre_hook),
        model.register_forward_hook(post_hook),
    ]
    return timer, handles


def remove_hooks(handles: list[Any]) -> None:
    """移除测试脚本注册的 hook，避免影响后续模式运行。"""

    for handle in handles:
        handle.remove()


def run(
    mode: str,
    tokenizer: AutoTokenizer,
    gpu_cache: dict[int, int],
    cpu_cache: dict[int, int],
    cache_policy: str,
) -> dict[str, object]:
    """运行一次 serial 或 async 模式，并返回生成耗时。"""

    config = RuntimeConfig(
        model_dir=MODEL_DIR,
        cache_manager_mode=mode,
        non_expert_device=DEVICE,
        shared_expert_device=DEVICE,
        routed_expert_device=DEVICE,
        cache_policy=cache_policy,
        max_gpu_cache_capacity=gpu_cache,
        max_cpu_cache_capacity=cpu_cache,
    )

    engine: MoeRuntimeEngine | None = None
    try:
        engine = MoeRuntimeEngine(config)
        engine.build_components()
        engine.load_model()
        counters = wrap_cache_manager(engine.cache_manager, mode)
        expert_trace = wrap_expert_forward_trace(engine) if TRACE_ROUTER else []

        inputs = tokenizer(PROMPT, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        phase_timer, hook_handles = register_phase_timer(engine)

        sync_cuda()
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                output_ids = engine.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            sync_cuda()
        finally:
            remove_hooks(hook_handles)

        end = time.perf_counter()
        generate_sec = end - start
        prefill_start = float(phase_timer["prefill_start"])
        prefill_end = float(phase_timer["prefill_end"])
        decode_start = float(phase_timer["decode_start"])
        if prefill_end == 0.0:
            # 如果只发生了一次 forward，则整段 generate 都属于 prefill。
            prefill_end = end
            decode_start = end

        new_tokens = output_ids.shape[-1] - inputs["input_ids"].shape[-1]
        stats_fn = getattr(engine.cache_manager, "stats", None)
        return {
            "mode": mode,
            "cache_manager": type(engine.cache_manager).__name__,
            "time": generate_sec,
            "prefill_time": max(0.0, prefill_end - prefill_start),
            "decode_time": max(0.0, end - decode_start),
            "prefill_tokens": int(inputs["input_ids"].shape[-1]),
            "decode_tokens": int(new_tokens),
            "tokens": new_tokens,
            "text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
            "counters": counters,
            "stats": stats_fn() if stats_fn is not None else {},
            "experts": expert_trace,
        }
    finally:
        cleanup(engine)


def print_expert_compare(serial_experts: list[list[object]], async_experts: list[list[object]]) -> None:
    """逐位置比较两次推理中每层每次 forward 的 token 级 expert 选择。"""

    same_layer_choices = serial_experts == async_experts

    print(f"forward calls: serial={len(serial_experts)}, async={len(async_experts)}")
    print(f"forward expert choices same: {same_layer_choices}")

    if same_layer_choices:
        return

    for index, (serial_item, async_item) in enumerate(zip(serial_experts, async_experts)):
        if serial_item != async_item:
            print(f"first forward choice diff: index={index}, serial={serial_item}, async={async_item}")
            print_first_token_diff(serial_item, async_item)
            return

    print(f"forward call length diff: serial={len(serial_experts)}, async={len(async_experts)}")


def print_first_token_diff(serial_item: list[object], async_item: list[object]) -> None:
    """打印某次 forward 内第一个 token 级 top-k 差异。"""

    serial_forward_idx, serial_layer_idx, serial_top_k = serial_item
    async_forward_idx, async_layer_idx, async_top_k = async_item

    if serial_forward_idx != async_forward_idx or serial_layer_idx != async_layer_idx:
        print(
            "forward header diff: "
            f"serial=({serial_forward_idx}, {serial_layer_idx}), "
            f"async=({async_forward_idx}, {async_layer_idx})"
        )
        return

    for token_idx, (serial_token_experts, async_token_experts) in enumerate(zip(serial_top_k, async_top_k)):
        if serial_token_experts != async_token_experts:
            print(
                "first token top-k diff: "
                f"forward_idx={serial_forward_idx}, "
                f"layer_idx={serial_layer_idx}, "
                f"token_idx={token_idx}, "
                f"serial={serial_token_experts}, "
                f"async={async_token_experts}"
            )
            return

    print(f"token count diff: serial={len(serial_top_k)}, async={len(async_top_k)}")


def main() -> None:
    """对比 Qwen 模型串行和异步 expert 加载耗时。"""

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    gpu_cache, cpu_cache = build_cache_config()

    print("=" * 60)
    print("Qwen 串行 / 异步 expert 加载对比")
    print("=" * 60)
    print(f"cache_policy: {args.cache_policy}")

    async_result = run("async", tokenizer, gpu_cache, cpu_cache, args.cache_policy)
    serial = run("serial", tokenizer, gpu_cache, cpu_cache, args.cache_policy)

    serial_time = float(serial["time"])
    async_time = float(async_result["time"])
    speedup = serial_time / async_time if async_time > 0 else 0.0
    serial_prefill = float(serial["prefill_time"])
    async_prefill = float(async_result["prefill_time"])
    serial_decode = float(serial["decode_time"])
    async_decode = float(async_result["decode_time"])
    prefill_speedup = serial_prefill / async_prefill if async_prefill > 0 else 0.0
    decode_speedup = serial_decode / async_decode if async_decode > 0 else 0.0

    print(f"serial: {serial_time:.2f}s, cache={serial['cache_manager']}")
    print(f"async:  {async_time:.2f}s, cache={async_result['cache_manager']}")
    print()
    print("stage time:")
    print("mode      total(s)   prefill(s)   decode(s)   decode tokens/s")
    print("-" * 61)
    print(
        f"serial   {serial_time:>8.2f}   "
        f"{serial_prefill:>10.2f}   "
        f"{serial_decode:>9.2f}   "
        f"{float(serial['decode_tokens']) / serial_decode if serial_decode > 0 else 0.0:>15.3f}"
    )
    print(
        f"async    {async_time:>8.2f}   "
        f"{async_prefill:>10.2f}   "
        f"{async_decode:>9.2f}   "
        f"{float(async_result['decode_tokens']) / async_decode if async_decode > 0 else 0.0:>15.3f}"
    )
    print(
        f"speedup  {speedup:>8.2f}x  "
        f"{prefill_speedup:>10.2f}x  "
        f"{decode_speedup:>9.2f}x"
    )
    print(f"tokens: {serial['tokens']}")
    print(f"speedup: {speedup:.2f}x")
    print(f"output same: {serial['text'] == async_result['text']}")
    print("\nserial text:")
    print(serial["text"])
    print("\nasync text:")
    print(async_result["text"])
    print(f"\nserial counters: {serial['counters']}")
    print(f"async counters: {async_result['counters']}")
    print(f"async stats: {async_result['stats']}")
    if TRACE_ROUTER:
        print()
        print_expert_compare(
            serial_experts=serial["experts"],
            async_experts=async_result["experts"],
        )
    else:
        print("\nrouter trace: disabled")


if __name__ == "__main__":
    main()
