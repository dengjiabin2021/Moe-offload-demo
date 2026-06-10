#!/usr/bin/env python3
"""Profiler 功能测试脚本。

测试 moe_profiler 的端到端探测能力，验证：
1. 零侵入集成（仅一行 profiler.wrap(engine)）
2. 耗时采集（加载 / 预填充 / 解码，含 GPU compute + 数据搬运拆分）
3. 内存时间线采集（后台线程周期采样 GPU / CPU 内存）
4. 吞吐量采集（磁盘 IO / CPU→GPU 传输）
5. 缓存统计采集（GPU hit / CPU hit / disk fetch，按阶段拆分）
6. 报告输出（终端摘要 + JSON 导出）

用法：
    cd <repo_root>
    python test/test_profiler.py
    python test/test_profiler.py --max-new-tokens 64
    python test/test_profiler.py --gpu-cache-per-layer 4 --cpu-cache-per-layer 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 将项目根目录加入 sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer, AutoConfig

from moe_runtime.config import RuntimeConfig
from moe_runtime.runtime_engine import MoeRuntimeEngine
from moe_profiler import Profiler


# ======================================================================
# 命令行参数
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profiler 端到端测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test/test_profiler.py
  python test/test_profiler.py --max-new-tokens 64
  python test/test_profiler.py --gpu-cache-per-layer 4 --cpu-cache-per-layer 4
  python test/test_profiler.py --prompt "中国的首都是" --max-new-tokens 128
        """,
    )
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "models" / "DeepSeek-V2-Lite"),
        help="本地模型目录",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="默认设备 (default: cuda)",
    )
    parser.add_argument(
        "--prompt",
        default="中国和美国的首都分别在哪？",
        help="测试用 prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="最大生成 token 数 (default: 32)",
    )
    parser.add_argument(
        "--gpu-cache-per-layer",
        type=int,
        default=4,
        help="GPU 每层缓存专家数 (default: 4)",
    )
    parser.add_argument(
        "--cpu-cache-per-layer",
        type=int,
        default=4,
        help="CPU 每层缓存专家数 (default: 16)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" ),
        help="报告输出目录 (default: /results)",
    )
    return parser.parse_args()


# ======================================================================
# 辅助函数
# ======================================================================

def build_cache_configs(
    args: argparse.Namespace, num_layers: int
) -> tuple[dict[int, int] | None, dict[int, int] | None]:
    """构建 GPU / CPU 每层缓存容量配置。"""
    gpu_cap = None
    cpu_cap = None
    if args.gpu_cache_per_layer > 0:
        gpu_cap = {i: args.gpu_cache_per_layer for i in range(num_layers)}
    if args.cpu_cache_per_layer > 0:
        cpu_cap = {i: args.cpu_cache_per_layer for i in range(num_layers)}
    return gpu_cap, cpu_cap


def check_report_validity(report) -> dict[str, bool]:
    """验证报告各字段是否合理，返回每个检查项的结果。"""
    checks = {}

    # 耗时检查
    checks["timing_loading_time"] = report.loading_time_s > 0
    checks["timing_prefill_time"] = report.prefill_time_s > 0
    checks["timing_decode_time"] = report.decode_time_s > 0
    checks["timing_decode_per_token"] = report.decode_gpu_time_per_token_s > 0
    checks["timing_prefill_tokens"] = report.prefill_token_count > 0
    checks["timing_decode_tokens"] = report.decode_token_count > 0

    # 内存检查（至少有一个采样点）
    checks["memory_has_samples"] = len(report.memory_timeline) > 0

    # 内存峰值检查
    checks["memory_loading_gpu"] = report.loading_peak_gpu_mb > 0
    # 预填充和解码阶段 GPU 显存峰值取决于实际运行，不一定都 > 0

    # 吞吐量检查（磁盘 IO 在加载阶段肯定有值）
    checks["throughput_loading_disk"] = report.loading_disk_throughput_mbps > 0

    # 缓存检查（如果配置了缓存）
    total_requests = report.gpu_cache_hits + report.cpu_cache_hits + report.disk_fetches
    checks["cache_has_requests"] = total_requests > 0
    checks["cache_hit_rate_valid"] = 0.0 <= report.gpu_cache_hit_rate <= 1.0
    checks["cache_cpu_rate_valid"] = 0.0 <= report.cpu_cache_hit_rate <= 1.0
    checks["cache_per_layer"] = len(report.layer_cache_stats) > 0

    # 数据搬运检查
    checks["data_move_prefill"] = report.prefill_data_time_s >= 0
    checks["data_move_decode"] = report.decode_data_time_s >= 0

    return checks


# ======================================================================
# 主流程
# ======================================================================

def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()

    # ---- 验证前置条件 ----
    manifest_path = model_dir / "weight_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not torch.cuda.is_available() and args.device == "cuda":
        print("[WARN] CUDA 不可用，回退到 CPU（性能会非常慢）")
        args.device = "cpu"

    # ---- 准备输出目录 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Step 1: 构造 RuntimeConfig
    # ==================================================================
    print("=" * 64)
    print("  Profiler 端到端测试")
    print("=" * 64)

    print("\n[1/7] 加载模型配置 ...")
    model_config = AutoConfig.from_pretrained(model_dir)
    num_layers = model_config.num_hidden_layers
    gpu_cap, cpu_cap = build_cache_configs(args, num_layers)
    print(f"  模型: {model_config.model_type}")
    print(f"  层数: {num_layers}")
    print(f"  GPU 缓存/层: {args.gpu_cache_per_layer}")
    print(f"  CPU 缓存/层: {args.cpu_cache_per_layer}")

    runtime_config = RuntimeConfig(
        model_dir=model_dir,
        manifest_path=manifest_path,
        non_expert_device=args.device,
        shared_expert_device=args.device,
        routed_expert_device=args.device,
        max_gpu_cache_capacity=gpu_cap,
        max_cpu_cache_capacity=cpu_cap,
    )

    # ==================================================================
    # Step 2: 创建引擎 + 注入 Profiler
    # ==================================================================
    print("\n[2/7] 创建 MoeRuntimeEngine 并注入 Profiler ...")
    engine = MoeRuntimeEngine(runtime_config)
    profiler = Profiler()

    # ★ 唯一需要添加的一行 ★
    profiler.wrap(engine)

    print(f"  Profiler 已注入 (has_cuda={profiler._has_cuda})")

    # ==================================================================
    # Step 3: 构建组件
    # ==================================================================
    print("\n[3/7] 构建组件 ...")
    t0 = time.perf_counter()
    engine.build_components()
    build_sec = time.perf_counter() - t0
    print(f"  组件构建完成 ({build_sec:.2f}s)")
    print(f"  adapter:     {type(engine.adapter).__name__}")
    print(f"  tensor_loader: {type(engine.tensor_loader).__name__}")
    print(f"  cache_manager:  {type(engine.cache_manager).__name__ if engine.cache_manager else 'None'}")

    # ==================================================================
    # Step 4: 加载模型
    # ==================================================================
    print("\n[4/7] 加载模型常驻权重 ...")
    engine.load_model()
    print(f"  模型加载完成")

    # 验证组件替换
    print(f"  参数总数: {sum(1 for _ in engine.model_runtime.model.parameters())}")
    print(f"  设备分布: {_count_device_distribution(engine)}")

    # ==================================================================
    # Step 5: Tokenizer + 编码
    # ==================================================================
    print("\n[5/7] 加载 Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    if args.device.startswith("cuda"):
        input_ids = input_ids.to(args.device)
    print(f"  prompt: {args.prompt!r}")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  max_new_tokens: {args.max_new_tokens}")

    # ==================================================================
    # Step 6: 推理
    # ==================================================================
    print("\n[6/7] 执行推理 ...")
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output_ids = engine.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    generated_tokens = int(output_ids.shape[1] - input_ids.shape[1])
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"  生成 token 数: {generated_tokens}")
    print(f"  输出: {output_text[-120:]}")

    # ==================================================================
    # Step 7: 生成报告
    # ==================================================================
    print("\n[7/7] 生成 Profiler 报告 ...")
    report = profiler.report()

    # ---- 终端摘要 ----
    report.print_summary()

    # ---- 导出 JSON ----
    json_path = output_dir / "profile_report.json"
    report.save_json(json_path)

    # ---- 导出内存 CSV ----
    csv_path = output_dir / "memory_timeline.csv"
    report.save_memory_csv(csv_path)

    # ==================================================================
    # 验证
    # ==================================================================
    print("\n" + "=" * 64)
    print("  验证报告有效性")
    print("=" * 64)

    checks = check_report_validity(report)
    all_pass = True
    for name, passed in checks.items():
        status = "✓" if passed else "✗ FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print()

    # ---- 额外数据点 ----
    print(f"  总 forward 次数: {report.total_forward_count}")
    print(f"  内存采样点数: {len(report.memory_timeline)}")
    print(f"  GPU 缓存命中率: {report.gpu_cache_hit_rate:.2%}")
    print(f"  CPU 缓存命中率: {report.cpu_cache_hit_rate:.2%}")
    print(f"  预填充 GPU 耗时: {report.prefill_gpu_time_s:.2f}s (数据搬运: {report.prefill_data_time_s:.2f}s)")
    print(f"  解码 GPU 耗时: {report.decode_gpu_time_s:.2f}s (数据搬运: {report.decode_data_time_s:.2f}s)")

    # ---- 预填充 / 解码缓存统计详情 ----
    print(f"\n  分阶段缓存统计:")
    for label, stats in [
        ("prefill", report.prefill_cache_stats),
        ("decode", report.decode_cache_stats),
    ]:
        print(f"    {label}: "
              f"requests={stats.total_requests}, "
              f"GPU hits={stats.gpu_hits}, "
              f"CPU hits={stats.cpu_hits}, "
              f"disk={stats.disk_fetches}, "
              f"GPU rate={stats.gpu_hit_rate:.2%}")

    if all_pass:
        print("\n" + "=" * 64)
        print("  ✓ 所有检查通过 — Profiler 工作正常")
        print("=" * 64)
        print(f"\n  报告文件:")
        print(f"    JSON: {json_path.resolve()}")
        print(f"    CSV:  {csv_path.resolve()}")
    else:
        print("\n" + "=" * 64)
        print("  ✗ 部分检查未通过，请查看上方详情")
        print("=" * 64)
        sys.exit(1)


def _count_device_distribution(engine) -> dict[str, int]:
    """统计模型参数的设备分布。"""
    dist: dict[str, int] = {}
    for name, param in engine.model_runtime.model.named_parameters():
        dev = str(param.device)
        dist[dev] = dist.get(dev, 0) + 1
    return dist


if __name__ == "__main__":
    main()
