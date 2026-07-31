from __future__ import annotations

"""Gate 专家预取指标采集脚本。

本脚本只在测试侧对运行时实例做 monkeypatch，不修改 ``moe_runtime`` 或
``moe_profiler`` 的实现。它统计目标层累计预测、来源层预测、CPU 预取任务完成情况，
以及预测提交时专家已经位于 CPU/GPU 缓存中的概率。
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import functools
import gc
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any
import unicodedata

import torch
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moe_runtime import MoeRuntimeEngine, RuntimeConfig


@dataclass
class LayerMetrics:
    """保存一个目标 MoE 层的全部预取指标计数。

    prediction 字段描述按 decode step 去重后的累计预测；cache 字段描述每次来源层提交时
    的缓存位置；task 字段描述 CacheManager 真正创建的 CPU 预取任务。
    """

    prediction_events: int = 0
    predicted_experts: int = 0
    correct_experts: int = 0
    cache_checks: int = 0
    gpu_cache_hits: int = 0
    cpu_cache_hits: int = 0
    any_cache_hits: int = 0
    submitted_tasks: int = 0
    completed_tasks: int = 0


@dataclass
class SourceAccuracy:
    """保存一个来源层到目标层的预测正确率计数。

    events 是该来源实际预测目标层的 decode 次数，predicted_experts 和
    correct_experts 用于计算该来源层独立的专家级正确率。
    """

    events: int = 0
    predicted_experts: int = 0
    correct_experts: int = 0


class PrefetchMetricsTracker:
    """线程安全地汇总 Gate 预取指标。

    _open_predictions 保存当前 forward 尚未到达目标层的来源预测；目标层 demand 到达时
    统一结算。调度线程可以同时上报 CPU 预取完成事件，因此所有计数由独立锁保护。
    """

    def __init__(self) -> None:
        """
        初始化预取指标状态。

        输入：无。
        输出：无。
        功能：创建逐层计数、来源层计数、待结算预测和线程本地来源层上下文。
        """

        self._lock = threading.RLock()
        self._source_context = threading.local()
        self._layer_metrics: dict[int, LayerMetrics] = {}
        self._source_accuracy: dict[tuple[int, int], SourceAccuracy] = {}
        self._open_predictions: dict[int, dict[int, set[int]]] = {}

    def set_current_source_layer(self, layer_idx: int | None) -> int | None:
        """
        替换当前线程正在执行的预取来源层。

        输入：新的来源层编号；None 表示离开 Gate 预取上下文。
        输出：替换前的来源层编号。
        功能：让 CacheManager 的 prefetch_experts wrapper 能识别本次预测来自哪一层。
        """

        previous = getattr(self._source_context, "layer_idx", None)
        self._source_context.layer_idx = layer_idx
        return previous

    def current_source_layer(self) -> int | None:
        """
        读取当前线程的预取来源层。

        输入：无。
        输出：来源层编号；不在 Gate 预取调用中时返回 None。
        功能：把预取器实例上下文传递给 CacheManager 指标 wrapper。
        """

        return getattr(self._source_context, "layer_idx", None)

    def record_prefetch_call(
        self,
        *,
        source_layer_idx: int,
        target_layer_idx: int,
        expert_ids: set[int],
        gpu_hit_ids: set[int],
        cpu_hit_ids: set[int],
        submitted_task_count: int,
    ) -> None:
        """
        记录一次来源层对目标层的预测提交。

        输入：来源层、目标层、预测专家集合、CPU/GPU 缓存命中集合和新建任务数。
        输出：无。
        功能：累积本次 forward 的预测并立即记录预测提交时的缓存位置和任务创建数。
        """

        with self._lock:
            target_predictions = self._open_predictions.setdefault(
                target_layer_idx,
                {},
            )
            source_predictions = target_predictions.setdefault(source_layer_idx, set())
            # 同一来源在一次 target demand 前重复给出相同 expert 时只保留一次预测。
            source_predictions.update(expert_ids)

            metrics = self._layer_metrics.setdefault(target_layer_idx, LayerMetrics())
            metrics.cache_checks += len(expert_ids)
            metrics.gpu_cache_hits += len(gpu_hit_ids)
            metrics.cpu_cache_hits += len(cpu_hit_ids)
            metrics.any_cache_hits += len(gpu_hit_ids | cpu_hit_ids)
            metrics.submitted_tasks += submitted_task_count

    def settle_demand(
        self,
        target_layer_idx: int,
        actual_expert_ids: set[int],
    ) -> None:
        """
        用目标层真实 demand 结算尚未关闭的预测。

        输入：目标层编号和真实 gate 命中的专家集合。
        输出：无。
        功能：目标层指标按所有来源预测并集结算，来源指标分别结算；预取阶段不影响正确性判定。
        """

        with self._lock:
            predictions_by_source = self._open_predictions.pop(target_layer_idx, None)
            if not predictions_by_source:
                return

            accumulated_predictions: set[int] = set()
            for source_layer_idx, predicted_experts in predictions_by_source.items():
                # 来源层正确率分别结算，同一 expert 被不同来源预测时各自获得一次样本。
                accumulated_predictions.update(predicted_experts)
                source_metrics = self._source_accuracy.setdefault(
                    (source_layer_idx, target_layer_idx),
                    SourceAccuracy(),
                )
                source_metrics.events += 1
                source_metrics.predicted_experts += len(predicted_experts)
                source_metrics.correct_experts += len(
                    predicted_experts & actual_expert_ids
                )

            layer_metrics = self._layer_metrics.setdefault(
                target_layer_idx,
                LayerMetrics(),
            )
            # 目标层总体正确率按累计预测并集结算，匹配运行时的累积预取语义。
            layer_metrics.prediction_events += 1
            layer_metrics.predicted_experts += len(accumulated_predictions)
            layer_metrics.correct_experts += len(
                accumulated_predictions & actual_expert_ids
            )

    def record_completed_tasks(self, completed_by_layer: Counter[int]) -> None:
        """
        记录正式进入 CPU 缓存的预取任务。

        输入：目标层到本次完成任务数的映射。
        输出：无。
        功能：只统计转入 prefetch_ready_cpu 的任务；错误但已完成的预测同样计入。
        """

        with self._lock:
            for layer_idx, count in completed_by_layer.items():
                metrics = self._layer_metrics.setdefault(layer_idx, LayerMetrics())
                metrics.completed_tasks += count

    def build_report(self, routed_expert_layers: list[int]) -> dict[str, Any]:
        """
        构造可打印和可序列化的最终指标报告。

        输入：按模型顺序排列的真实 routed expert 层。
        输出：包含全局、逐目标层、逐来源层和未结算预测的报告字典。
        功能：计算所有比例，并给来源层指标补充 decoder 层间隔和 routed expert 层间隔。
        """

        with self._lock:
            layer_items = sorted(self._layer_metrics.items())
            source_items = sorted(
                self._source_accuracy.items(),
                key=lambda item: (item[0][1], item[0][0]),
            )
            open_predictions = {
                target_layer_idx: {
                    source_layer_idx: set(expert_ids)
                    for source_layer_idx, expert_ids in source_predictions.items()
                }
                for target_layer_idx, source_predictions in self._open_predictions.items()
            }

        layer_rows = [
            _layer_metrics_row(layer_idx, metrics)
            for layer_idx, metrics in layer_items
        ]
        total_metrics = LayerMetrics(
            prediction_events=sum(item.prediction_events for _, item in layer_items),
            predicted_experts=sum(item.predicted_experts for _, item in layer_items),
            correct_experts=sum(item.correct_experts for _, item in layer_items),
            cache_checks=sum(item.cache_checks for _, item in layer_items),
            gpu_cache_hits=sum(item.gpu_cache_hits for _, item in layer_items),
            cpu_cache_hits=sum(item.cpu_cache_hits for _, item in layer_items),
            any_cache_hits=sum(item.any_cache_hits for _, item in layer_items),
            submitted_tasks=sum(item.submitted_tasks for _, item in layer_items),
            completed_tasks=sum(item.completed_tasks for _, item in layer_items),
        )

        routed_positions = {
            layer_idx: position
            for position, layer_idx in enumerate(routed_expert_layers)
        }
        source_rows: list[dict[str, Any]] = []
        for (source_layer_idx, target_layer_idx), metrics in source_items:
            source_position = routed_positions.get(source_layer_idx)
            target_position = routed_positions.get(target_layer_idx)
            routed_distance = None
            if source_position is not None and target_position is not None:
                routed_distance = target_position - source_position

            source_rows.append(
                {
                    "source_layer": source_layer_idx,
                    "target_layer": target_layer_idx,
                    "decoder_layer_gap": target_layer_idx - source_layer_idx,
                    "routed_layer_distance": routed_distance,
                    "events": metrics.events,
                    "predicted_experts": metrics.predicted_experts,
                    "correct_experts": metrics.correct_experts,
                    "accuracy": _safe_rate(
                        metrics.correct_experts,
                        metrics.predicted_experts,
                    ),
                }
            )

        unsettled_rows: list[dict[str, int]] = []
        for target_layer_idx, source_predictions in sorted(open_predictions.items()):
            for source_layer_idx, expert_ids in sorted(source_predictions.items()):
                unsettled_rows.append(
                    {
                        "source_layer": source_layer_idx,
                        "target_layer": target_layer_idx,
                        "predicted_experts": len(expert_ids),
                    }
                )

        return {
            "definitions": {
                "accuracy": "同一目标层、同一次 demand 前所有来源预测并集与真实专家集合的交集比例；未完成但被 demand 提升的预取仍算正确。",
                "completion_rate": "进入 prefetch_ready_cpu 的任务数除以 CacheManager 真正新建的 CPU 预取任务数；错误但已完成的预取计入完成数。",
                "gpu_cache_hit_rate": "来源层提交预测时，预测专家已经位于可读 GPU slot 的比例。",
                "cpu_cache_hit_rate": "来源层提交预测时，预测专家已经位于可读 CPU slot 的比例。",
                "source_accuracy": "每个来源层对指定目标层的预测集合与真实专家集合的交集比例。",
            },
            "global": _layer_metrics_row("all", total_metrics),
            "by_target_layer": layer_rows,
            "accuracy_by_source_layer": source_rows,
            "unsettled_predictions": unsettled_rows,
        }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """
    安全计算比例。

    输入：分子和分母计数。
    输出：分母大于零时返回零到一之间的比例，否则返回 None。
    功能：避免没有预测或没有新建任务的层产生误导性的零比例。
    """

    if denominator <= 0:
        return None
    return numerator / denominator


def _layer_metrics_row(
    layer_idx: int | str,
    metrics: LayerMetrics,
) -> dict[str, Any]:
    """
    把目标层原始计数转换成报告行。

    输入：目标层编号或全局标识，以及该层累计计数。
    输出：包含原始计数和派生比例的字典。
    功能：统一逐层与全局指标字段，保证 JSON 和 CSV 使用相同口径。
    """

    return {
        "target_layer": layer_idx,
        "prediction_events": metrics.prediction_events,
        "predicted_experts": metrics.predicted_experts,
        "correct_experts": metrics.correct_experts,
        "accuracy": _safe_rate(
            metrics.correct_experts,
            metrics.predicted_experts,
        ),
        "submitted_tasks": metrics.submitted_tasks,
        "completed_tasks": metrics.completed_tasks,
        "completion_rate": _safe_rate(
            metrics.completed_tasks,
            metrics.submitted_tasks,
        ),
        "cache_checks": metrics.cache_checks,
        "gpu_cache_hits": metrics.gpu_cache_hits,
        "gpu_cache_hit_rate": _safe_rate(
            metrics.gpu_cache_hits,
            metrics.cache_checks,
        ),
        "cpu_cache_hits": metrics.cpu_cache_hits,
        "cpu_cache_hit_rate": _safe_rate(
            metrics.cpu_cache_hits,
            metrics.cache_checks,
        ),
        "any_cache_hits": metrics.any_cache_hits,
        "any_cache_hit_rate": _safe_rate(
            metrics.any_cache_hits,
            metrics.cache_checks,
        ),
    }


def install_metrics_patch(
    engine: MoeRuntimeEngine,
    tracker: PrefetchMetricsTracker,
) -> int:
    """
    在已加载模型和 CacheManager 实例上安装指标 monkeypatch。

    输入：已完成 load_model 的 RuntimeEngine 和指标 Tracker。
    输出：成功包装的 GatePrefetcher 实例数量。
    功能：包装预测提交、真实 demand、CPU 预取完成和每个来源层预取器，不改变原方法决策。
    """

    cache_manager = engine.cache_manager
    if cache_manager is None:
        raise RuntimeError("CacheManager 尚未构建，无法安装预取指标 patch。")
    if engine.model_runtime is None or engine.model_runtime.model is None:
        raise RuntimeError("模型尚未加载，无法定位逐层 GatePrefetcher。")

    original_prefetch_experts = cache_manager.prefetch_experts
    original_request_experts = cache_manager.request_experts
    original_complete_prefetch = (
        cache_manager._complete_finished_prefetch_disk_reads_locked
    )

    @functools.wraps(original_prefetch_experts)
    def prefetch_experts_wrapper(
        layer_idx: int,
        expert_ids: list[int],
    ) -> Any:
        """
        记录预测提交时的来源、缓存位置和新建任务数。

        输入：目标层编号和预测 expert 编号列表。
        输出：原 prefetch_experts 的返回值。
        功能：在同一 CacheManager 锁内读取前后状态，保证指标观察不会引入竞态或改变调度。
        """

        source_layer_idx = tracker.current_source_layer()
        if source_layer_idx is None:
            raise RuntimeError("无法识别本次预取的来源层，GatePrefetcher patch 可能未安装。")

        normalized_ids = {int(expert_idx) for expert_idx in expert_ids}
        keys = {(int(layer_idx), expert_idx) for expert_idx in normalized_ids}

        # 外层持锁覆盖状态快照和原方法调用，防止调度线程在两次观察之间推进请求。
        with cache_manager._cv:
            gpu_hit_ids = {
                expert_idx
                for expert_idx in normalized_ids
                if cache_manager._find_expert_in_slots(
                    cache_manager._gpu_slots,
                    cache_manager._gpu_layer_slot_map,
                    int(layer_idx),
                    expert_idx,
                )
                is not None
            }
            cpu_hit_ids = {
                expert_idx
                for expert_idx in normalized_ids
                if cache_manager._find_expert_in_slots(
                    cache_manager._cpu_slots,
                    cache_manager._cpu_layer_slot_map,
                    int(layer_idx),
                    expert_idx,
                )
                is not None
            }
            existing_keys = {
                key
                for key in keys
                if key in cache_manager._requests
            }

            result = original_prefetch_experts(
                layer_idx=layer_idx,
                expert_ids=expert_ids,
            )

            submitted_task_count = 0
            for key in keys - existing_keys:
                request = cache_manager._requests.get(key)
                # 只有原方法真正创建的 prefetch request 才进入完成率分母。
                if request is not None and request.is_prefetch:
                    submitted_task_count += 1

        tracker.record_prefetch_call(
            source_layer_idx=source_layer_idx,
            target_layer_idx=int(layer_idx),
            expert_ids=normalized_ids,
            gpu_hit_ids=gpu_hit_ids,
            cpu_hit_ids=cpu_hit_ids,
            submitted_task_count=submitted_task_count,
        )
        return result

    @functools.wraps(original_request_experts)
    def request_experts_wrapper(
        layer_idx: int,
        expert_ids: list[int],
    ) -> Any:
        """
        用当前层真实 demand 结算预测正确率。

        输入：目标层编号和真实 expert 编号列表。
        输出：原 request_experts 的返回值。
        功能：先让原方法完成预取提升与错误预测清理，再按真实集合结算所有来源预测。
        """

        result = original_request_experts(
            layer_idx=layer_idx,
            expert_ids=expert_ids,
        )
        tracker.settle_demand(
            int(layer_idx),
            {int(expert_idx) for expert_idx in expert_ids},
        )
        return result

    @functools.wraps(original_complete_prefetch)
    def complete_prefetch_wrapper() -> bool:
        """
        记录真正进入 CPU 预取缓存的任务。

        输入：无；调用方保持原私有调度方法的锁约束。
        输出：原完成检查方法的 progressed 结果。
        功能：只把 prefetch_loading_cpu 到 prefetch_ready_cpu 的状态转换计为预取完成。
        """

        with cache_manager._cv:
            loading_before = set(cache_manager._prefetch_loading_cpu)
            result = original_complete_prefetch()
            completed_by_layer: Counter[int] = Counter()
            for key in loading_before:
                request = cache_manager._requests.get(key)
                # pending 命中被提升后走 demand 完成路径，不会被误计为预取完成。
                if (
                    request is not None
                    and request.is_prefetch
                    and request.stage == "prefetch_ready_cpu"
                ):
                    completed_by_layer[int(key[0])] += 1

        if completed_by_layer:
            tracker.record_completed_tasks(completed_by_layer)
        return result

    cache_manager.prefetch_experts = prefetch_experts_wrapper
    cache_manager.request_experts = request_experts_wrapper
    cache_manager._complete_finished_prefetch_disk_reads_locked = (
        complete_prefetch_wrapper
    )

    patched_prefetchers: set[int] = set()
    for module in engine.model_runtime.model.modules():
        prefetcher = getattr(module, "prefetcher", None)
        if prefetcher is None or id(prefetcher) in patched_prefetchers:
            continue

        source_layer_idx = int(prefetcher.layer_idx)
        original_prefetch = prefetcher.prefetch

        @functools.wraps(original_prefetch)
        def gate_prefetch_wrapper(
            hidden_states: torch.Tensor,
            _original_prefetch: Any = original_prefetch,
            _source_layer_idx: int = source_layer_idx,
        ) -> Any:
            """
            为一个逐层 GatePrefetcher 注入来源层上下文。

            输入：当前专家模块收到的 hidden_states，以及闭包保存的原方法和来源层。
            输出：原 GatePrefetcher.prefetch 的返回值。
            功能：调用期间设置线程本地来源层，并在结束或异常时恢复先前上下文。
            """

            # 来源层只在原预取方法调用范围内有效，finally 防止异常污染后续层。
            previous_source = tracker.set_current_source_layer(_source_layer_idx)
            try:
                return _original_prefetch(hidden_states)
            finally:
                tracker.set_current_source_layer(previous_source)

        prefetcher.prefetch = gate_prefetch_wrapper
        patched_prefetchers.add(id(prefetcher))

    if not patched_prefetchers:
        raise RuntimeError(
            "没有找到 GatePrefetcher，请确认 prefetch_policy='gate' 且模型包含 routed expert 层。"
        )
    return len(patched_prefetchers)


def parse_args() -> argparse.Namespace:
    """
    解析预取指标脚本参数。

    输入：命令行参数。
    输出：解析后的 argparse Namespace。
    功能：提供模型、设备、生成长度、缓存容量、预取距离、空闲等待和输出目录配置。
    """

    parser = argparse.ArgumentParser(description="采集 Gate 专家预取准确率与完成率。")
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "Qwen1.5-Moe-A2.7B"),
        help="本地模型目录。",
    )
    parser.add_argument("--device", default="cuda", help="运行设备。")
    parser.add_argument(
        "--prompt",
        default="问：你好 答：你好！有什么我可以帮你的吗? 问: 1+1=? 答：",
        help="生成 prompt。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="最大生成 token 数。",
    )
    parser.add_argument(
        "--gpu-cache-per-layer",
        type=int,
        default=6,
        help="每个 routed expert 层的 GPU cache 数量。",
    )
    parser.add_argument(
        "--cpu-cache-per-layer",
        type=int,
        default=10,
        help="每个 routed expert 层的 CPU cache 数量。",
    )
    parser.add_argument(
        "--cache-policy",
        choices=("lru", "lfu"),
        default="lru",
        help="缓存淘汰策略。",
    )
    parser.add_argument(
        "--prefetch-distance",
        type=int,
        default=3,
        help="向后预测的 routed expert 层数量。",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=60.0,
        help="生成结束后等待异步状态机清空的秒数。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "prefetch_metrics"),
        help="JSON 和 CSV 指标输出目录。",
    )
    return parser.parse_args()


def build_cache_config(
    model_dir: str,
    gpu_per_layer: int,
    cpu_per_layer: int,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    为模型所有 decoder 层生成缓存容量配置。

    输入：模型目录和每层 GPU、CPU expert 容量。
    输出：GPU 与 CPU 的逐层容量字典。
    功能：先覆盖所有 decoder 层，真实非 routed 层由 CacheManager 初始化时过滤。
    """

    model_config = AutoConfig.from_pretrained(model_dir)
    all_layers = range(model_config.num_hidden_layers)
    gpu_cache = {layer_idx: gpu_per_layer for layer_idx in all_layers}
    cpu_cache = {layer_idx: cpu_per_layer for layer_idx in all_layers}
    return gpu_cache, cpu_cache


def wait_cache_idle(cache_manager: Any, timeout_s: float) -> bool:
    """
    等待异步 CacheManager 完成所有请求。

    输入：异步 CacheManager 和超时时间。
    输出：超时前状态机清空时返回 True，否则返回 False。
    功能：确保调度线程已上报最后的 CPU 预取完成事件，再生成指标报告。
    """

    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with cache_manager._cv:
            if (
                not cache_manager._requests
                and not cache_manager._loading_cpu
                and not cache_manager._loading_gpu
                and not cache_manager._prefetch_loading_cpu
                and not cache_manager._compute_pending
            ):
                return True
        time.sleep(0.01)
    return False


def cleanup_engine(engine: MoeRuntimeEngine | None) -> None:
    """
    释放一次指标运行产生的模型和缓存资源。

    输入：可能尚未完整初始化的 RuntimeEngine。
    输出：无。
    功能：同步 CUDA、关闭 CacheManager、断开大对象引用并回收 allocator 缓存。
    """

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


def _display_rate(value: float | None) -> str:
    """
    把比例格式化为终端百分比。

    输入：零到一的比例或 None。
    输出：保留两位小数的百分比；没有分母时返回 N/A。
    功能：让逐层表格明确区分零命中和没有样本。
    """

    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _right_align_terminal(value: Any, width: int) -> str:
    """
    按终端显示宽度右对齐文本。

    输入：待显示的值和目标列宽。
    输出：补齐左侧空格后的字符串。
    功能：将全角和宽字符按两个终端列计算，避免中文表头与 ASCII 数值行错位。
    """

    text = str(value)
    display_width = sum(
        2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
        for char in text
    )
    return " " * max(width - display_width, 0) + text


def print_report(report: dict[str, Any]) -> None:
    """
    在终端打印全局、逐目标层和逐来源层指标。

    输入：Tracker 构造的指标报告。
    输出：无。
    功能：用紧凑表格展示用户要求的五类指标，并提示未结算预测。
    """

    global_row = report["global"]
    print("\n" + "=" * 118)
    print("Gate 专家预取指标")
    print("=" * 118)
    print(
        "全局: "
        f"正确率={_display_rate(global_row['accuracy'])} "
        f"({global_row['correct_experts']}/{global_row['predicted_experts']}), "
        f"完成率={_display_rate(global_row['completion_rate'])} "
        f"({global_row['completed_tasks']}/{global_row['submitted_tasks']}), "
        f"GPU 已缓存率={_display_rate(global_row['gpu_cache_hit_rate'])}, "
        f"CPU 已缓存率={_display_rate(global_row['cpu_cache_hit_rate'])}"
    )

    print("\n按目标层统计：")
    print(" ".join([
        _right_align_terminal("层", 5),
        _right_align_terminal("事件", 7),
        _right_align_terminal("预测", 8),
        _right_align_terminal("正确", 8),
        _right_align_terminal("正确率", 10),
        _right_align_terminal("任务", 8),
        _right_align_terminal("完成", 8),
        _right_align_terminal("完成率", 10),
        _right_align_terminal("GPU命中/检查", 15),
        _right_align_terminal("GPU率", 10),
        _right_align_terminal("CPU命中/检查", 15),
        _right_align_terminal("CPU率", 10),
    ]))
    for row in report["by_target_layer"]:
        # 先构造命中计数字符串，避免表格格式化表达式掩盖实际分母。
        gpu_hit_text = f"{row['gpu_cache_hits']}/{row['cache_checks']}"
        cpu_hit_text = f"{row['cpu_cache_hits']}/{row['cache_checks']}"
        print(" ".join([
            _right_align_terminal(row["target_layer"], 5),
            _right_align_terminal(row["prediction_events"], 7),
            _right_align_terminal(row["predicted_experts"], 8),
            _right_align_terminal(row["correct_experts"], 8),
            _right_align_terminal(_display_rate(row["accuracy"]), 10),
            _right_align_terminal(row["submitted_tasks"], 8),
            _right_align_terminal(row["completed_tasks"], 8),
            _right_align_terminal(_display_rate(row["completion_rate"]), 10),
            _right_align_terminal(gpu_hit_text, 15),
            _right_align_terminal(_display_rate(row["gpu_cache_hit_rate"]), 10),
            _right_align_terminal(cpu_hit_text, 15),
            _right_align_terminal(_display_rate(row["cpu_cache_hit_rate"]), 10),
        ]))

    print("\n按来源层 → 目标层统计正确率：")
    print(" ".join([
        _right_align_terminal("来源层", 7),
        _right_align_terminal("目标层", 7),
        _right_align_terminal("MoE距离", 9),
        _right_align_terminal("事件", 7),
        _right_align_terminal("预测", 8),
        _right_align_terminal("正确", 8),
        _right_align_terminal("正确率", 10),
    ]))
    for row in report["accuracy_by_source_layer"]:
        distance = row["routed_layer_distance"]
        distance_text = "N/A" if distance is None else str(distance)
        print(" ".join([
            _right_align_terminal(row["source_layer"], 7),
            _right_align_terminal(row["target_layer"], 7),
            _right_align_terminal(distance_text, 9),
            _right_align_terminal(row["events"], 7),
            _right_align_terminal(row["predicted_experts"], 8),
            _right_align_terminal(row["correct_experts"], 8),
            _right_align_terminal(_display_rate(row["accuracy"]), 10),
        ]))

    if report["unsettled_predictions"]:
        print("\n警告：仍有未到达目标层的预测，详见 JSON 的 unsettled_predictions。")


def save_report(
    report: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """
    保存 JSON 和两张 CSV 指标表。

    输入：完整指标报告和输出目录。
    输出：JSON、逐目标层 CSV、逐来源层 CSV 的路径。
    功能：保留机器可读定义、全局值和所有逐层明细，便于后续画图或跨实验比较。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "prefetch_metrics.json"
    layer_csv_path = output_dir / "prefetch_metrics_by_layer.csv"
    source_csv_path = output_dir / "prefetch_accuracy_by_source.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    layer_rows = report["by_target_layer"]
    if layer_rows:
        with layer_csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(layer_rows[0]))
            writer.writeheader()
            writer.writerows(layer_rows)

    source_rows = report["accuracy_by_source_layer"]
    if source_rows:
        with source_csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(source_rows[0]))
            writer.writeheader()
            writer.writerows(source_rows)

    return json_path, layer_csv_path, source_csv_path


def main() -> None:
    """
    加载模型、执行确定性生成并输出 Gate 预取指标。

    输入：命令行参数。
    输出：无。
    功能：仅通过实例 monkeypatch 观测预取，不改变模型输出、缓存决策或运行时源码。
    """

    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    gpu_cache, cpu_cache = build_cache_config(
        str(model_dir),
        args.gpu_cache_per_layer,
        args.cpu_cache_per_layer,
    )
    config = RuntimeConfig(
        model_dir=model_dir,
        cache_manager_mode="async",
        prefetch_policy="gate",
        prefetch_distance=args.prefetch_distance,
        non_expert_device=args.device,
        shared_expert_device=args.device,
        routed_expert_device=args.device,
        cache_policy=args.cache_policy,
        max_gpu_cache_capacity=gpu_cache,
        max_cpu_cache_capacity=cpu_cache,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    engine: MoeRuntimeEngine | None = None
    try:
        print("[1/4] 构建运行时组件 ...")
        engine = MoeRuntimeEngine(config)
        engine.build_components()

        print("[2/4] 加载模型并安装测试侧指标 patch ...")
        engine.load_model()
        tracker = PrefetchMetricsTracker()
        patched_count = install_metrics_patch(engine, tracker)
        print(f"  已包装 GatePrefetcher: {patched_count}")

        print("[3/4] 执行确定性 generate ...")
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

        cache_idle = wait_cache_idle(engine.cache_manager, args.idle_timeout)
        if not cache_idle:
            print("  警告：等待 CacheManager 空闲超时，完成率可能遗漏仍在运行的任务。")

        print("[4/4] 汇总并保存指标 ...")
        routed_expert_layers = list(
            engine.adapter.routed_expert_layers(engine.model_config)
        )
        report = tracker.build_report(routed_expert_layers)
        new_tokens = int(output_ids.shape[-1] - inputs["input_ids"].shape[-1])
        report["run"] = {
            "model_dir": str(model_dir),
            "device": args.device,
            "prefetch_distance": args.prefetch_distance,
            "max_new_tokens": args.max_new_tokens,
            "generated_tokens": new_tokens,
            "elapsed_seconds": elapsed_s,
            "tokens_per_second": new_tokens / elapsed_s if elapsed_s > 0 else None,
            "gpu_cache_per_layer": args.gpu_cache_per_layer,
            "cpu_cache_per_layer": args.cpu_cache_per_layer,
            "cache_policy": args.cache_policy,
            "cache_idle_before_report": cache_idle,
        }

        print_report(report)
        json_path, layer_csv_path, source_csv_path = save_report(
            report,
            output_dir,
        )
        print(f"\nJSON: {json_path}")
        print(f"逐目标层 CSV: {layer_csv_path}")
        print(f"逐来源层 CSV: {source_csv_path}")
    finally:
        cleanup_engine(engine)


if __name__ == "__main__":
    main()
