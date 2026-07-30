from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


# 将项目根目录加入 sys.path，保证直接运行脚本时可以导入 moe_runtime。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from moe_runtime.blocks import patch_qwen2_prefetch
from moe_runtime.prefetcher import DeepseekV2GatePrefetcher, Qwen2GatePrefetcher


class FakeCacheManager:
    """记录 Prefetcher 提交结果的轻量 CacheManager。

    该测试替身不执行磁盘或 H2D，只保存目标层和 expert 列表，用于验证预测阶段、
    未来层选择和 CacheManager 调用参数。
    """

    def __init__(self) -> None:
        """
        初始化预取调用记录。

        输入：无。
        输出：无。
        功能：创建按提交顺序保存预取参数的 calls 列表。
        """

        self.calls: list[tuple[int, list[int]]] = []

    def prefetch_experts(self, layer_idx: int, expert_ids: list[int]) -> None:
        """
        记录一次预取请求。

        输入：目标层编号和预测 expert 编号列表。
        输出：无。
        功能：保存调用参数，供测试断言 Prefetcher 是否提交了正确目标。
        """

        self.calls.append((layer_idx, list(expert_ids)))


class FakeCache:
    """提供历史 token 长度的 KV Cache 测试替身。"""

    def __init__(self, sequence_length: int) -> None:
        """
        初始化 KV Cache 长度。

        输入：已经缓存的历史 token 数量。
        输出：无。
        功能：保存模型级 hook 判断 prefill/decode 所需的序列长度。
        """

        self.sequence_length = sequence_length

    def get_seq_length(self) -> int:
        """
        返回已经缓存的历史 token 数量。

        输入：无。
        输出：整数形式的 KV Cache 长度。
        功能：模拟 transformers Cache.get_seq_length 接口。
        """

        return self.sequence_length


class FakeQwenGate(nn.Module):
    """返回固定 top-k expert 的 Qwen gate 测试模块。"""

    def __init__(self, expert_ids: list[int]) -> None:
        """
        初始化固定路由结果。

        输入：每个 token 固定选择的 expert 编号。
        输出：无。
        功能：保存 top-k，测试时不依赖真实 gate 权重。
        """

        super().__init__()
        self.expert_ids = expert_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构造与 Qwen gate 相同结构的返回值。

        输入：形状为 batch、sequence、hidden 的 hidden_states。
        输出：占位 logits、占位 routing weights 和固定 selected experts。
        功能：验证 Prefetcher 是否调用未来层原始 gate 并转交其 top-k。
        """

        token_count = hidden_states.shape[0] * hidden_states.shape[1]
        selected = torch.tensor(self.expert_ids, dtype=torch.long).repeat(token_count, 1)
        logits = torch.zeros(token_count, 8)
        weights = torch.ones(token_count, len(self.expert_ids))
        return logits, weights, selected


class FakeMoeBlock(nn.Module):
    """保留 gate 属性并原样返回 hidden_states 的 MoE block 测试替身。"""

    def __init__(self, gate: nn.Module) -> None:
        """
        初始化假 MoE block。

        输入：当前 block 持有的 gate 模块。
        输出：无。
        功能：提供 patch 所需的 mlp.gate 结构，同时允许 forward pre-hook 获取原始输入。
        """

        super().__init__()
        self.gate = gate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        原样返回输入 hidden_states。

        输入：当前层 hidden_states。
        输出：未修改的 hidden_states。
        功能：仅触发注册在 MoE block 上的 forward pre-hook，不执行专家计算。
        """

        return hidden_states


class FakeDecoderLayer(nn.Module):
    """只包含 mlp 字段的 DecoderLayer 测试替身。"""

    def __init__(self, mlp: nn.Module) -> None:
        """
        初始化 DecoderLayer 的 mlp 字段。

        输入：当前层 MLP 或 MoE block。
        输出：无。
        功能：模拟 model.model.layers[layer_idx].mlp 访问路径。
        """

        super().__init__()
        self.mlp = mlp


class FakeBackbone(nn.Module):
    """保存 DecoderLayer 列表的模型骨干测试替身。"""

    def __init__(self, layers: list[FakeDecoderLayer]) -> None:
        """
        初始化模型层列表。

        输入：按 decoder 顺序排列的层。
        输出：无。
        功能：模拟 transformers 模型的 model.layers 结构。
        """

        super().__init__()
        self.layers = nn.ModuleList(layers)


class FakeQwenModel(nn.Module):
    """可触发模型级和 MoE 层级 hook 的 Qwen 测试模型。"""

    def __init__(self, layers: list[FakeDecoderLayer]) -> None:
        """
        初始化假 Qwen 模型。

        输入：按 decoder 顺序排列的层。
        输出：无。
        功能：构造 model.model.layers 路径，并按顺序调用每层 mlp。
        """

        super().__init__()
        self.model = FakeBackbone(layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: FakeCache,
    ) -> torch.Tensor:
        """
        执行假模型 forward。

        输入：token id 和携带历史长度的 KV Cache。
        输出：经过所有假 mlp 的 hidden_states。
        功能：根据输入序列长度构造 hidden_states，并触发预取 patch 注册的 hook。
        """

        hidden_states = torch.ones(input_ids.shape[0], input_ids.shape[1], 4)
        for layer in self.model.layers:
            hidden_states = layer.mlp(hidden_states)
        return hidden_states


class FakeDeepseekMoe(nn.Module):
    """使用真实 linear/top-k 形式模拟 DeepSeek-V2 路由。"""

    def __init__(self, weight: torch.Tensor, top_k: int) -> None:
        """
        初始化 DeepSeek gate 权重和 top-k。

        输入：gate 权重和每个 token 选择的 expert 数量。
        输出：无。
        功能：提供 gate.weight 与 route_tokens_to_experts 两个原实现依赖接口。
        """

        super().__init__()
        self.gate = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.gate.weight.data.copy_(weight)
        self.top_k = top_k

    def route_tokens_to_experts(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对最后一维执行 DeepSeek 风格 top-k。

        输入：形状为 batch、sequence、expert 的 router logits。
        输出：top-k expert 编号和对应分数。
        功能：验证 Prefetcher 复用目标 MoE block 的路由方法，而不是自行推断专家。
        """

        weights, indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        return indices, weights


def test_qwen_patch_prefill_and_decode() -> None:
    """
    验证 Qwen patch 只在 decode 阶段预测后续 routed 层。

    输入：无。
    输出：无。
    功能：保护 prefill 禁止预取、普通 MLP 被跳过，以及距离 1 指向下一个真实 MoE 层三个约束。
    """

    cache_manager = FakeCacheManager()
    layers = [
        FakeDecoderLayer(FakeMoeBlock(FakeQwenGate([0, 1]))),
        FakeDecoderLayer(nn.Identity()),
        FakeDecoderLayer(FakeMoeBlock(FakeQwenGate([3, 4]))),
    ]
    model = FakeQwenModel(layers)
    routed_layers = [0, 2]
    prefetcher = Qwen2GatePrefetcher(
        cache_manager=cache_manager,
        moe_blocks={layer_idx: layers[layer_idx].mlp for layer_idx in routed_layers},
        routed_expert_layers=routed_layers,
        prefetch_distance=1,
    )
    patch_qwen2_prefetch(
        model=model,
        routed_expert_layers=routed_layers,
        prefetcher=prefetcher,
    )

    model(
        input_ids=torch.ones(1, 3, dtype=torch.long),
        past_key_values=FakeCache(0),
    )

    # Prefill 即使经过所有 MoE 层也不能产生预取请求，避免多 token 污染缓存。
    assert cache_manager.calls == []

    model(
        input_ids=torch.ones(1, 1, dtype=torch.long),
        past_key_values=FakeCache(3),
    )

    # 下一个 routed 层是第 2 层，位于中间的普通 MLP 第 1 层必须被跳过。
    assert cache_manager.calls == [(2, [3, 4])]


def test_deepseek_prefetch_uses_target_router() -> None:
    """
    验证 DeepSeek Prefetcher 使用未来 MoE 层自身的路由方法。

    输入：无。
    输出：无。
    功能：保护普通 MLP 层跳过、目标层选择和 top-k 结果三个模型适配约束。
    """

    cache_manager = FakeCacheManager()
    target_moe = FakeDeepseekMoe(
        weight=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        ),
        top_k=2,
    )
    prefetcher = DeepseekV2GatePrefetcher(
        cache_manager=cache_manager,
        moe_blocks={1: target_moe, 3: target_moe},
        routed_expert_layers=[1, 3],
        prefetch_distance=1,
    )
    prefetcher.set_decode_mode(True)
    prefetcher.prefetch(
        current_layer_idx=1,
        hidden_states=torch.tensor([[[1.0, 0.5]]]),
    )

    # 目标层 3 的 gate 分数最高 expert 是 2 和 0，说明结果来自目标层 route 方法。
    assert cache_manager.calls == [(3, [0, 2])]


def main() -> None:
    """
    运行 gate 预取独立测试。

    输入：无。
    输出：无。
    功能：依次验证 Qwen hook 阶段控制和 DeepSeek 目标层路由复用。
    """

    test_qwen_patch_prefill_and_decode()
    test_deepseek_prefetch_uses_target_router()
    print("Gate 预取独立测试通过。")


if __name__ == "__main__":
    main()
