from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.cpp_extension import load


_CUDA_EXTENSION = None


def single_expert_torch(
    *,
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    """
    使用原始 PyTorch 三 linear 路径计算单个 SwiGLU expert。

    输入：当前 expert 负责的 token hidden states、gate/up/down 权重和 router 权重。
    输出：该 expert 对这些 token 的输出贡献。
    功能：作为所有优化 backend 的数值 reference，保持现有 block 内联实现的语义。
    """

    gate = nn.functional.linear(x, gate_weight)
    up = nn.functional.linear(x, up_weight)
    hidden = nn.functional.silu(gate) * up
    output = nn.functional.linear(hidden, down_weight)
    return output * router_weight


def single_expert_torch_gate_up(
    *,
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    """
    使用 PyTorch 合并 gate/up 投影计算单个 SwiGLU expert。

    输入：当前 token hidden states、连续的 gate_up 权重、down 权重和 router 权重。
    输出：该 expert 对这些 token 的输出贡献。
    功能：把 gate_proj 和 up_proj 合成一次 linear，减少一次 kernel launch 和一次输入读取。
    """

    gate_up = nn.functional.linear(x, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = nn.functional.silu(gate) * up
    output = nn.functional.linear(hidden, down_weight)
    return output * router_weight


def single_expert_triton_gate_up(
    *,
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    """
    使用 Triton 计算 gate/up/silu/mul，再用 PyTorch 计算 down 投影。

    输入：当前 token hidden states、连续的 gate_up 权重、down 权重和 router 权重。
    输出：该 expert 对这些 token 的输出贡献。
    功能：Triton 只优化单 expert 的第一段投影和激活，不做多 expert 合并。
    """

    from .triton_single_expert import triton_gate_up_swiglu

    hidden = triton_gate_up_swiglu(x, gate_up_weight)
    output = nn.functional.linear(hidden, down_weight)
    return output * router_weight


def single_expert_cuda_gate_up(
    *,
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    """
    使用 CUDA bf16 kernel 计算 gate/up/silu/mul，再用 PyTorch 计算 down 投影。

    输入：bf16 CUDA hidden states、连续的 bf16 gate_up 权重、down 权重和 router 权重。
    输出：该 expert 对这些 token 的输出贡献。
    功能：CUDA kernel 针对模型 bf16 权重实现单 expert gate/up 合并计算。
    """

    extension = _load_cuda_extension()
    hidden = extension.gate_up_swiglu(x.contiguous(), gate_up_weight.contiguous())
    output = nn.functional.linear(hidden, down_weight)
    return output * router_weight


def _load_cuda_extension():
    """
    懒加载 CUDA extension。

    输入：无。
    输出：已编译并加载的 PyTorch extension 模块。
    功能：只有用户选择 cuda_gate_up backend 时才触发编译，避免影响默认路径启动。
    """

    global _CUDA_EXTENSION
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION

    source_dir = Path(__file__).resolve().parent
    _CUDA_EXTENSION = load(
        name="moe_single_expert_cuda",
        sources=[
            str(source_dir / "cuda_single_expert.cpp"),
            str(source_dir / "cuda_single_expert_kernel.cu"),
        ],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["/O2"] if _is_windows() else ["-O3"],
        verbose=False,
    )
    return _CUDA_EXTENSION


def _is_windows() -> bool:
    """
    判断当前平台是否为 Windows。

    输入：无。
    输出：Windows 返回 True，否则返回 False。
    功能：torch extension 在 MSVC 和 GCC/Clang 下使用不同的优化参数。
    """

    import os

    return os.name == "nt"
