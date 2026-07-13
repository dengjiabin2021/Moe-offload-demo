from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - 环境可选依赖
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None


def triton_gate_up_swiglu(x: torch.Tensor, gate_up_weight: torch.Tensor) -> torch.Tensor:
    """
    用 Triton 计算单 expert 的 gate/up/silu/mul。

    输入：bf16/float16 CUDA hidden states 和连续 gate_up 权重。
    输出：与 expert intermediate 维度一致的 hidden tensor。
    功能：只替代 gate_proj、up_proj 和 SwiGLU 激活，down_proj 仍由 PyTorch 处理。
    """

    if triton is None or tl is None:
        raise ImportError("Triton is not installed.") from _TRITON_IMPORT_ERROR
    if not x.is_cuda or not gate_up_weight.is_cuda:
        raise RuntimeError("triton_gate_up_swiglu requires CUDA tensors.")
    if x.dim() != 2 or gate_up_weight.dim() != 2:
        raise RuntimeError("triton_gate_up_swiglu expects 2D tensors.")
    if gate_up_weight.shape[0] % 2 != 0:
        raise RuntimeError("gate_up_weight first dimension must be even.")

    tokens = x.shape[0]
    hidden_size = x.shape[1]
    intermediate = gate_up_weight.shape[0] // 2
    output = torch.empty((tokens, intermediate), device=x.device, dtype=x.dtype)

    block_m = 16
    block_n = 32
    block_k = 64
    grid = (triton.cdiv(tokens, block_m), triton.cdiv(intermediate, block_n))

    _gate_up_swiglu_kernel[grid](
        x,
        gate_up_weight,
        output,
        tokens,
        hidden_size,
        intermediate,
        x.stride(0),
        x.stride(1),
        gate_up_weight.stride(0),
        gate_up_weight.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return output


if triton is not None and tl is not None:

    @triton.jit
    def _gate_up_swiglu_kernel(
        x_ptr,
        gate_up_ptr,
        out_ptr,
        tokens: tl.constexpr,
        hidden_size: tl.constexpr,
        intermediate: tl.constexpr,
        x_stride_m: tl.constexpr,
        x_stride_k: tl.constexpr,
        w_stride_n: tl.constexpr,
        w_stride_k: tl.constexpr,
        out_stride_m: tl.constexpr,
        out_stride_n: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, hidden_size, BLOCK_K):
            k_idxs = k_start + offs_k
            x_mask = (offs_m[:, None] < tokens) & (k_idxs[None, :] < hidden_size)
            w_mask = (offs_n[:, None] < intermediate) & (k_idxs[None, :] < hidden_size)

            x_vals = tl.load(
                x_ptr + offs_m[:, None] * x_stride_m + k_idxs[None, :] * x_stride_k,
                mask=x_mask,
                other=0.0,
            )
            gate_vals = tl.load(
                gate_up_ptr + offs_n[:, None] * w_stride_n + k_idxs[None, :] * w_stride_k,
                mask=w_mask,
                other=0.0,
            )
            up_vals = tl.load(
                gate_up_ptr
                + (offs_n[:, None] + intermediate) * w_stride_n
                + k_idxs[None, :] * w_stride_k,
                mask=w_mask,
                other=0.0,
            )

            gate_acc += tl.dot(x_vals, tl.trans(gate_vals), out_dtype=tl.float32)
            up_acc += tl.dot(x_vals, tl.trans(up_vals), out_dtype=tl.float32)

        silu_gate = gate_acc / (1.0 + tl.exp(-gate_acc))
        result = silu_gate * up_acc
        out_mask = (offs_m[:, None] < tokens) & (offs_n[None, :] < intermediate)
        tl.store(
            out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n,
            result,
            mask=out_mask,
        )
