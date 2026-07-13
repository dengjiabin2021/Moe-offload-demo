#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void gate_up_swiglu_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate_up_weight,
    __nv_bfloat16* __restrict__ out,
    int tokens,
    int hidden_size,
    int intermediate) {
    const int token_idx = blockIdx.y;
    const int intermediate_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= tokens || intermediate_idx >= intermediate) {
        return;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    const __nv_bfloat16* x_row = x + token_idx * hidden_size;
    const __nv_bfloat16* gate_row = gate_up_weight + intermediate_idx * hidden_size;
    const __nv_bfloat16* up_row = gate_up_weight + (intermediate + intermediate_idx) * hidden_size;

    for (int k = 0; k < hidden_size; ++k) {
        const float x_val = __bfloat162float(x_row[k]);
        gate_acc += x_val * __bfloat162float(gate_row[k]);
        up_acc += x_val * __bfloat162float(up_row[k]);
    }

    const float silu_gate = gate_acc / (1.0f + expf(-gate_acc));
    out[token_idx * intermediate + intermediate_idx] = __float2bfloat16(silu_gate * up_acc);
}

}  // namespace

torch::Tensor gate_up_swiglu_cuda(torch::Tensor x, torch::Tensor gate_up_weight) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gate_up_weight.is_cuda(), "gate_up_weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be bf16");
    TORCH_CHECK(gate_up_weight.scalar_type() == torch::kBFloat16, "gate_up_weight must be bf16");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(gate_up_weight.dim() == 2, "gate_up_weight must be 2D");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gate_up_weight.is_contiguous(), "gate_up_weight must be contiguous");
    TORCH_CHECK(gate_up_weight.size(0) % 2 == 0, "gate_up_weight first dimension must be even");
    TORCH_CHECK(x.size(1) == gate_up_weight.size(1), "hidden size mismatch");

    const int tokens = static_cast<int>(x.size(0));
    const int hidden_size = static_cast<int>(x.size(1));
    const int intermediate = static_cast<int>(gate_up_weight.size(0) / 2);
    auto out = torch::empty({tokens, intermediate}, x.options());

    const int threads = 128;
    const dim3 blocks((intermediate + threads - 1) / threads, tokens);
    auto stream = at::cuda::getCurrentCUDAStream();

    gate_up_swiglu_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate_up_weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        tokens,
        hidden_size,
        intermediate);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
