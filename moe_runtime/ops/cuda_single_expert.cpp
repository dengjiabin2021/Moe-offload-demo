#include <torch/extension.h>

torch::Tensor gate_up_swiglu_cuda(torch::Tensor x, torch::Tensor gate_up_weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gate_up_swiglu", &gate_up_swiglu_cuda, "bf16 gate/up swiglu single expert");
}
