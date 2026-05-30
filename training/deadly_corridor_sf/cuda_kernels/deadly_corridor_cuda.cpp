#include <torch/extension.h>

void deadly_corridor_noop_cuda(torch::Tensor out);

void deadly_corridor_noop(torch::Tensor out) {
    deadly_corridor_noop_cuda(out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deadly_corridor_noop", &deadly_corridor_noop, "Deadly Corridor CUDA extension smoke kernel");
}
