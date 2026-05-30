#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

__global__ void deadly_corridor_noop_kernel(unsigned char* out, int64_t n) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = out[idx];
    }
}

}  // namespace

void deadly_corridor_noop_cuda(torch::Tensor out) {
    const int64_t n = out.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    deadly_corridor_noop_kernel<<<blocks, threads>>>(out.data_ptr<unsigned char>(), n);
}
