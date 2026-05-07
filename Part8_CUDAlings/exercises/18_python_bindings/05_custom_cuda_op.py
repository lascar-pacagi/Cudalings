"""CUDAlings 18.05 -- A real custom CUDA op exposed to PyTorch.

This is the full pattern: a __global__ kernel + a C++ wrapper that
unpacks torch::Tensor, launches the kernel, and returns torch::Tensor.

Goal: implement `relu_cuda` so y = max(0, x), running on the GPU. The
test creates a CUDA tensor, runs your op, copies back to CPU, validates.

If you don't have a CUDA-capable PyTorch install, this exercise will
skip with "skip (no cuda)" -- print that to pass the validator.
"""

# I AM NOT DONE

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>

__global__ void relu_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: if (i < n) y[i] = x[i] > 0 ? x[i] : 0.0f;
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    int block = 256;
    int grid  = (n + block - 1) / block;
    relu_kernel<<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

CPP_SRC = "torch::Tensor relu_cuda(torch::Tensor x);"


if not torch.cuda.is_available():
    print("skip (no cuda)")
else:
    mod = load_inline(name="cudalings_relu_cuda",
                      cpp_sources=[CPP_SRC],
                      cuda_sources=[CUDA_SRC],
                      functions=["relu_cuda"],
                      verbose=False)
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], device="cuda")
    y = mod.relu_cuda(x).cpu()
    expected = torch.tensor([0.0, 0.0, 1.0, 2.0])
    print("ok" if torch.allclose(y, expected) else f"FAIL {y}")
