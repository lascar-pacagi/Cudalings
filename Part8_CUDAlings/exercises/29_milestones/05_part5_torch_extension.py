"""CUDAlings 29.05 -- Part 5 Milestone: full PyTorch C++ extension.

Tie together Part 5 (Ch 17-18): library architecture + python bindings.
Build a custom op `fused_relu_scale(x, alpha) = relu(x) * alpha` exposed
as a PyTorch CUDA op via torch.utils.cpp_extension.load_inline. Wrap it
with torch.autograd.Function so it has a working backward.

If no CUDA, the exercise prints "skip (no cuda)" and the validator
accepts that.
"""

# I AM NOT DONE

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
__global__ void fwd_kernel(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (x[i] > 0 ? x[i] : 0.f) * a;
}
__global__ void bwd_kernel(const float* x, const float* dy, float* dx, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = (x[i] > 0 ? 1.f : 0.f) * dy[i] * a;
}
torch::Tensor fwd(torch::Tensor x, double a) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    fwd_kernel<<<(n+255)/256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), (float)a, n);
    return y;
}
torch::Tensor bwd(torch::Tensor x, torch::Tensor dy, double a) {
    auto dx = torch::empty_like(x);
    int n = x.numel();
    bwd_kernel<<<(n+255)/256, 256>>>(x.data_ptr<float>(), dy.data_ptr<float>(), dx.data_ptr<float>(), (float)a, n);
    return dx;
}
"""

CPP_SRC = """
torch::Tensor fwd(torch::Tensor x, double a);
torch::Tensor bwd(torch::Tensor x, torch::Tensor dy, double a);
"""

if not torch.cuda.is_available():
    print("skip (no cuda)")
else:
    mod = load_inline(name="cudalings_milestone5",
                      cpp_sources=[CPP_SRC],
                      cuda_sources=[CUDA_SRC],
                      functions=["fwd", "bwd"],
                      verbose=False)

    class FusedReluScale(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.save_for_backward(x)
            ctx.alpha = alpha
            # TODO: call into the compiled CUDA forward and return its result
            return x * 0.0

        @staticmethod
        def backward(ctx, dy):
            (x,) = ctx.saved_tensors
            # TODO: call into the compiled CUDA backward; return (dx, None) -- alpha has no grad
            return None, None

    x = torch.tensor([-1.0, 0.0, 1.0, 2.0], device="cuda", requires_grad=True)
    y = FusedReluScale.apply(x, 3.0)
    y.sum().backward()
    fwd_ok = torch.allclose(y.cpu(), torch.tensor([0., 0., 3., 6.]))
    bwd_ok = torch.allclose(x.grad.cpu(), torch.tensor([0., 0., 3., 3.]))
    print("ok" if (fwd_ok and bwd_ok) else f"FAIL")
