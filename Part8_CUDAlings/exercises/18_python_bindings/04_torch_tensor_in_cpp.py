"""CUDAlings 18.04 -- Take a torch::Tensor in C++; return torch::Tensor.

The natural way to write a custom PyTorch op: pass a torch::Tensor across
the binding, do the C++ work in-place or via a returned tensor, and let
PyTorch handle the storage / dtype / device.

We stay on CPU for this drill (the next exercise upgrades to a real CUDA
kernel). The op: scale every element by a scalar.

Goal: implement `cpu_scale` in C++ so y = x * a, then call it from Python.
"""

# I AM NOT DONE

import torch
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor cpu_scale(torch::Tensor x, double a) {
    // TODO: return x * a;       // torch::Tensor supports operator overloading
    return x;
}
"""

mod = load_inline(name="cudalings_torch_in",
                  cpp_sources=[CPP_SRC],
                  functions=["cpu_scale"],
                  verbose=False)

x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = mod.cpu_scale(x, 0.5)
expected = torch.tensor([0.5, 1.0, 1.5, 2.0])
print("ok" if torch.allclose(y, expected) else f"FAIL {y}")
