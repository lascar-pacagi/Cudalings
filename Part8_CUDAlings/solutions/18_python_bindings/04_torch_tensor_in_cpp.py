import torch
from torch.utils.cpp_extension import load_inline
CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor cpu_scale(torch::Tensor x, double a) { return x * a; }
"""
mod = load_inline(name="cudalings_torch_in", cpp_sources=[CPP_SRC],
                  functions=["cpu_scale"], verbose=False)
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = mod.cpu_scale(x, 0.5)
expected = torch.tensor([0.5, 1.0, 1.5, 2.0])
print("ok" if torch.allclose(y, expected) else f"FAIL {y}")
