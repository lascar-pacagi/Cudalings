"""CUDAlings 18.02 -- A pybind11 module compiled at runtime.

torch.utils.cpp_extension.load_inline lets you JIT-compile a C++ source
string into an importable Python module. The same machinery underlies
every "custom CUDA op" in modern PyTorch.

Goal: implement `square` as a C++ function that takes a vector<float> and
returns a vector<float> with each element squared, exposed via pybind11
through `load_inline`.
"""

# I AM NOT DONE

from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
#include <vector>
std::vector<float> square(const std::vector<float>& xs) {
    std::vector<float> out;
    out.reserve(xs.size());
    // TODO: fill `out` with the square of each input element
    return out;
}
"""

mod = load_inline(
    name="cudalings_pybind_hello",
    cpp_sources=[CPP_SRC],
    functions=["square"],
    verbose=False,
)

result = mod.square([1.0, 2.0, 3.0, 4.0])     # expect [1, 4, 9, 16]
total = sum(result)
print("ok" if abs(total - 30.0) < 1e-5 else f"FAIL got {result}")
