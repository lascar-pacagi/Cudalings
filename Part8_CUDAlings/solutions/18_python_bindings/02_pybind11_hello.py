from torch.utils.cpp_extension import load_inline
CPP_SRC = r"""
#include <vector>
std::vector<float> square(const std::vector<float>& xs) {
    std::vector<float> out;
    out.reserve(xs.size());
    for (auto x : xs) out.push_back(x * x);
    return out;
}
"""
mod = load_inline(name="cudalings_pybind_hello", cpp_sources=[CPP_SRC],
                  functions=["square"], verbose=False)
result = mod.square([1.0, 2.0, 3.0, 4.0])
print("ok" if abs(sum(result) - 30.0) < 1e-5 else f"FAIL {result}")
