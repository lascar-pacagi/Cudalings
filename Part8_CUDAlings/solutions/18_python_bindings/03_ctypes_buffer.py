import ctypes, os, subprocess, tempfile
C_SRC = r"""
#include <stddef.h>
void axpy_in_place(size_t n, float a, const float* x, float* y) {
    for (size_t i = 0; i < n; ++i) y[i] += a * x[i];
}
"""
def _compile():
    tmp = tempfile.mkdtemp(prefix="cl_ctypes_")
    src, lib = os.path.join(tmp, "k.c"), os.path.join(tmp, "libk.so")
    with open(src, "w") as f: f.write(C_SRC)
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", lib, src])
    return ctypes.CDLL(lib)
lib = _compile()
lib.axpy_in_place.argtypes = [ctypes.c_size_t, ctypes.c_float,
                              ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
lib.axpy_in_place.restype = None
N = 100
x = (ctypes.c_float * N)(*[1.0]*N)
y = (ctypes.c_float * N)(*[2.0]*N)
lib.axpy_in_place(N, 3.0, x, y)
print("ok" if abs(sum(y) - 500.0) < 1e-3 else "FAIL")
