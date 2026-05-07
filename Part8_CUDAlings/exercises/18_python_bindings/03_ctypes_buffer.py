"""CUDAlings 18.03 -- ctypes: call a plain C function from Python.

ctypes is the "no dependencies" way to call C from Python. We compile a
shared object on the fly with the system C compiler, then dlopen it via
ctypes.CDLL. The cost: you're at the raw-pointer level. The benefit:
zero Python build dependencies.

Goal: implement `axpy_in_place` in C so that y[i] += a*x[i], compile to
a .so, and call it from Python. The validator checks the resulting sum.
"""

# I AM NOT DONE

import ctypes
import os
import subprocess
import tempfile

C_SRC = r"""
#include <stddef.h>
void axpy_in_place(size_t n, float a, const float* x, float* y) {
    /* TODO: implement the AXPY update y += a*x in place */
}
"""

def _compile():
    tmp = tempfile.mkdtemp(prefix="cl_ctypes_")
    src, lib = os.path.join(tmp, "k.c"), os.path.join(tmp, "libk.so")
    with open(src, "w") as f: f.write(C_SRC)
    subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", lib, src])
    return ctypes.CDLL(lib)

lib = _compile()
lib.axpy_in_place.argtypes = [
    ctypes.c_size_t, ctypes.c_float,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
lib.axpy_in_place.restype = None

N = 100
x = (ctypes.c_float * N)(*[1.0] * N)
y = (ctypes.c_float * N)(*[2.0] * N)
lib.axpy_in_place(N, 3.0, x, y)        # y[i] should become 5
total = sum(y)
print("ok" if abs(total - 500.0) < 1e-3 else f"FAIL {total}")
