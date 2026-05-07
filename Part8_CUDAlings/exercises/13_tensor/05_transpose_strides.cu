// CUDAlings 13.05 — Transpose as a strides swap (not a memcpy)
//
// `T.transpose(0, 1)` in PyTorch returns a view with strides (s1, s0)
// instead of (s0, s1). The buffer is unchanged. This is why
// `x.transpose().contiguous()` is necessary before some kernels --
// .contiguous() is the call that actually reorders memory.
//
// Goal: write `swap_strides_2d` and verify a 3x4 matrix logically becomes
// 4x3 with the same buffer. We sum the first "logical" column of the
// transposed view.

// I AM NOT DONE

#include <cstdio>

struct TView2D { const int* data; int d0, d1, s0, s1; };

TView2D swap_strides_2d(TView2D v) {
    // TODO: return a view with the two dims and the two strides swapped
    return v;
}

int main() {
    int buf[12] = { 0, 1, 2, 3,
                    4, 5, 6, 7,
                    8, 9, 10, 11 };
    TView2D v = {buf, 3, 4, 4, 1};        // rows=3, cols=4
    TView2D t = swap_strides_2d(v);        // logically 4 x 3

    // Sum t's first column (logical column 0 of the transposed view):
    // that's buf[0], buf[1], buf[2], buf[3] in original layout = 0+1+2+3 = 6.
    int s = 0;
    for (int r = 0; r < t.d0; ++r) s += t.data[r * t.s0 + 0 * t.s1];
    printf("col0_sum=%d\n", s);   // expected 6
    return 0;
}
