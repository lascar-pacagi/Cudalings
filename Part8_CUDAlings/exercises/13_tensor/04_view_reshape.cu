// CUDAlings 13.04 — Views: reshape without copying
//
// A "view" is a new (shape, stride) pair pointing at the same buffer.
// Reshaping (B, T, E) to (B*T, E) is free if the source is contiguous --
// the new strides are just (E, 1).
//
// Goal: implement `flatten_first_two` so the test below succeeds. We use
// a tiny stand-in Tensor struct with a pointer + shape + stride.

// I AM NOT DONE

#include <cstdio>

struct TView {
    const float* data;
    int dim0, dim1;
    int s0, s1;       // strides
};

TView flatten_first_two(const float* buf, int B, int T, int E) {
    // Original strides: (T*E, E, 1). After flattening (B,T) → (BT,):
    //   shape  = (B*T, E)
    //   stride = (E, 1)
    TView v;
    v.data = buf;
    // TODO: set v's two dims and two strides per the comment above
    v.dim0 = 0; v.dim1 = 0; v.s0 = 0; v.s1 = 0;
    return v;
}

int main() {
    int B = 2, T = 3, E = 4;
    float buf[24];
    for (int i = 0; i < 24; ++i) buf[i] = (float)i;

    TView v = flatten_first_two(buf, B, T, E);
    // Sum the third row (index 2): should pull elements 8, 9, 10, 11.
    float s = 0;
    for (int j = 0; j < v.dim1; ++j) s += v.data[2 * v.s0 + j * v.s1];
    printf("row2_sum=%.0f\n", s);    // 8+9+10+11 = 38
    return 0;
}
