#include <cstdio>
struct TView {
    const float* data;
    int dim0, dim1;
    int s0, s1;
};
TView flatten_first_two(const float* buf, int B, int T, int E) {
    TView v;
    v.data = buf;
    v.dim0 = B*T; v.dim1 = E;
    v.s0 = E; v.s1 = 1;
    return v;
}
int main() {
    float buf[24];
    for (int i = 0; i < 24; ++i) buf[i] = (float)i;
    TView v = flatten_first_two(buf, 2, 3, 4);
    float s = 0;
    for (int j = 0; j < v.dim1; ++j) s += v.data[2 * v.s0 + j * v.s1];
    printf("row2_sum=%.0f\n", s);
    return 0;
}
