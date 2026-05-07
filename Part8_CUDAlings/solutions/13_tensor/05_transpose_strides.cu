#include <cstdio>
struct TView2D { const int* data; int d0, d1, s0, s1; };
TView2D swap_strides_2d(TView2D v) { return TView2D{v.data, v.d1, v.d0, v.s1, v.s0}; }
int main() {
    int buf[12] = { 0,1,2,3, 4,5,6,7, 8,9,10,11 };
    TView2D v = {buf, 3, 4, 4, 1};
    TView2D t = swap_strides_2d(v);
    int s = 0;
    for (int r = 0; r < t.d0; ++r) s += t.data[r * t.s0 + 0 * t.s1];
    printf("col0_sum=%d\n", s);
    return 0;
}
