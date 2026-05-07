// CUDAlings 13.02 — Computing element offsets via strides
//
// PyTorch and every serious tensor library represents shape + memory
// layout as (shape[N], stride[N]). For a contiguous (B, T, E) tensor:
//   stride = (T*E, E, 1)
//   offset(b, t, e) = b * stride[0] + t * stride[1] + e * stride[2]
//
// Goal: write `offset3d` so the test below sums the diagonal (where b=t=e)
// and prints 0+1+2+3 = 6 for shape (4, 4, 4).

// I AM NOT DONE

#include <cstdio>

int offset3d(int b, int t, int e, int sb, int st, int se) {
    // TODO: return b*sb + t*st + e*se
    return 0;
}

int main() {
    int B = 4, T = 4, E = 4;
    int sb = T*E, st = E, se = 1;
    int data[64];
    for (int i = 0; i < 64; ++i) data[i] = i;     // contiguous

    int sum = 0;
    for (int i = 0; i < 4; ++i) sum += data[offset3d(i, i, i, sb, st, se)];
    printf("diag=%d\n", sum);   // expected (0,0,0)=0 + (1,1,1)=21 + (2,2,2)=42 + (3,3,3)=63 = 126
    return 0;
}
