// CUDAlings 07.02 — Register pressure and its effect on occupancy
//
// Two kernels with the same launch config but different per-thread arrays:
//   light: 4 ints per thread       → small register footprint
//   heavy: 64 ints per thread      → forces register spills, lower occupancy
//
// We don't measure perf; we just query the achievable blocks/SM for each.
// The light one will report MORE blocks per SM than the heavy one.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

template <int K>
__global__ void poly(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int reg[K];
    #pragma unroll
    for (int k = 0; k < K; ++k) reg[k] = i + k;
    int s = 0;
    #pragma unroll
    for (int k = 0; k < K; ++k) s ^= reg[k];
    out[i] = s;
}

int main() {
    int blocks_light = 0, blocks_heavy = 0;
    // TODO: query the occupancy API for poly<4> and poly<64> at blockSize=256
    printf("light=%d heavy=%d\n", blocks_light, blocks_heavy);
    // The validator checks `light >= heavy` (heavier kernel never has more
    // active blocks), and that both are positive.
    return 0;
}
