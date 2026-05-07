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
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_light, poly<4>,  256, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_heavy, poly<64>, 256, 0);
    printf("light=%d heavy=%d\n", blocks_light, blocks_heavy);
    return 0;
}
