#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (float)i;
}
int main() {
    int min_grid = 0, block = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, k, 0, 0);
    printf("recommended_block=%d\n", block);
    return 0;
}
