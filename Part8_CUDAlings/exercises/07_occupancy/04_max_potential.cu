// CUDAlings 07.04 — cudaOccupancyMaxPotentialBlockSize
//
// One call returns BOTH the recommended grid size and block size to
// achieve the highest theoretical occupancy for your kernel. This is
// the "I just want defaults" API and it's almost always a great start.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void k(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (float)i;
}

int main() {
    int min_grid = 0, block = 0;
    // TODO: cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, k, 0, 0);
    printf("recommended_block=%d\n", block);
    return 0;
}
