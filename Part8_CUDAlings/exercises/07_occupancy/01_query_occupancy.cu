// CUDAlings 07.01 — Query achievable occupancy
//
// cudaOccupancyMaxActiveBlocksPerMultiprocessor tells you how many blocks of
// a given launch config can fit on one SM, given register/shared-mem usage.
// Useful for confirming that a launch isn't underutilizing the GPU.
//
// Goal: print "blocks_per_sm=<n>" for the kernel below at blockDim=256.
// Any positive integer passes.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void k() {
    __shared__ float buf[256];
    buf[threadIdx.x] = threadIdx.x;
    __syncthreads();
}

int main() {
    int blocks_per_sm = 0;
    // TODO: cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //          &blocks_per_sm, k, 256 /*blockSize*/, 0 /*dynamicSMem*/)
    printf("blocks_per_sm=%d\n", blocks_per_sm);
    return 0;
}
