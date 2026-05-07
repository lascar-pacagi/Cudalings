#include <cstdio>
#include <cuda_runtime.h>
__global__ void k() {
    __shared__ float buf[256];
    buf[threadIdx.x] = threadIdx.x;
    __syncthreads();
}
int main() {
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, k, 256, 0);
    printf("blocks_per_sm=%d\n", blocks_per_sm);
    return 0;
}
