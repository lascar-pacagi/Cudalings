#include <cstdio>
#include <cuda_runtime.h>
__global__ void k() {
    __shared__ float buf[64];
    if (threadIdx.x < 64) buf[threadIdx.x] = (float)threadIdx.x;
    __syncthreads();
}
int main() {
    int sizes[] = {64, 128, 256, 512, 1024};
    int best_size = 0, best_threads = 0;
    for (int s : sizes) {
        int blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k, s, 0);
        int active = blocks * s;
        if (active > best_threads) { best_threads = active; best_size = s; }
    }
    printf("best=%d threads=%d\n", best_size, best_threads);
    return 0;
}
