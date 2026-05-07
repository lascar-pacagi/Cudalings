#include <cstdio>
#include <cuda_runtime.h>
__global__ void noop() {}
int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) noop<<<1, 32>>>();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("elapsed=%.3f\n", ms);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
