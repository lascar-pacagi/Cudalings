#include <cstdio>
#include <cuda_runtime.h>
#define N 4096
int main() {
    float* h_pinned = nullptr;
    cudaMallocHost(&h_pinned, N * sizeof(float));
    if (!h_pinned) { printf("FAIL alloc\n"); return 1; }
    for (int i = 0; i < N; ++i) h_pinned[i] = 1.0f;
    float* d; cudaMalloc(&d, N*sizeof(float));
    cudaStream_t s; cudaStreamCreate(&s);
    cudaMemcpyAsync(d, h_pinned, N*sizeof(float), cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
    cudaMemcpy(h_pinned, d, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_pinned[i];
    printf("sum=%.0f\n", sum);
    cudaStreamDestroy(s);
    cudaFree(d);
    cudaFreeHost(h_pinned);
    return 0;
}
