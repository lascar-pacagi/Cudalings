#include <cstdio>
#include <cuda_runtime.h>
#define N (1 << 24)
#define BYTES (N * sizeof(float))
float time_memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind k) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaMemcpy(dst, src, bytes, k);
    cudaEventRecord(a);
    for (int i = 0; i < 5; ++i) cudaMemcpy(dst, src, bytes, k);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / 5.0f;
}
int main() {
    float *h_pinned; cudaMallocHost(&h_pinned, BYTES);
    float *d_a, *d_b;
    cudaMalloc(&d_a, BYTES);
    cudaMalloc(&d_b, BYTES);
    for (int i = 0; i < N; ++i) h_pinned[i] = 1.0f;
    float h2d = time_memcpy(d_a, h_pinned, BYTES, cudaMemcpyHostToDevice);
    float d2d = time_memcpy(d_b, d_a,      BYTES, cudaMemcpyDeviceToDevice);
    float d2h = time_memcpy(h_pinned, d_b, BYTES, cudaMemcpyDeviceToHost);
    auto bw = [](float ms) { return BYTES / (ms / 1e3) / 1e9; };
    printf("h2d=%.1f GB/s\n", bw(h2d));
    printf("d2d=%.1f GB/s\n", bw(d2d));
    printf("d2h=%.1f GB/s\n", bw(d2h));
    cudaFreeHost(h_pinned);
    cudaFree(d_a); cudaFree(d_b);
    return 0;
}
