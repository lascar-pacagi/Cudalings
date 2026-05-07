// CUDAlings 27.04 — Compare cudaMemcpy directions: H2D, D2D, D2H
//
// Three transfer types have very different bandwidths:
//   D2D (device-to-device):  pure HBM/GDDR bandwidth   -- fastest
//   H2D / D2H (over PCIe):    bottlenecked by PCIe gen3 ~12 GB/s
//                              with pinned host memory; ~6 GB/s pageable
//
// Goal: time each direction with cudaEvents and print all three. We
// validate that each line is present.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 24)        // 64 MB
#define BYTES (N * sizeof(float))

float time_memcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind k) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    cudaMemcpy(dst, src, bytes, k);   // warm up
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

    float h2d = 0, d2d = 0, d2h = 0;
    // TODO: time each of the three memcpy directions (H2D, D2D, D2H) via the helper

    auto bw = [](float ms) { return BYTES / (ms / 1e3) / 1e9; };
    printf("h2d=%.1f GB/s\n", bw(h2d));
    printf("d2d=%.1f GB/s\n", bw(d2d));
    printf("d2h=%.1f GB/s\n", bw(d2h));

    cudaFreeHost(h_pinned);
    cudaFree(d_a); cudaFree(d_b);
    return 0;
}
