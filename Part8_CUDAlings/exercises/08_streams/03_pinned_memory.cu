// CUDAlings 08.03 — Pinned (page-locked) host memory for fast async copies
//
// cudaMallocHost (aka pinned, aka page-locked) returns host memory that
// the OS won't swap out. The DMA engine can copy directly to/from this
// memory without going through the page cache. Two consequences:
//   1. cudaMemcpyAsync is actually async only with pinned host memory.
//      With pageable memory it falls back to a synchronous copy.
//   2. Bandwidth is ~2x higher for H↔D transfers.
//
// Goal: allocate a pinned buffer, fill it, async-copy it to the device,
// verify, free with cudaFreeHost.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 4096

int main() {
    float* h_pinned = nullptr;
    // TODO: cudaMallocHost(&h_pinned, N * sizeof(float));
    if (!h_pinned) { printf("FAIL alloc\n"); return 1; }
    for (int i = 0; i < N; ++i) h_pinned[i] = 1.0f;

    float* d; cudaMalloc(&d, N*sizeof(float));
    cudaStream_t s; cudaStreamCreate(&s);
    cudaMemcpyAsync(d, h_pinned, N*sizeof(float), cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);

    cudaMemcpy(h_pinned, d, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_pinned[i];
    printf("sum=%.0f\n", sum);    // N

    cudaStreamDestroy(s);
    cudaFree(d);
    // TODO: cudaFreeHost(h_pinned);
    return 0;
}
