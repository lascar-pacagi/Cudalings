// CUDAlings 08.04 — Pipeline H2D copy + compute via chunks and streams
//
// The classic "fill the pipeline" pattern:
//
//   chunk 0 H2D ─┐
//                  chunk 0 KERNEL ─┐
//   chunk 1 H2D ─┘                  chunk 1 KERNEL ─┐
//                  chunk 2 H2D ───┘                  chunk 2 KERNEL ...
//
// While the GPU is computing on chunk N, we transfer chunk N+1. With
// pinned host memory + multiple streams the H2D copies overlap kernel
// execution. Total time approaches max(H2D, compute) instead of their sum.
//
// Goal: split a large array into 4 chunks, alternate two streams. The
// kernel just doubles each element. Validate the final sum is correct.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 18)
#define CHUNKS 4
#define CHUNK_N (N / CHUNKS)

__global__ void doublify(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.0f;
}

int main() {
    float* h_pinned;
    cudaMallocHost(&h_pinned, N * sizeof(float));
    for (int i = 0; i < N; ++i) h_pinned[i] = 1.0f;

    float* d; cudaMalloc(&d, N * sizeof(float));

    cudaStream_t s[2];
    cudaStreamCreate(&s[0]);
    cudaStreamCreate(&s[1]);

    for (int c = 0; c < CHUNKS; ++c) {
        cudaStream_t cs = s[c & 1];
        size_t off    = (size_t)c * CHUNK_N;
        size_t nbytes = CHUNK_N * sizeof(float);
        // TODO: enqueue this chunk's three steps (H2D, kernel, D2H) on stream `cs`
        //       so different chunks can overlap on different streams.
    }
    cudaDeviceSynchronize();

    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_pinned[i];
    printf("sum=%.0f\n", sum);   // 2 * N = 524288

    cudaStreamDestroy(s[0]); cudaStreamDestroy(s[1]);
    cudaFree(d); cudaFreeHost(h_pinned);
    return 0;
}
