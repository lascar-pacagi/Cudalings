// CUDAlings 29.02 — Part 2 Milestone: optimized async copy
//
// Tie together Part 2 (Ch 05-09): coalescing, shared memory, occupancy,
// streams, warp primitives.
//
// Build a copy that:
//   1. Uses pinned host memory (Ch 08).
//   2. Uses two streams + chunked H2D + kernel + D2H (Ch 08).
//   3. Kernel is coalesced and grid-stride (Ch 05, Ch 01).
//   4. CUDA_CHECK every call (Ch 04).
//
// Validator: final sum of doubled buffer is correct.

// I AM NOT DONE

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CUDA_CHECK(c) do{cudaError_t e=(c); if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}while(0)

#define N (1 << 18)
#define CHUNKS 4
#define CHUNK_N (N / CHUNKS)

__global__ void doublify(float* x, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // TODO: grid-stride loop, x[i] *= 2.0f
}

int main() {
    float* h_pinned = nullptr;
    // TODO: cudaMallocHost(&h_pinned, N*sizeof(float))
    if (!h_pinned) { printf("FAIL alloc\n"); return 1; }
    for (int i = 0; i < N; ++i) h_pinned[i] = 1.0f;

    float* d; CUDA_CHECK(cudaMalloc(&d, N*sizeof(float)));
    cudaStream_t s[2];
    CUDA_CHECK(cudaStreamCreate(&s[0]));
    CUDA_CHECK(cudaStreamCreate(&s[1]));

    for (int c = 0; c < CHUNKS; ++c) {
        cudaStream_t cs = s[c & 1];
        size_t off = (size_t)c * CHUNK_N;
        size_t bytes = CHUNK_N * sizeof(float);
        // TODO: H2D async, kernel on stream cs, D2H async
        (void)cs; (void)off; (void)bytes;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_pinned[i];
    printf("sum=%.0f\n", sum);     // 2 * N = 524288

    CUDA_CHECK(cudaStreamDestroy(s[0])); CUDA_CHECK(cudaStreamDestroy(s[1]));
    CUDA_CHECK(cudaFree(d));
    cudaFreeHost(h_pinned);
    return 0;
}
