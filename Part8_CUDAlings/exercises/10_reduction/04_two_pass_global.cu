// CUDAlings 10.04 — Two-pass global reduction for arbitrary N
//
// You can't grid-sync within a single kernel on Pascal, so a global
// reduction needs two kernel launches:
//   Pass 1: each block reduces its chunk → partials[block_count]
//   Pass 2: one block reduces partials → final scalar
//
// Goal: drive the kernel from `block_reduce` (chapter 10.01) twice.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 16)
#define BLOCK 256

__global__ void block_reduce(const float* in, float* out, int n) {
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    tile[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) tile[tid] += tile[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = tile[0];
}

int main() {
    float* h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_partials, *d_final;
    int blocks = (N + BLOCK - 1) / BLOCK;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_partials, blocks*sizeof(float));
    cudaMalloc(&d_final, sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);

    // TODO: Pass 1: block_reduce<<<blocks, BLOCK, BLOCK*sizeof(float)>>>(d_in, d_partials, N);
    // TODO: Pass 2: block_reduce<<<1, BLOCK, BLOCK*sizeof(float)>>>(d_partials, d_final, blocks);

    float r;
    cudaMemcpy(&r, d_final, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum=%.0f\n", r);   // expected N = 65536
    cudaFree(d_in); cudaFree(d_partials); cudaFree(d_final); delete[] h;
    return 0;
}
