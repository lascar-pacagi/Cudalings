// CUDAlings 06.05 — One thread loads, all read: broadcast via shared mem
//
// Many parallel patterns need every thread in a block to read the same
// scalar (e.g. a per-block bias, a learning rate). The clean implementation
// is "thread 0 loads from global into shared; sync; everyone reads from
// shared." That's 1 global read per block instead of blockDim.x.
//
// Goal: every thread in the block adds the per-block bias to its own
// element. Bias[blockIdx.x] is loaded once into shared memory.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK 256

__global__ void add_per_block_bias(float* x, const float* bias_per_block) {
    __shared__ float bias;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if (tid == 0) {
        // TODO: bias = bias_per_block[blockIdx.x];
    }
    __syncthreads();
    if (gid < N) {
        // TODO: x[gid] += bias;
    }
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float h_bias[N/BLOCK] = {10.f, 20.f, 30.f, 40.f};   // per-block bias

    float *d_x, *d_b;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_b, (N/BLOCK)*sizeof(float));
    cudaMemcpy(d_x, h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_bias, (N/BLOCK)*sizeof(float), cudaMemcpyHostToDevice);
    add_per_block_bias<<<N/BLOCK, BLOCK>>>(d_x, d_b);
    cudaMemcpy(h, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    // Each block adds (1 + bias) per element.
    // Block 0: 256 * 11 = 2816, blk1: 256*21=5376, blk2: 256*31=7936, blk3: 256*41=10496.
    // Total = 26624.
    printf("sum=%.0f\n", sum);
    cudaFree(d_x); cudaFree(d_b); delete[] h;
    return 0;
}
