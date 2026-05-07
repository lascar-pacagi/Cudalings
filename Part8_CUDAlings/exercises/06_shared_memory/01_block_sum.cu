// CUDAlings 06.01 — Block-level sum via shared memory
//
// Goal: each block of 256 threads loads 256 elements into shared memory,
// then thread 0 sums them and writes the partial to out[blockIdx.x].
// This is a stepping stone to a real reduction (chapter 10).

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 4096

__global__ void block_sum(const float* in, float* out) {
    __shared__ float tile[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // TODO: load tile[tid] = in[gid]
    // TODO: __syncthreads()
    // TODO: if (tid == 0) sum the 256 entries and write out[blockIdx.x]
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, (N/256)*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    block_sum<<<N/256, 256>>>(d_in, d_out);
    float *hout = new float[N/256];
    cudaMemcpy(hout, d_out, (N/256)*sizeof(float), cudaMemcpyDeviceToHost);
    double total = 0;
    for (int i = 0; i < N/256; ++i) total += hout[i];
    printf("total=%.0f\n", total);  // 4096
    cudaFree(d_in); cudaFree(d_out); delete[] h; delete[] hout;
    return 0;
}
