// CUDAlings 10.01 — Block-level tree reduction
//
// Pattern: at step s = blockDim/2, blockDim/4, ..., 1, thread t (if t < s)
// adds tile[t + s] into tile[t], synchronizing between steps. After log2
// steps, tile[0] holds the block sum.
//
// Goal: implement the reduction. Validate by summing N=4096 ones; expect 4096.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 4096

__global__ void block_reduce(const float* in, float* out) {
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    tile[tid] = (gid < N) ? in[gid] : 0.0f;
    __syncthreads();

    // TODO: for s in blockDim/2, /4, ... 1: if (tid < s) tile[tid] += tile[tid + s]; __syncthreads();
    if (tid == 0) out[blockIdx.x] = tile[0];
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, (N/256)*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    block_reduce<<<N/256, 256, 256*sizeof(float)>>>(d_in, d_out);
    float *hout = new float[N/256];
    cudaMemcpy(hout, d_out, (N/256)*sizeof(float), cudaMemcpyDeviceToHost);
    double total = 0;
    for (int i = 0; i < N/256; ++i) total += hout[i];
    printf("total=%.0f\n", total);
    cudaFree(d_in); cudaFree(d_out); delete[] h; delete[] hout;
    return 0;
}
