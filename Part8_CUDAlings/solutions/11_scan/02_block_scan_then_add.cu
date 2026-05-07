#include <cstdio>
#include <cuda_runtime.h>
#define BS 64
#define BLOCKS 4
#define N (BS * BLOCKS)
__global__ void scan_per_block(const float* in, float* out) {
    __shared__ float buf[BS];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BS + tid;
    buf[tid] = in[gid];
    __syncthreads();
    for (int d = 1; d < BS; d <<= 1) {
        float read = (tid >= d) ? buf[tid - d] : 0.0f;
        __syncthreads();
        buf[tid] += read;
        __syncthreads();
    }
    out[gid] = buf[tid];
}
int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    scan_per_block<<<BLOCKS, BS>>>(d_in, d_out);
    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("blk0_last=%.0f\n", h[BS - 1]);
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
