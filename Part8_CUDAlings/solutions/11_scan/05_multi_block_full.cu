#include <cstdio>
#include <cuda_runtime.h>
#define BS 64
#define BLOCKS 4
#define N (BS * BLOCKS)
__global__ void block_scan(const float* in, float* out, float* block_sums) {
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
    if (tid == BS - 1) block_sums[blockIdx.x] = buf[BS - 1];
}
__global__ void scan_block_sums(float* sums, int n) {
    int tid = threadIdx.x;
    if (tid != 0) return;
    for (int i = 1; i < n; ++i) sums[i] += sums[i - 1];
}
__global__ void add_offset(float* out, const float* block_sums) {
    int b = blockIdx.x;
    if (b == 0) return;
    int tid = threadIdx.x;
    int gid = b * BS + tid;
    out[gid] += block_sums[b - 1];
}
int main() {
    float h[N]; for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out, *d_sums;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMalloc(&d_sums, BLOCKS*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    block_scan<<<BLOCKS, BS>>>(d_in, d_out, d_sums);
    scan_block_sums<<<1, 1>>>(d_sums, BLOCKS);
    add_offset<<<BLOCKS, BS>>>(d_out, d_sums);
    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("last=%.0f\n", h[N - 1]);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_sums);
    return 0;
}
