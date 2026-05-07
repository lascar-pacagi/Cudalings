#include <cstdio>
#include <cuda_runtime.h>
#define H 32
#define W 32
__global__ void row_sum_warp(const float* M, float* row_sums) {
    int row  = blockIdx.x;
    int lane = threadIdx.x;
    if (lane >= W) return;
    float v = M[row * W + lane];
    for (int d = 16; d > 0; d >>= 1)
        v += __shfl_down_sync(0xffffffff, v, d);
    if (lane == 0) row_sums[row] = v;
}
int main() {
    float *h_M = new float[H*W];
    for (int i = 0; i < H*W; ++i) h_M[i] = 1.0f;
    float *d_M, *d_rs;
    cudaMalloc(&d_M,  H*W*sizeof(float));
    cudaMalloc(&d_rs, H*sizeof(float));
    cudaMemcpy(d_M, h_M, H*W*sizeof(float), cudaMemcpyHostToDevice);
    row_sum_warp<<<H, W>>>(d_M, d_rs);
    float h_rs[H];
    cudaMemcpy(h_rs, d_rs, H*sizeof(float), cudaMemcpyDeviceToHost);
    double total = 0;
    for (int r = 0; r < H; ++r) total += h_rs[r];
    printf("total=%.0f\n", total);
    cudaFree(d_M); cudaFree(d_rs); delete[] h_M;
    return 0;
}
