#include <cstdio>
#include <cuda_runtime.h>
#define N 1024
#define BLOCK 256
__global__ void add_per_block_bias(float* x, const float* bias_per_block) {
    __shared__ float bias;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if (tid == 0) bias = bias_per_block[blockIdx.x];
    __syncthreads();
    if (gid < N) x[gid] += bias;
}
int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float h_bias[N/BLOCK] = {10.f, 20.f, 30.f, 40.f};
    float *d_x, *d_b;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_b, (N/BLOCK)*sizeof(float));
    cudaMemcpy(d_x, h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_bias, (N/BLOCK)*sizeof(float), cudaMemcpyHostToDevice);
    add_per_block_bias<<<N/BLOCK, BLOCK>>>(d_x, d_b);
    cudaMemcpy(h, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d_x); cudaFree(d_b); delete[] h;
    return 0;
}
