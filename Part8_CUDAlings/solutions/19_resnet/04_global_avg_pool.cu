#include <cstdio>
#include <cuda_runtime.h>
#define B 2
#define C 3
#define H 4
#define W 4
__global__ void gap(const float* x, float* y) {
    int b = blockIdx.y;
    int c = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float buf[];
    float local = 0.f;
    for (int i = tid; i < H * W; i += blockDim.x) local += x[((b*C + c)*H*W) + i];
    buf[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[b * C + c] = buf[0] / (float)(H * W);
}
int main() {
    float h_x[B*C*H*W];
    for (int i = 0; i < B*C*H*W; ++i) h_x[i] = 1.0f;
    float h_y[B*C] = {0};
    float *d_x, *d_y;
    cudaMalloc(&d_x, sizeof(h_x));
    cudaMalloc(&d_y, sizeof(h_y));
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    dim3 grid(C, B);
    int block = 32;
    gap<<<grid, block, block*sizeof(float)>>>(d_x, d_y);
    cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < B*C; ++i) sum += h_y[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
