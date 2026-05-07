#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#define B 4
#define C 8
__global__ void softmax_rows(const float* x, float* y) {
    __shared__ float buf[C];
    __shared__ float row_max, row_sum;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= C) return;
    float v = x[row*C + tid];
    buf[tid] = v;
    __syncthreads();
    if (tid == 0) {
        float m = buf[0];
        for (int k = 1; k < C; ++k) if (buf[k] > m) m = buf[k];
        row_max = m;
    }
    __syncthreads();
    float e = expf(v - row_max);
    buf[tid] = e;
    __syncthreads();
    if (tid == 0) {
        float s = 0;
        for (int k = 0; k < C; ++k) s += buf[k];
        row_sum = s;
    }
    __syncthreads();
    y[row*C + tid] = e / row_sum;
}
int main() {
    float h_in[B*C];
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c) h_in[b*C + c] = (float)c;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  B*C*sizeof(float));
    cudaMalloc(&d_out, B*C*sizeof(float));
    cudaMemcpy(d_in, h_in, B*C*sizeof(float), cudaMemcpyHostToDevice);
    softmax_rows<<<B, C>>>(d_in, d_out);
    float h_out[B*C];
    cudaMemcpy(h_out, d_out, B*C*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < B*C; ++i) sum += h_out[i];
    printf("sum=%.4f\n", sum);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
