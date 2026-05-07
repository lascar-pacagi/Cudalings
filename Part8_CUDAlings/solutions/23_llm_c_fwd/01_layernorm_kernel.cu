#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#define BT 4
#define E  8
__global__ void layernorm(const float* x, float* y, const float* gamma, const float* beta) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= E) return;
    __shared__ float buf[E];
    __shared__ float mean, rstd;
    float v = x[row*E + tid];
    buf[tid] = v;
    __syncthreads();
    if (tid == 0) {
        float s = 0, ss = 0;
        for (int e = 0; e < E; ++e) { s += buf[e]; ss += buf[e]*buf[e]; }
        mean = s / E;
        float var = ss / E - mean * mean;
        rstd = rsqrtf(var + 1e-5f);
    }
    __syncthreads();
    y[row*E + tid] = gamma[tid] * (v - mean) * rstd + beta[tid];
}
int main() {
    float h_x[BT*E], h_y[BT*E], h_g[E], h_b[E];
    for (int r = 0; r < BT; ++r) for (int e = 0; e < E; ++e) h_x[r*E + e] = (float)e;
    for (int e = 0; e < E; ++e) { h_g[e] = 1.0f; h_b[e] = 0.0f; }
    float *d_x, *d_y, *d_g, *d_b;
    cudaMalloc(&d_x, BT*E*sizeof(float));
    cudaMalloc(&d_y, BT*E*sizeof(float));
    cudaMalloc(&d_g, E*sizeof(float));
    cudaMalloc(&d_b, E*sizeof(float));
    cudaMemcpy(d_x, h_x, BT*E*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, E*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, E*sizeof(float), cudaMemcpyHostToDevice);
    layernorm<<<BT, E>>>(d_x, d_y, d_g, d_b);
    cudaMemcpy(h_y, d_y, BT*E*sizeof(float), cudaMemcpyDeviceToHost);
    double row0_sum = 0;
    for (int e = 0; e < E; ++e) row0_sum += h_y[e];
    printf("row0_sum=%.4f\n", row0_sum);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_g); cudaFree(d_b);
    return 0;
}
