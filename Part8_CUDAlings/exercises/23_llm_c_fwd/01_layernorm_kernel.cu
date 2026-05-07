// CUDAlings 23.01 -- Layernorm forward kernel (one block per row)
//
// Given x (B*T, E), compute y[b,e] = gamma[e] * (x[b,e] - mean) / std + beta[e]
// where mean and std are over the E dim of each row.
//
// Goal: implement layernorm so that running on a 4-row, E=8 input where all
// rows are [0,1,2,3,4,5,6,7] produces y rows whose sum is 0 (since mean=3.5
// and after subtracting we sum to 0). With gamma=ones, beta=zeros, the
// validator checks the row mean is ~0.

// I AM NOT DONE

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define BT 4
#define E  8

__global__ void layernorm(const float* x, float* y,
                          const float* gamma, const float* beta) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= E) return;
    __shared__ float buf[E];
    __shared__ float mean, rstd;

    float v = x[row*E + tid];
    buf[tid] = v;
    __syncthreads();

    // TODO: thread 0 computes mean and rstd; broadcast via shared mem.
    // y[row*E + tid] = gamma[tid] * (v - mean) * rstd + beta[tid];
    y[row*E + tid] = 0;
}

int main() {
    float h_x[BT*E], h_y[BT*E], h_g[E], h_b[E];
    for (int r = 0; r < BT; ++r)
        for (int e = 0; e < E; ++e) h_x[r*E + e] = (float)e;
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
    printf("row0_sum=%.4f\n", row0_sum);  // expected ~ 0
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_g); cudaFree(d_b);
    return 0;
}
