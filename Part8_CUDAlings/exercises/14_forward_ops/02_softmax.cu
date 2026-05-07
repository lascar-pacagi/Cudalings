// CUDAlings 14.02 — Numerically-stable softmax (one row per block)
//
// For a row of length C, softmax(x)_j = exp(x_j - m) / sum_k exp(x_k - m)
// where m = max_k x_k. Subtracting the max keeps exp from overflowing.
//
// Goal: implement on a B=4, C=8 input. Each block handles one row,
// blockDim=8 threads work the row cooperatively in shared memory.

// I AM NOT DONE

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define B 4
#define C 8

__global__ void softmax_rows(const float* x, float* y) {
    __shared__ float buf[C];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= C) return;
    float v = x[row*C + tid];
    buf[tid] = v;
    __syncthreads();
    // TODO: reduce to find the row maximum (broadcast it via shared)
    // TODO: replace buf[tid] with exp(v - row_max), syncing afterwards
    // TODO: reduce to find the row sum
    // TODO: write the normalized softmax value into y
    y[row*C + tid] = 0.0f;  // placeholder
}

int main() {
    float h_in[B*C];
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
            h_in[b*C + c] = (float)c;          // every row is [0,1,...,7]
    float *d_in, *d_out;
    cudaMalloc(&d_in,  B*C*sizeof(float));
    cudaMalloc(&d_out, B*C*sizeof(float));
    cudaMemcpy(d_in, h_in, B*C*sizeof(float), cudaMemcpyHostToDevice);
    softmax_rows<<<B, C>>>(d_in, d_out);
    float h_out[B*C];
    cudaMemcpy(h_out, d_out, B*C*sizeof(float), cudaMemcpyDeviceToHost);
    // Each row should sum to 1.0; total should be B = 4.0.
    double sum = 0;
    for (int i = 0; i < B*C; ++i) sum += h_out[i];
    printf("sum=%.4f\n", sum);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
