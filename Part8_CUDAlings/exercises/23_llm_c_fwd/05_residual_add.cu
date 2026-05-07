// CUDAlings 23.05 — In-place residual add: x[i] += y[i]
//
// The shortest CUDA kernel in any transformer. Used twice per block:
// once after attention, once after MLP. Trivially memory-bound.
//
// Goal: implement and verify.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024

__global__ void residual_add(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // TODO: x[i] += y[i];
    }
}

int main() {
    float h[N], h_y[N];
    for (int i = 0; i < N; ++i) { h[i] = 1.0f; h_y[i] = 2.0f; }
    float *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMemcpy(d_x, h,   N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);
    residual_add<<<(N+255)/256, 256>>>(d_x, d_y, N);
    cudaMemcpy(h, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.0f\n", sum);     // 3 * N = 3072
    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
