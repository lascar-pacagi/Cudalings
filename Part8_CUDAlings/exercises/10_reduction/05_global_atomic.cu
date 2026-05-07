// CUDAlings 10.05 — Reduction via global atomicAdd (the "lazy" variant)
//
// For small problem sizes or when you only need an approximate answer fast,
// `atomicAdd(&out, value)` from every thread works. It's slow when many
// threads contend, but trivially correct and 1-line short.
//
// Goal: each thread does one atomicAdd. Verify the sum.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 4096

__global__ void atomic_sum(const float* in, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // TODO: atomicAdd(out, in[i]);
    }
}

int main() {
    float* h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    atomic_sum<<<(N+255)/256, 256>>>(d_in, d_out);
    float r;
    cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum=%.0f\n", r);
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
