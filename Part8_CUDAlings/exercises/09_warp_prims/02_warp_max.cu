// CUDAlings 09.02 — Warp-shuffle max (different operator, same shape)
//
// Same xor-tree as warp_reduce(), but with `max` instead of `+`. The
// pattern is "for d in 16, 8, 4, 2, 1: combine self with __shfl_down(d)".
//
// Goal: each thread holds one float. After warp_max, lane 0 holds the
// max across the warp.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__inline__ __device__ float warp_max(float v) {
    // TODO: for d in 16, 8, 4, 2, 1:
    //   float other = __shfl_down_sync(0xffffffff, v, d);
    //   if (other > v) v = other;
    return v;
}

__global__ void k(const float* in, float* out) {
    int lane = threadIdx.x & 31;
    float v = in[threadIdx.x];
    v = warp_max(v);
    if (lane == 0) *out = v;
}

int main() {
    float h[32];
    for (int i = 0; i < 32; ++i) h[i] = (float)(i + 1);   // max is 32
    float *d_in, *d_out;
    cudaMalloc(&d_in, 32*sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h, 32*sizeof(float), cudaMemcpyHostToDevice);
    k<<<1, 32>>>(d_in, d_out);
    float r = 0;
    cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("max=%.0f\n", r);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
