// CUDAlings 09.01 — Warp-level reduction with __shfl_down_sync
//
// Within a warp, lane k can hand its register value to lane k-d in one
// instruction with __shfl_down_sync(mask, val, d). Cycle d=16,8,4,2,1 and
// you've summed 32 values in 5 cycles, no shared memory needed.
//
// Goal: implement warp_reduce on 32 input values; lane 0 should hold the sum.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__inline__ __device__ float warp_reduce(float v) {
    // TODO: implement the 5-step shuffle-down sum described in the header
    return v;
}

__global__ void k(const float* in, float* out) {
    int lane = threadIdx.x & 31;
    float v = in[threadIdx.x];
    v = warp_reduce(v);
    if (lane == 0) *out = v;
}

int main() {
    float h[32];
    for (int i = 0; i < 32; ++i) h[i] = (float)(i + 1);  // 1..32, sum=528
    float *d_in, *d_out;
    cudaMalloc(&d_in, 32*sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h, 32*sizeof(float), cudaMemcpyHostToDevice);
    k<<<1, 32>>>(d_in, d_out);
    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum=%.1f\n", result);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
