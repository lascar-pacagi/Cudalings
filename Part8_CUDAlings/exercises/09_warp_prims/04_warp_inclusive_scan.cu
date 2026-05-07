// CUDAlings 09.04 — Warp-level inclusive scan via __shfl_up_sync
//
// At round d in {1, 2, 4, 8, 16}:
//   neighbor = __shfl_up_sync(mask, v, d)
//   if (lane >= d) v += neighbor
//
// After log2(32)=5 rounds, lane k holds sum(input[0..k]). All in registers,
// no shared memory.
//
// Goal: input is all ones, so lane k should end up holding (k+1).
// Lane 31 should hold 32.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__inline__ __device__ float warp_inc_scan(float v) {
    int lane = threadIdx.x & 31;
    // TODO: for d in 1, 2, 4, 8, 16:
    //   float n = __shfl_up_sync(0xffffffff, v, d);
    //   if (lane >= d) v += n;
    return v;
}

__global__ void k(const float* in, float* out) {
    int lane = threadIdx.x & 31;
    float v = in[threadIdx.x];
    v = warp_inc_scan(v);
    out[lane] = v;
}

int main() {
    float h_in[32]; for (int i = 0; i < 32; ++i) h_in[i] = 1.0f;
    float h_out[32];
    float *d_in, *d_out;
    cudaMalloc(&d_in, 32*sizeof(float));
    cudaMalloc(&d_out, 32*sizeof(float));
    cudaMemcpy(d_in, h_in, 32*sizeof(float), cudaMemcpyHostToDevice);
    k<<<1, 32>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, 32*sizeof(float), cudaMemcpyDeviceToHost);
    printf("last=%.0f\n", h_out[31]);     // expected 32
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
