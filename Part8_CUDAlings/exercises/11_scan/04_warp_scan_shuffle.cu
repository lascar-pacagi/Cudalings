// CUDAlings 11.04 — Warp inclusive scan via __shfl_up_sync
//
// Same pattern as 09.04 but verified by writing every lane's value out and
// summing.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void warp_scan(const float* in, float* out) {
    int lane = threadIdx.x & 31;
    float v = in[lane];
    // TODO: 5-round shuffle-up inclusive scan (same pattern as 09.04)
    out[lane] = v;
}

int main() {
    float h_in[32], h_out[32];
    for (int i = 0; i < 32; ++i) h_in[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, 32*sizeof(float));
    cudaMalloc(&d_out, 32*sizeof(float));
    cudaMemcpy(d_in, h_in, 32*sizeof(float), cudaMemcpyHostToDevice);
    warp_scan<<<1, 32>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, 32*sizeof(float), cudaMemcpyDeviceToHost);
    // After inclusive scan of ones: out = [1,2,...,32]; sum = 32*33/2 = 528.
    double s = 0;
    for (int i = 0; i < 32; ++i) s += h_out[i];
    printf("sum=%.0f\n", s);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
