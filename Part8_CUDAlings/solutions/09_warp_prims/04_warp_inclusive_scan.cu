#include <cstdio>
#include <cuda_runtime.h>
__inline__ __device__ float warp_inc_scan(float v) {
    int lane = threadIdx.x & 31;
    for (int d = 1; d <= 16; d <<= 1) {
        float n = __shfl_up_sync(0xffffffff, v, d);
        if (lane >= d) v += n;
    }
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
    printf("last=%.0f\n", h_out[31]);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
