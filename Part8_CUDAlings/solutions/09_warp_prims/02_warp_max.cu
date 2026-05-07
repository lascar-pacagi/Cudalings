#include <cstdio>
#include <cuda_runtime.h>
__inline__ __device__ float warp_max(float v) {
    for (int d = 16; d > 0; d >>= 1) {
        float o = __shfl_down_sync(0xffffffff, v, d);
        if (o > v) v = o;
    }
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
    for (int i = 0; i < 32; ++i) h[i] = (float)(i + 1);
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
