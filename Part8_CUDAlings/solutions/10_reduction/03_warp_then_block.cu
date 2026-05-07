#include <cstdio>
#include <cuda_runtime.h>
#define N 256
__inline__ __device__ float warp_sum(float v) {
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}
__global__ void block_sum(const float* in, float* out) {
    __shared__ float warp_results[8];
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid & 31;
    float v = in[tid];
    v = warp_sum(v);
    if (lane == 0) warp_results[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        float w = (lane < 8) ? warp_results[lane] : 0.0f;
        w = warp_sum(w);
        if (lane == 0) *out = w;
    }
}
int main() {
    float h[N]; for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    block_sum<<<1, N>>>(d_in, d_out);
    float r = 0;
    cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum=%.0f\n", r);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
