#include <cstdio>
#include <cuda_runtime.h>
__global__ void broadcast(float* sum_out) {
    int lane = threadIdx.x & 31;
    float v = (lane == 0) ? 42.0f : 0.0f;
    v = __shfl_sync(0xffffffff, v, 0);
    for (int d = 16; d > 0; d >>= 1)
        v += __shfl_down_sync(0xffffffff, v, d);
    if (lane == 0) *sum_out = v;
}
int main() {
    float* d_out; cudaMalloc(&d_out, sizeof(float));
    broadcast<<<1, 32>>>(d_out);
    float h = 0;
    cudaMemcpy(&h, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum=%.0f\n", h);
    cudaFree(d_out);
    return 0;
}
