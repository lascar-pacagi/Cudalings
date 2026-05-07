#include <cstdio>
#include <cuda_runtime.h>
#define N (1 << 20)
__global__ void copy_coalesced(const float* in, float* out) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) out[gid] = in[gid];
}
int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    copy_coalesced<<<(N+255)/256, 256>>>(d_in, d_out);
    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
