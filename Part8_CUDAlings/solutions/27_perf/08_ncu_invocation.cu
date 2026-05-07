#include <cstdio>
#include <cuda_runtime.h>
#define N (1 << 24)
__global__ void scale(const float* in, float* out, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a * in[i];
}
int main() {
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemset(d_in, 0, N*sizeof(float));
    for (int i = 0; i < 5; ++i) scale<<<(N+255)/256, 256>>>(d_in, d_out, 2.0f);
    cudaDeviceSynchronize();
    printf("ncu --set basic ./08_ncu_invocation\n");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
