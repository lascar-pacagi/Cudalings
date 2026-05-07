#include <cstdio>
#include <cuda_runtime.h>
#define N 4096
__global__ void scale(float* x, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = a * x[i];
}
int main() {
    float *d1, *d2;
    cudaMalloc(&d1, N*sizeof(float));
    cudaMalloc(&d2, N*sizeof(float));
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaMemsetAsync(d1, 0, N*sizeof(float), 0);
    cudaMemsetAsync(d2, 0, N*sizeof(float), 0);
    scale<<<(N+255)/256, 256, 0, s1>>>(d1, 2.0f);
    scale<<<(N+255)/256, 256, 0, s2>>>(d2, 3.0f);
    cudaDeviceSynchronize();
    float *h1 = new float[N], *h2 = new float[N];
    cudaMemcpy(h1, d1, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h2, d2, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("done\n");
    cudaStreamDestroy(s1); cudaStreamDestroy(s2);
    cudaFree(d1); cudaFree(d2); delete[] h1; delete[] h2;
    return 0;
}
