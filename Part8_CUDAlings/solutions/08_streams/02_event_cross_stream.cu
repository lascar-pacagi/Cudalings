#include <cstdio>
#include <cuda_runtime.h>
#define N 1024
__global__ void scale(float* x, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= a;
}
int main() {
    float *d_x; cudaMalloc(&d_x, N*sizeof(float));
    float h[N]; for (int i = 0; i < N; ++i) h[i] = 1.0f;
    cudaMemcpy(d_x, h, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaStream_t sA, sB;
    cudaStreamCreate(&sA);
    cudaStreamCreate(&sB);
    cudaEvent_t e;
    cudaEventCreate(&e);
    scale<<<(N+255)/256, 256, 0, sA>>>(d_x, 2.0f);
    cudaEventRecord(e, sA);
    cudaStreamWaitEvent(sB, e, 0);
    scale<<<(N+255)/256, 256, 0, sB>>>(d_x, 3.0f);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.0f\n", sum);
    cudaEventDestroy(e);
    cudaStreamDestroy(sA); cudaStreamDestroy(sB);
    cudaFree(d_x);
    return 0;
}
