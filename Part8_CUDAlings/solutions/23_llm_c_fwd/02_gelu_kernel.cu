#include <cstdio>
#include <cuda_runtime.h>
#define N 7
__global__ void gelu(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = x[i];
    const float k = 0.7978845608f;
    float u = k * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + tanhf(u));
}
int main() {
    float h[N] = {-3, -2, -1, 0, 1, 2, 3};
    float *d; cudaMalloc(&d, N*sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);
    gelu<<<1, N>>>(d);
    cudaMemcpy(h, d, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.4f\n", sum);
    cudaFree(d);
    return 0;
}
