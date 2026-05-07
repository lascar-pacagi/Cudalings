#include <cstdio>
#include <cuda_runtime.h>
#define N 4096
__global__ void scale(float* x, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = a * x[i];
}
int main() {
    float *p = nullptr;
    cudaMallocManaged(&p, N*sizeof(float));
    for (int i = 0; i < N; ++i) p[i] = 1.0f;
    scale<<<16, 256>>>(p, 3.0f);
    cudaDeviceSynchronize();
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += p[i];
    printf("sum=%.1f\n", sum);
    cudaFree(p);
    return 0;
}
