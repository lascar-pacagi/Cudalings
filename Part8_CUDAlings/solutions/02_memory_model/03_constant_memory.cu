#include <cstdio>
#include <cuda_runtime.h>
#define N 1024
__constant__ float c_coeffs[5];
__global__ void apply(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = c_coeffs[i % 5];
}
int main() {
    float h_coeffs[5] = {1,2,3,4,5};
    cudaMemcpyToSymbol(c_coeffs, h_coeffs, sizeof(h_coeffs));
    float *d; cudaMalloc(&d, N*sizeof(float));
    apply<<<4, 256>>>(d);
    float *h = new float[N];
    cudaMemcpy(h, d, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.1f\n", sum);
    cudaFree(d); delete[] h;
    return 0;
}
