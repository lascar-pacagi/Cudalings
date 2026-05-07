#include <cstdio>
#include <cuda_runtime.h>
#define N 1024
__global__ void noop(float* x) { (void)x; }
int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = (float)i;
    float *d = nullptr;
    cudaMalloc(&d, N*sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);
    noop<<<4, 256>>>(d);
    cudaMemcpy(h, d, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.1f\n", sum);
    delete[] h;
    return 0;
}
