#include <cstdio>
#include <cuda_runtime.h>

#define N 10000

__global__ void doublify(float* x) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < N; i += stride) x[i] = 2.0f * x[i];
}

int main() {
    float *h_x = new float[N];
    for (int i = 0; i < N; ++i) h_x[i] = 1.0f;
    float *d_x; cudaMalloc(&d_x, N*sizeof(float));
    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
    doublify<<<64, 128>>>(d_x);
    cudaMemcpy(h_x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_x[i];
    printf("sum=%.1f\n", sum);
    cudaFree(d_x); delete[] h_x;
    return 0;
}
