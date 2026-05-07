#include <cstdio>
#include <cuda_runtime.h>

#define N 1024

__global__ void add(const float* a, const float* b, float* c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main() {
    float *h_a = new float[N], *h_b = new float[N], *h_c = new float[N];
    for (int i = 0; i < N; ++i) { h_a[i] = i; h_b[i] = 2.0f * i; }
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    add<<<blocks, threads>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_c[i];
    printf("sum=%.1f\n", sum);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    return 0;
}
