#include <cstdio>
#include <cuda_runtime.h>
#define N 128
#define R 1
#define K (2 * R + 1)
__constant__ float c_filter[K];
__global__ void conv1d(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < R || i >= n - R) { if (i < n) y[i] = 0.0f; return; }
    float acc = 0.f;
    for (int kk = 0; kk < K; ++kk) acc += x[i + kk - R] * c_filter[kk];
    y[i] = acc;
}
int main() {
    float h_filter[K] = {1.0f, 2.0f, 1.0f};
    cudaMemcpyToSymbol(c_filter, h_filter, sizeof(h_filter));
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMemcpy(d_x, h, N*sizeof(float), cudaMemcpyHostToDevice);
    conv1d<<<(N+63)/64, 64>>>(d_x, d_y, N);
    cudaMemcpy(h, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0;
    for (int i = 0; i < N; ++i) s += h[i];
    printf("sum=%.0f\n", s);
    cudaFree(d_x); cudaFree(d_y); delete[] h;
    return 0;
}
