#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 20)

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < n; i += stride) y[i] = a * x[i] + y[i];
}

int main() {
    float *h_x = new float[N], *h_y = new float[N];
    for (int i = 0; i < N; ++i) { h_x[i] = 1.0f; h_y[i] = 1.0f; }
    float *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);
    saxpy<<<128, 256>>>(N, 1.0f, d_x, d_y);
    cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sumsq = 0;
    for (int i = 0; i < N; ++i) sumsq += (double)h_y[i] * h_y[i];
    printf("l2norm=%.1f\n", std::sqrt(sumsq));
    cudaFree(d_x); cudaFree(d_y);
    delete[] h_x; delete[] h_y;
    return 0;
}
