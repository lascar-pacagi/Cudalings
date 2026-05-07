#include <cstdio>
#include <cuda_runtime.h>
#define N 256
#define BLOCK 64
__global__ void stencil_3pt(const float* x, float* y, int n) {
    __shared__ float tile[BLOCK + 2];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    tile[tid + 1] = (gid < n) ? x[gid] : 0.f;
    if (tid == 0) {
        tile[0]         = (gid > 0)             ? x[gid - 1]     : 0.f;
        tile[BLOCK + 1] = (gid + BLOCK < n)     ? x[gid + BLOCK] : 0.f;
    }
    __syncthreads();
    if (gid < n) y[gid] = (tile[tid] + tile[tid + 1] + tile[tid + 2]) / 3.0f;
}
int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMemcpy(d_x, h, N*sizeof(float), cudaMemcpyHostToDevice);
    stencil_3pt<<<N / BLOCK, BLOCK>>>(d_x, d_y, N);
    cudaMemcpy(h, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    float s = 0;
    for (int i = 1; i < N - 1; ++i) s += h[i];
    printf("sum=%.0f\n", s);
    cudaFree(d_x); cudaFree(d_y); delete[] h;
    return 0;
}
