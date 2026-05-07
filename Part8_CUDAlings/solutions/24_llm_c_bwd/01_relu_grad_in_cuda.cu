#include <cstdio>
#include <cuda_runtime.h>
#define N 4
__global__ void sigmoid_bwd(const float* y, const float* dy, float* dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dx[i] = dy[i] * y[i] * (1.0f - y[i]);
}
int main() {
    float h_y[N] = {0.5, 0.5, 0.5, 0.5};
    float h_dy[N] = {1, 1, 1, 1};
    float h_dx[N] = {0};
    float *d_y, *d_dy, *d_dx;
    cudaMalloc(&d_y,  N*sizeof(float));
    cudaMalloc(&d_dy, N*sizeof(float));
    cudaMalloc(&d_dx, N*sizeof(float));
    cudaMemcpy(d_y,  h_y,  N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, h_dy, N*sizeof(float), cudaMemcpyHostToDevice);
    sigmoid_bwd<<<1, N>>>(d_y, d_dy, d_dx);
    cudaMemcpy(h_dx, d_dx, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_dx[i];
    printf("sum=%.4f\n", sum);
    cudaFree(d_y); cudaFree(d_dy); cudaFree(d_dx);
    return 0;
}
