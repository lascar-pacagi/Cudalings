#include <cstdio>
#include <cuda_runtime.h>
#define C 8
__global__ void softmax_bwd(const float* p, const float* dy, float* dx) {
    __shared__ float sdot;
    int i = threadIdx.x;
    if (i >= C) return;
    if (i == 0) {
        float s = 0;
        for (int j = 0; j < C; ++j) s += p[j] * dy[j];
        sdot = s;
    }
    __syncthreads();
    dx[i] = p[i] * (dy[i] - sdot);
}
int main() {
    float h_p[C], h_dy[C], h_dx[C];
    for (int i = 0; i < C; ++i) { h_p[i] = 1.0f/C; h_dy[i] = (i == 0) ? 1.0f : 0.0f; }
    float *d_p, *d_dy, *d_dx;
    cudaMalloc(&d_p,  C*sizeof(float));
    cudaMalloc(&d_dy, C*sizeof(float));
    cudaMalloc(&d_dx, C*sizeof(float));
    cudaMemcpy(d_p,  h_p,  C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, h_dy, C*sizeof(float), cudaMemcpyHostToDevice);
    softmax_bwd<<<1, C>>>(d_p, d_dy, d_dx);
    cudaMemcpy(h_dx, d_dx, C*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < C; ++i) sum += h_dx[i];
    printf("sum=%.4f\n", sum);
    cudaFree(d_p); cudaFree(d_dy); cudaFree(d_dx);
    return 0;
}
