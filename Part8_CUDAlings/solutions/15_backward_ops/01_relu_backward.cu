#include <cstdio>
#include <cuda_runtime.h>
#define N 1024
__global__ void relu_bwd(const float* x, const float* dy, float* dx) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) dx[i] = x[i] > 0 ? dy[i] : 0.0f;
}
int main() {
    float *hx = new float[N], *hdy = new float[N];
    for (int i = 0; i < N; ++i) { hx[i] = (float)(i - 512); hdy[i] = 1.0f; }
    float *dx, *dy_d, *dx_d;
    cudaMalloc(&dx, N*sizeof(float));
    cudaMalloc(&dy_d, N*sizeof(float));
    cudaMalloc(&dx_d, N*sizeof(float));
    cudaMemcpy(dx,   hx,  N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy_d, hdy, N*sizeof(float), cudaMemcpyHostToDevice);
    relu_bwd<<<(N+255)/256, 256>>>(dx, dy_d, dx_d);
    float *hdx = new float[N];
    cudaMemcpy(hdx, dx_d, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += hdx[i];
    printf("sum=%.0f\n", sum);
    cudaFree(dx); cudaFree(dy_d); cudaFree(dx_d);
    delete[] hx; delete[] hdy; delete[] hdx;
    return 0;
}
