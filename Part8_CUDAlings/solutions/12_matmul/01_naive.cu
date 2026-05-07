#include <cstdio>
#include <cuda_runtime.h>
#define M 128
#define N 128
#define K 64
__global__ void matmul(const float* A, const float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float s = 0;
        for (int k = 0; k < K; ++k) s += A[row*K + k] * B[k*N + col];
        C[row*N + col] = s;
    }
}
int main() {
    float *hA = new float[M*K], *hB = new float[K*N], *hC = new float[M*N];
    for (int i = 0; i < M*K; ++i) hA[i] = 1.0f;
    for (int i = 0; i < K*N; ++i) hB[i] = 1.0f;
    float *dA, *dB, *dC;
    cudaMalloc(&dA, M*K*sizeof(float));
    cudaMalloc(&dB, K*N*sizeof(float));
    cudaMalloc(&dC, M*N*sizeof(float));
    cudaMemcpy(dA, hA, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((N+15)/16, (M+15)/16);
    matmul<<<grid, block>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    double tr = 0;
    int sz = M < N ? M : N;
    for (int i = 0; i < sz; ++i) tr += hC[i*N + i];
    printf("trace=%.0f\n", tr);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
