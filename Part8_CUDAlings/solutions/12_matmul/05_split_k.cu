#include <cstdio>
#include <cuda_runtime.h>
#define M 16
#define N 16
#define K 256
#define K_SLICES 4
#define K_PER_SLICE (K / K_SLICES)
__global__ void matmul_split_k(const float* A, const float* B, float* C) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    int slice = blockIdx.z;
    int k0 = slice * K_PER_SLICE;
    float acc = 0.f;
    for (int k = k0; k < k0 + K_PER_SLICE; ++k)
        acc += A[row*K + k] * B[k*N + col];
    atomicAdd(&C[row*N + col], acc);
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
    cudaMemset(dC, 0, M*N*sizeof(float));
    dim3 grid(N, M, K_SLICES);
    matmul_split_k<<<grid, 1>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    double tr = 0;
    for (int i = 0; i < M; ++i) tr += hC[i*N + i];
    printf("trace=%.0f\n", tr);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
