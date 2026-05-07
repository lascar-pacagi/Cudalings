#include <cstdio>
#include <cuda_runtime.h>
#define M 128
#define N 128
#define K 64
#define T 16
__global__ void matmul_tiled(const float* A, const float* B, float* C) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float acc = 0;
    for (int tk = 0; tk < K; tk += T) {
        As[threadIdx.y][threadIdx.x] =
            (row < M && tk + threadIdx.x < K) ? A[row*K + tk + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (tk + threadIdx.y < K && col < N) ? B[(tk + threadIdx.y)*N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N + col] = acc;
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
    dim3 block(T, T);
    dim3 grid(N/T, M/T);
    matmul_tiled<<<grid, block>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    double tr = 0;
    for (int i = 0; i < M; ++i) tr += hC[i*N + i];
    printf("trace=%.0f\n", tr);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
