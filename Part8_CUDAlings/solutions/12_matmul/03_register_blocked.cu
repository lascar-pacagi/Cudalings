#include <cstdio>
#include <cuda_runtime.h>
#define M 64
#define N 64
#define K 64
#define BT 32
#define TT 2
__global__ void matmul_rb(const float* A, const float* B, float* C) {
    __shared__ float As[BT][BT];
    __shared__ float Bs[BT][BT];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row0 = blockIdx.y * BT + ty * TT;
    int col0 = blockIdx.x * BT + tx * TT;
    float acc[TT][TT] = {{0,0},{0,0}};
    for (int t = 0; t < K; t += BT) {
        for (int i = 0; i < TT; ++i)
            for (int j = 0; j < TT; ++j) {
                int ar = row0 + i, ac = t + tx * TT + j;
                int br = t + ty * TT + i, bc = col0 + j;
                As[ty*TT + i][tx*TT + j] = (ar < M && ac < K) ? A[ar*K + ac] : 0.f;
                Bs[ty*TT + i][tx*TT + j] = (br < K && bc < N) ? B[br*N + bc] : 0.f;
            }
        __syncthreads();
        for (int k = 0; k < BT; ++k)
            for (int i = 0; i < TT; ++i)
                for (int j = 0; j < TT; ++j)
                    acc[i][j] += As[ty*TT + i][k] * Bs[k][tx*TT + j];
        __syncthreads();
    }
    for (int i = 0; i < TT; ++i)
        for (int j = 0; j < TT; ++j)
            if (row0+i < M && col0+j < N)
                C[(row0+i)*N + (col0+j)] = acc[i][j];
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
    dim3 block(BT/TT, BT/TT);
    dim3 grid(N/BT, M/BT);
    matmul_rb<<<grid, block>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    double tr = 0;
    for (int i = 0; i < M; ++i) tr += hC[i*N + i];
    printf("trace=%.0f\n", tr);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
