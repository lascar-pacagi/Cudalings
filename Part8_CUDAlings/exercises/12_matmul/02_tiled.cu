// CUDAlings 12.02 — Tiled matrix multiply with shared memory
//
// Naive matmul reads A and B from global memory K times each per output —
// massively memory-bound. Tiled matmul stages 16x16 chunks of A and B in
// shared memory, then has each thread accumulate from those tiles. With
// K iterating over tiles, each global element is read once per tile vs.
// once per output thread.
//
// Pattern:
//   for tile_k in 0, 16, 32, ...:
//     load As[ty][tx] = A[row, tile_k + tx]
//     load Bs[ty][tx] = B[tile_k + ty, col]
//     __syncthreads()
//     for k in 0..16: acc += As[ty][k] * Bs[k][tx]
//     __syncthreads()

// I AM NOT DONE

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
    // TODO: loop over tile_k, stage A and B, sync, accumulate, sync.
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
    printf("trace=%.0f\n", tr);  // M * K = 8192
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
