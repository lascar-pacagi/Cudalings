// CUDAlings 12.05 — Split-K matmul (parallelism along the inner dim)
//
// When M and N are small but K is huge, the regular tiled matmul leaves
// the GPU underutilized -- few output blocks. Split-K splits the K
// dimension across multiple "K slices", computes a partial C per slice,
// then sums them with atomicAdd.
//
//   for k in [k_lo, k_hi):
//       acc += A[m, k] * B[k, n]
//   atomicAdd(&C[m, n], acc)
//
// Goal: each block handles one (row, col, K-slice) and adds its partial
// dot product to C with atomicAdd. The validator checks the trace.

// I AM NOT DONE

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
    // TODO: for k in k0 .. k0 + K_PER_SLICE: acc += A[row*K + k] * B[k*N + col]
    // TODO: atomicAdd(&C[row*N + col], acc)
}

int main() {
    float *hA = new float[M*K], *hB = new float[K*N], *hC = new float[M*N];
    for (int i = 0; i < M*K; ++i) hA[i] = 1.0f;
    for (int i = 0; i < K*N; ++i) hB[i] = 1.0f;
    for (int i = 0; i < M*N; ++i) hC[i] = 0.0f;
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
    printf("trace=%.0f\n", tr);    // M * K = 4096
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
