// CUDAlings 12.03 — Register-blocked tiled matmul
//
// Each thread computes a 2x2 sub-tile of C instead of one element. We still
// stage A and B in shared memory, but each thread holds 4 accumulators in
// registers. Same shared-mem reads, 4x more FMAs per shared-mem access.
//
// Layout per thread:
//   row in {2*ty, 2*ty+1}, col in {2*tx, 2*tx+1}
//   acc[2][2] held in registers
//
// Goal: implement matmul with 32x32 block tiles where each thread computes
// 2x2 outputs. Block dim is (16, 16) -- 256 threads per block, each owning
// 4 outputs = 32x32 outputs per block.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define M 64
#define N 64
#define K 64
#define BT 32     // block tile in M and N
#define TT 2      // thread tile -- each thread computes 2x2

__global__ void matmul_rb(const float* A, const float* B, float* C) {
    __shared__ float As[BT][BT];
    __shared__ float Bs[BT][BT];
    int tx = threadIdx.x;       // 0..15
    int ty = threadIdx.y;       // 0..15
    int row0 = blockIdx.y * BT + ty * TT;
    int col0 = blockIdx.x * BT + tx * TT;
    float acc[TT][TT] = {0};

    for (int t = 0; t < K; t += BT) {
        // TODO: each thread loads its TT*TT slice of A and B into shared mem
        __syncthreads();
        // TODO: contract along the BT-deep tile, updating each acc[i][j]
        __syncthreads();
    }
    // TODO: write the TT*TT register tile to its slot in C
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
    dim3 block(BT/TT, BT/TT);     // 16x16
    dim3 grid(N/BT, M/BT);         // 2x2
    matmul_rb<<<grid, block>>>(dA, dB, dC);
    cudaMemcpy(hC, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    double tr = 0;
    for (int i = 0; i < M; ++i) tr += hC[i*N + i];
    printf("trace=%.0f\n", tr);     // M * K = 4096
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    return 0;
}
