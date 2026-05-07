// CUDAlings 26.05 — Stream compaction via exclusive scan
//
// Given an array `data` and a predicate (here: keep even values), produce
// a packed output of just the kept elements. Algorithm:
//   1. flags[i] = predicate(data[i]) ? 1 : 0
//   2. positions = exclusive_scan(flags)        // where each kept val goes
//   3. if (flags[i]) out[positions[i]] = data[i]
//
// We do single-block (N=8) so we can scan in shared memory without
// inter-block reductions. The validator checks the sum of the output.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 8

__global__ void compact_evens(const int* data, int* out, int* count) {
    __shared__ int flags[N];
    __shared__ int pos[N];
    int tid = threadIdx.x;

    int v = data[tid];
    flags[tid] = (v % 2 == 0) ? 1 : 0;
    __syncthreads();

    // Exclusive Hillis-Steele scan over flags into pos.
    // (For brevity we do a simple sequential scan in thread 0.)
    if (tid == 0) {
        pos[0] = 0;
        for (int i = 1; i < N; ++i) pos[i] = pos[i-1] + flags[i-1];
        *count = pos[N-1] + flags[N-1];
    }
    __syncthreads();

    // TODO: if (flags[tid]) out[pos[tid]] = v;
}

int main() {
    int h_data[N] = {1, 2, 3, 4, 5, 6, 7, 8};   // evens are 2,4,6,8 → sum 20
    int h_out[N]  = {0,0,0,0,0,0,0,0};
    int h_count = 0;

    int *d_data, *d_out, *d_count;
    cudaMalloc(&d_data, N*sizeof(int));
    cudaMalloc(&d_out,  N*sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_data, h_data, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(int));
    compact_evens<<<1, N>>>(d_data, d_out, d_count);
    cudaMemcpy(h_out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < h_count; ++i) sum += h_out[i];
    printf("sum=%d\n", sum);     // 2+4+6+8 = 20
    cudaFree(d_data); cudaFree(d_out); cudaFree(d_count);
    return 0;
}
