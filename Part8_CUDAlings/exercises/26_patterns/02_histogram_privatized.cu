// CUDAlings 26.02 — Privatized histogram (per-block local + final reduce)
//
// The naive histogram serializes ~N atomic adds. Privatize:
//   1. Each block builds its own histogram in __shared__ memory.
//   2. After processing inputs, each block atomic-adds its private hist
//      to the global hist (BINS atomicAdds per block).
//
// On Pascal this is typically 5-10x faster than naive global atomics
// because shared atomics are nearly free and the global atomics are
// O(blocks * BINS) instead of O(N).

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024
#define BINS 4

__global__ void hist_priv(const int* data, int* hist, int n) {
    __shared__ int local[BINS];
    int tid = threadIdx.x;
    if (tid < BINS) local[tid] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + tid;
    if (i < n) {
        // TODO: atomicAdd(&local[data[i]], 1);   // shared-mem atomics
    }
    __syncthreads();

    // TODO: if (tid < BINS) atomicAdd(&hist[tid], local[tid]);
}

int main() {
    int *h_data = new int[N];
    for (int i = 0; i < N; ++i) h_data[i] = i % BINS;
    int h_hist[BINS] = {0};
    int *d_data, *d_hist;
    cudaMalloc(&d_data, N*sizeof(int));
    cudaMalloc(&d_hist, BINS*sizeof(int));
    cudaMemset(d_hist, 0, BINS*sizeof(int));
    cudaMemcpy(d_data, h_data, N*sizeof(int), cudaMemcpyHostToDevice);
    hist_priv<<<(N+255)/256, 256>>>(d_data, d_hist, N);
    cudaMemcpy(h_hist, d_hist, BINS*sizeof(int), cudaMemcpyDeviceToHost);
    int sum = 0;
    for (int b = 0; b < BINS; ++b) sum += h_hist[b];
    printf("total=%d\n", sum);
    cudaFree(d_data); cudaFree(d_hist); delete[] h_data;
    return 0;
}
