// CUDAlings 26.01 — Histogram with atomicAdd
//
// Each input value `data[i] in [0, BINS)` votes for bin `data[i]`. Many
// threads will hit the same bin -- atomicAdd is the safe write.
//
// Goal: for N inputs uniformly distributed across BINS=4, the kernel
// should produce roughly N/4 in each bin. We seed a deterministic input
// so the test is exact: data[i] = i % 4. Each bin gets N/4 votes.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024
#define BINS 4

__global__ void histogram(const int* data, int* hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int v = data[i];
    // TODO: increment the hist bin for v atomically
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
    histogram<<<(N+255)/256, 256>>>(d_data, d_hist, N);
    cudaMemcpy(h_hist, d_hist, BINS*sizeof(int), cudaMemcpyDeviceToHost);
    int sum = 0;
    for (int b = 0; b < BINS; ++b) sum += h_hist[b];
    printf("total=%d\n", sum);   // expected N=1024
    cudaFree(d_data); cudaFree(d_hist); delete[] h_data;
    return 0;
}
