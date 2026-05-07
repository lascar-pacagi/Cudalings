#include <cstdio>
#include <cuda_runtime.h>
#define N 1024
#define BINS 4
__global__ void histogram(const int* data, int* hist, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&hist[data[i]], 1);
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
    printf("total=%d\n", sum);
    cudaFree(d_data); cudaFree(d_hist); delete[] h_data;
    return 0;
}
