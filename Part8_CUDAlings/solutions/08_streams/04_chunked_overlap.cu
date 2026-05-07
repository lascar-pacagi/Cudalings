#include <cstdio>
#include <cuda_runtime.h>
#define N (1 << 18)
#define CHUNKS 4
#define CHUNK_N (N / CHUNKS)
__global__ void doublify(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.0f;
}
int main() {
    float* h_pinned; cudaMallocHost(&h_pinned, N * sizeof(float));
    for (int i = 0; i < N; ++i) h_pinned[i] = 1.0f;
    float* d; cudaMalloc(&d, N * sizeof(float));
    cudaStream_t s[2]; cudaStreamCreate(&s[0]); cudaStreamCreate(&s[1]);
    for (int c = 0; c < CHUNKS; ++c) {
        cudaStream_t cs = s[c & 1];
        size_t off = (size_t)c * CHUNK_N;
        size_t nbytes = CHUNK_N * sizeof(float);
        cudaMemcpyAsync(d + off, h_pinned + off, nbytes, cudaMemcpyHostToDevice, cs);
        doublify<<<(CHUNK_N+255)/256, 256, 0, cs>>>(d + off, CHUNK_N);
        cudaMemcpyAsync(h_pinned + off, d + off, nbytes, cudaMemcpyDeviceToHost, cs);
    }
    cudaDeviceSynchronize();
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_pinned[i];
    printf("sum=%.0f\n", sum);
    cudaStreamDestroy(s[0]); cudaStreamDestroy(s[1]);
    cudaFree(d); cudaFreeHost(h_pinned);
    return 0;
}
