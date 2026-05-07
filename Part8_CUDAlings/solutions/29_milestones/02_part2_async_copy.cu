#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CUDA_CHECK(c) do{cudaError_t e=(c); if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);}}while(0)
#define N (1 << 18)
#define CHUNKS 4
#define CHUNK_N (N / CHUNKS)
__global__ void doublify(float* x, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = gid; i < n; i += stride) x[i] *= 2.0f;
}
int main() {
    float* h_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pinned, N*sizeof(float)));
    for (int i = 0; i < N; ++i) h_pinned[i] = 1.0f;
    float* d; CUDA_CHECK(cudaMalloc(&d, N*sizeof(float)));
    cudaStream_t s[2];
    CUDA_CHECK(cudaStreamCreate(&s[0]));
    CUDA_CHECK(cudaStreamCreate(&s[1]));
    for (int c = 0; c < CHUNKS; ++c) {
        cudaStream_t cs = s[c & 1];
        size_t off = (size_t)c * CHUNK_N;
        size_t bytes = CHUNK_N * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(d + off, h_pinned + off, bytes, cudaMemcpyHostToDevice, cs));
        doublify<<<32, 128, 0, cs>>>(d + off, CHUNK_N);
        CUDA_CHECK(cudaMemcpyAsync(h_pinned + off, d + off, bytes, cudaMemcpyDeviceToHost, cs));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_pinned[i];
    printf("sum=%.0f\n", sum);
    CUDA_CHECK(cudaStreamDestroy(s[0])); CUDA_CHECK(cudaStreamDestroy(s[1]));
    CUDA_CHECK(cudaFree(d)); cudaFreeHost(h_pinned);
    return 0;
}
