#include <cstdio>
#include <cuda_runtime.h>
#define N (1 << 24)
__global__ void copy(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}
int main() {
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemset(d_in, 0, N*sizeof(float));
    copy<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    int reps = 10;
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) copy<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    double bytes_per_iter = 2.0 * N * sizeof(float);
    double seconds = (ms / 1e3) / reps;
    double gbps = bytes_per_iter / seconds / 1e9;
    printf("bw=%.1f GB/s\n", gbps);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return 0;
}
