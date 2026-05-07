#include <cstdio>
#include <cuda_runtime.h>
#define N (1 << 22)
__global__ void uniform_branch(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int warp = i / 32;
    if (warp & 1) x[i] = sinf(x[i]); else x[i] = cosf(x[i]);
}
__global__ void checker_branch(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i & 1) x[i] = sinf(x[i]); else x[i] = cosf(x[i]);
}
float time_kernel(void (*f)(float*), float* d) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f<<<(N+255)/256, 256>>>(d);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < 10; ++i) f<<<(N+255)/256, 256>>>(d);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / 10.0f;
}
int main() {
    float* d; cudaMalloc(&d, N*sizeof(float));
    cudaMemset(d, 0, N*sizeof(float));
    float t_uniform = time_kernel(uniform_branch, d);
    float t_checker = time_kernel(checker_branch, d);
    printf("uniform_ms=%.3f\n", t_uniform);
    printf("checker_ms=%.3f\n", t_checker);
    cudaFree(d);
    return 0;
}
