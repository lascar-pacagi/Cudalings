#include <cstdio>
#include <cuda_runtime.h>
#define N (1 << 20)
#define STRIDE 32
__global__ void copy_strided(const float* in, float* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = t * STRIDE;
    if (idx < N) out[idx] = in[idx];
}
int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemset(d_out, 0, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    int threads_total = N / STRIDE;
    copy_strided<<<(threads_total + 255) / 256, 256>>>(d_in, d_out);
    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
