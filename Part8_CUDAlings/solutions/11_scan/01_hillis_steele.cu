#include <cstdio>
#include <cuda_runtime.h>
#define N 128
__global__ void scan(const float* in, float* out) {
    __shared__ float buf[N];
    int tid = threadIdx.x;
    buf[tid] = in[tid];
    __syncthreads();
    for (int d = 1; d < N; d <<= 1) {
        float read = (tid >= d) ? buf[tid - d] : 0.0f;
        __syncthreads();
        buf[tid] += read;
        __syncthreads();
    }
    out[tid] = buf[tid];
}
int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    scan<<<1, N>>>(d_in, d_out);
    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("last=%.0f\n", h[N-1]);
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
