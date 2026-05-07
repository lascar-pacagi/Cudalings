#include <cstdio>
#include <cuda_runtime.h>
#define N 32
__global__ void blelloch_scan(const float* in, float* out) {
    __shared__ float buf[N];
    int tid = threadIdx.x;
    if (tid < N) buf[tid] = in[tid];
    __syncthreads();
    int offset = 1;
    for (int d = N / 2; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int left  = offset * (2*tid + 1) - 1;
            int right = offset * (2*tid + 2) - 1;
            buf[right] += buf[left];
        }
        offset *= 2;
    }
    if (tid == 0) buf[N - 1] = 0.0f;
    for (int d = 1; d < N; d *= 2) {
        offset /= 2;
        __syncthreads();
        if (tid < d) {
            int left  = offset * (2*tid + 1) - 1;
            int right = offset * (2*tid + 2) - 1;
            float t = buf[left];
            buf[left] = buf[right];
            buf[right] += t;
        }
    }
    __syncthreads();
    if (tid < N) out[tid] = buf[tid];
}
int main() {
    float h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    blelloch_scan<<<1, N>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("last=%.0f\n", h_out[N - 1]);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
