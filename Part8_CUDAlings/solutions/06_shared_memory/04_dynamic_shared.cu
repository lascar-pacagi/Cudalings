#include <cstdio>
#include <cuda_runtime.h>
#define N 1024
__global__ void block_sum_dyn(const float* in, float* out) {
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    tile[tid] = in[gid];
    __syncthreads();
    if (tid == 0) {
        float s = 0;
        for (int i = 0; i < blockDim.x; ++i) s += tile[i];
        out[blockIdx.x] = s;
    }
}
int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, (N/256)*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    block_sum_dyn<<<N/256, 256, 256 * sizeof(float)>>>(d_in, d_out);
    float h_out[N/256];
    cudaMemcpy(h_out, d_out, (N/256)*sizeof(float), cudaMemcpyDeviceToHost);
    double total = 0;
    for (int i = 0; i < N/256; ++i) total += h_out[i];
    printf("total=%.0f\n", total);
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
