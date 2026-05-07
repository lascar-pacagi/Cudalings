#include <cstdio>
#include <cuda_runtime.h>
#define N 64
__global__ void transpose_tiled(const float* in, float* out) {
    __shared__ float tile[32][33];   // +1 padding to dodge bank conflicts
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    int x2 = blockIdx.y * 32 + threadIdx.x;
    int y2 = blockIdx.x * 32 + threadIdx.y;
    if (x2 < N && y2 < N)
        out[y2 * N + x2] = tile[threadIdx.x][threadIdx.y];
}
int main() {
    float *h = new float[N*N];
    for (int i = 0; i < N*N; ++i) h[i] = (float)i;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*N*sizeof(float));
    cudaMalloc(&d_out, N*N*sizeof(float));
    cudaMemcpy(d_in, h, N*N*sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(32, 32);
    dim3 grid(N/32, N/32);
    transpose_tiled<<<grid, block>>>(d_in, d_out);
    float *hout = new float[N*N];
    cudaMemcpy(hout, d_out, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N*N; ++i) sum += hout[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d_in); cudaFree(d_out); delete[] h; delete[] hout;
    return 0;
}
