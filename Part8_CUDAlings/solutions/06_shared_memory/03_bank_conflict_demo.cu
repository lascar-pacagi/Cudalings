#include <cstdio>
#include <cuda_runtime.h>
#define N 32
__global__ void col_sum_padded(const float* in, float* out) {
    __shared__ float tile[N][N + 1];
    int row = threadIdx.y;
    int col = threadIdx.x;
    tile[row][col] = in[row * N + col];
    __syncthreads();
    if (row == 0) {
        float s = 0;
        for (int r = 0; r < N; ++r) s += tile[r][col];
        out[col] = s;
    }
}
int main() {
    float h_in[N*N];
    for (int i = 0; i < N*N; ++i) h_in[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*N*sizeof(float), cudaMemcpyHostToDevice);
    col_sum_padded<<<1, dim3(N, N)>>>(d_in, d_out);
    float h_out[N];
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    double total = 0;
    for (int c = 0; c < N; ++c) total += h_out[c];
    printf("total=%.0f\n", total);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
