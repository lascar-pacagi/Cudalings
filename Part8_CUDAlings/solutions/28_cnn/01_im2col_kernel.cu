#include <cstdio>
#include <cuda_runtime.h>
#define IC 1
#define IH 4
#define IW 4
#define KH 3
#define KW 3
#define OH (IH - KH + 1)
#define OW (IW - KW + 1)
__global__ void im2col(const float* x, float* cols) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (w >= OW || h >= OH) return;
    int col_idx = h * OW + w;
    int OHW = OH * OW;
    for (int ic = 0; ic < IC; ++ic)
        for (int kh = 0; kh < KH; ++kh)
            for (int kw = 0; kw < KW; ++kw) {
                int row = ic * KH * KW + kh * KW + kw;
                cols[row * OHW + col_idx] = x[ic*IH*IW + (h+kh)*IW + (w+kw)];
            }
}
int main() {
    float h_x[IC*IH*IW];
    for (int i = 0; i < IC*IH*IW; ++i) h_x[i] = 1.0f;
    int rows = IC*KH*KW;
    int cols_n = OH*OW;
    float* h_c = new float[rows * cols_n];
    float *d_x, *d_c;
    cudaMalloc(&d_x, sizeof(h_x));
    cudaMalloc(&d_c, rows * cols_n * sizeof(float));
    cudaMemset(d_c, 0, rows * cols_n * sizeof(float));
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    dim3 block(8, 8);
    dim3 grid((OW+7)/8, (OH+7)/8);
    im2col<<<grid, block>>>(d_x, d_c);
    cudaMemcpy(h_c, d_c, rows * cols_n * sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < rows * cols_n; ++i) sum += h_c[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d_x); cudaFree(d_c); delete[] h_c;
    return 0;
}
