// CUDAlings 28.01 — im2col as a CUDA kernel
//
// im2col turns an (IC, IH, IW) input into a (IC*KH*KW, OH*OW) matrix where
// each column is one output position's "receptive field" flattened.
//
// One thread per output column position (h, w):
//   for ic in IC:
//     for kh in KH:
//       for kw in KW:
//         cols[(ic*KH*KW + kh*KW + kw), h*OW + w] = x[ic, h+kh, w+kw]
//
// Goal: implement the kernel; sum the resulting cols matrix.

// I AM NOT DONE

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
    // TODO: triple loop ic, kh, kw
    //   cols[(ic*KH*KW + kh*KW + kw) * OH*OW + col_idx] = x[ic*IH*IW + (h+kh)*IW + (w+kw)]
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
    // Each output column has IC*KH*KW = 9 ones. cols_n = OH*OW = 4. sum = 36.
    printf("sum=%.0f\n", sum);
    cudaFree(d_x); cudaFree(d_c); delete[] h_c;
    return 0;
}
