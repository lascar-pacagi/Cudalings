// CUDAlings 14.04 — Naive 2D convolution (no padding, stride 1)
//
// y[oc, h, w] = sum_{ic, kh, kw} x[ic, h+kh, w+kw] * W[oc, ic, kh, kw]
//
// One thread per output element. The point is the indexing math, not
// performance — Winograd or im2col + matmul are how you'd do this for real.
//
// Goal: implement the kernel for a tiny case: in_channels=1, out_channels=1,
// kernel 3x3 = ones, input 5x5 = ones. Output is 3x3 of nines.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define IC 1
#define OC 1
#define IH 5
#define IW 5
#define KH 3
#define KW 3
#define OH (IH - KH + 1)
#define OW (IW - KW + 1)

__global__ void conv2d_naive(const float* x, const float* W, float* y) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    if (ow >= OW || oh >= OH) return;

    float acc = 0.f;
    // TODO: loops over ic, kh, kw
    //   acc += x[(ic)*IH*IW + (oh+kh)*IW + (ow+kw)] * W[oc*IC*KH*KW + ic*KH*KW + kh*KW + kw]
    y[oc*OH*OW + oh*OW + ow] = acc;
}

int main() {
    float h_x[IC*IH*IW], h_W[OC*IC*KH*KW], h_y[OC*OH*OW];
    for (int i = 0; i < IC*IH*IW; ++i)     h_x[i] = 1.0f;
    for (int i = 0; i < OC*IC*KH*KW; ++i)  h_W[i] = 1.0f;
    float *d_x, *d_W, *d_y;
    cudaMalloc(&d_x, sizeof(h_x));
    cudaMalloc(&d_W, sizeof(h_W));
    cudaMalloc(&d_y, sizeof(h_y));
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, sizeof(h_W), cudaMemcpyHostToDevice);
    dim3 block(8, 8);
    dim3 grid((OW+7)/8, (OH+7)/8, OC);
    conv2d_naive<<<grid, block>>>(d_x, d_W, d_y);
    cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < OC*OH*OW; ++i) sum += h_y[i];
    // Each output is 9, output count is OH*OW = 9, total = 81.
    printf("sum=%.0f\n", sum);
    cudaFree(d_x); cudaFree(d_W); cudaFree(d_y);
    return 0;
}
