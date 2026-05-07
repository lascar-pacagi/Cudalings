// CUDAlings 14.05 — Conv2d with per-output-channel bias
//
// Same as 14.04 but each output channel has a learned bias added to every
// position:  y[oc, h, w] += bias[oc]
//
// Goal: 2 output channels (OC=2), 1 input channel, 5x5 input ones,
// 3x3 weights ones, bias=[1.0, 2.0]. Each output element of channel 0
// is 9+1=10; channel 1 is 9+2=11. Output count per channel = 9. Total
// sum = 9*10 + 9*11 = 189.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define IC 1
#define OC 2
#define IH 5
#define IW 5
#define KH 3
#define KW 3
#define OH (IH - KH + 1)
#define OW (IW - KW + 1)

__global__ void conv2d_bias(const float* x, const float* W, const float* b, float* y) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    if (ow >= OW || oh >= OH) return;
    float acc = 0.f;
    for (int ic = 0; ic < IC; ++ic)
        for (int kh = 0; kh < KH; ++kh)
            for (int kw = 0; kw < KW; ++kw)
                acc += x[ic*IH*IW + (oh+kh)*IW + (ow+kw)]
                     * W[oc*IC*KH*KW + ic*KH*KW + kh*KW + kw];
    // TODO: add b[oc]
    y[oc*OH*OW + oh*OW + ow] = acc;
}

int main() {
    float h_x[IC*IH*IW], h_W[OC*IC*KH*KW], h_b[OC] = {1.0f, 2.0f}, h_y[OC*OH*OW];
    for (int i = 0; i < IC*IH*IW; ++i)    h_x[i] = 1.0f;
    for (int i = 0; i < OC*IC*KH*KW; ++i) h_W[i] = 1.0f;
    float *d_x, *d_W, *d_b, *d_y;
    cudaMalloc(&d_x, sizeof(h_x));
    cudaMalloc(&d_W, sizeof(h_W));
    cudaMalloc(&d_b, sizeof(h_b));
    cudaMalloc(&d_y, sizeof(h_y));
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, sizeof(h_W), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice);
    dim3 block(8, 8);
    dim3 grid((OW+7)/8, (OH+7)/8, OC);
    conv2d_bias<<<grid, block>>>(d_x, d_W, d_b, d_y);
    cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < OC*OH*OW; ++i) sum += h_y[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d_x); cudaFree(d_W); cudaFree(d_b); cudaFree(d_y);
    return 0;
}
