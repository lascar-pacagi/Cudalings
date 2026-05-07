// CUDAlings 28.03 — BatchNorm2d forward (training mode)
//
// Per output channel c, across the (B, H, W) plane:
//   mean[c] = avg of x[:, c, :, :]
//   var[c]  = avg of (x[:, c, :, :] - mean[c])^2
//   y[b,c,h,w] = gamma[c] * (x[b,c,h,w] - mean[c]) / sqrt(var[c] + eps)
//             + beta[c]
//
// One block per channel; threads cooperate to compute mean/var via shared mem.
//
// Goal: with x = ones(2, 4, 3, 3), gamma = ones(4), beta = zeros(4):
//   mean = 1, var = 0 -> y = 0 everywhere. Sum = 0.

// I AM NOT DONE

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define B 2
#define C 4
#define H 3
#define W 3
#define BHW (B*H*W)

__global__ void bn_forward(const float* x, float* y,
                           const float* gamma, const float* beta) {
    __shared__ float sum, sumsq, mean, rstd;
    int c = blockIdx.x;
    int tid = threadIdx.x;
    if (tid == 0) { sum = 0.f; sumsq = 0.f; }
    __syncthreads();

    // Phase 1: each thread accumulates a slice; serialize via single thread for clarity.
    if (tid == 0) {
        for (int b = 0; b < B; ++b)
            for (int h = 0; h < H; ++h)
                for (int w = 0; w < W; ++w) {
                    float v = x[((b*C + c)*H + h)*W + w];
                    sum += v; sumsq += v*v;
                }
        mean = sum / BHW;
        float var = sumsq / BHW - mean * mean;
        rstd = rsqrtf(var + 1e-5f);
    }
    __syncthreads();

    // Phase 2: each thread writes some outputs.
    int n = B*H*W;
    for (int i = tid; i < n; i += blockDim.x) {
        int b = i / (H*W);
        int hw = i % (H*W);
        int h = hw / W;
        int w = hw % W;
        int idx = ((b*C + c)*H + h)*W + w;
        // TODO: y[idx] = gamma[c] * (x[idx] - mean) * rstd + beta[c];
    }
}

int main() {
    float h_x[B*C*H*W];
    for (int i = 0; i < B*C*H*W; ++i) h_x[i] = 1.0f;
    float h_g[C] = {1.f, 1.f, 1.f, 1.f};
    float h_b[C] = {0.f, 0.f, 0.f, 0.f};
    float h_y[B*C*H*W];

    float *d_x, *d_y, *d_g, *d_b;
    cudaMalloc(&d_x, sizeof(h_x));
    cudaMalloc(&d_y, sizeof(h_y));
    cudaMalloc(&d_g, sizeof(h_g));
    cudaMalloc(&d_b, sizeof(h_b));
    cudaMemcpy(d_x, h_x, sizeof(h_x), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_g, sizeof(h_g), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(h_b), cudaMemcpyHostToDevice);
    bn_forward<<<C, 32>>>(d_x, d_y, d_g, d_b);
    cudaMemcpy(h_y, d_y, sizeof(h_y), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < B*C*H*W; ++i) sum += h_y[i];
    printf("sum=%.4f\n", sum);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_g); cudaFree(d_b);
    return 0;
}
