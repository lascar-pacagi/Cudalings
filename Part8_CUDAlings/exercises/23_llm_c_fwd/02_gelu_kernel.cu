// CUDAlings 23.02 -- GELU forward kernel (tanh approximation, GPT-2 style)
//
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Goal: implement the elementwise kernel, run it on x = -3, -2, -1, 0, 1, 2, 3,
// and print the sum of outputs. Expected ~3.0 (positive side dominates).

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>
#define N 7

__global__ void gelu(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = x[i];
    // TODO: write the tanh-approx GELU formula from the header back into x[i]
    x[i] = v;  // placeholder
}

int main() {
    float h[N] = {-3, -2, -1, 0, 1, 2, 3};
    float *d; cudaMalloc(&d, N*sizeof(float));
    cudaMemcpy(d, h, N*sizeof(float), cudaMemcpyHostToDevice);
    gelu<<<1, N>>>(d);
    cudaMemcpy(h, d, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.4f\n", sum);   // ~ -0.004 + -0.045 + -0.158 + 0 + 0.842 + 1.955 + 2.996 = ~5.586 - 0.207 = ~3.0
    cudaFree(d);
    return 0;
}
