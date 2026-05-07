// CUDAlings 24.01 -- A backward kernel: gradient of an elementwise op.
//
// As a stepping stone before tackling layernorm/attention backwards, do
// the simplest case first: gradient of sigmoid.
//   forward:  y = sigmoid(x) = 1 / (1 + exp(-x))
//   backward: dx = dy * y * (1 - y)
//
// We pass the saved y (forward output) so we don't recompute the sigmoid.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>
#define N 4

__global__ void sigmoid_bwd(const float* y, const float* dy, float* dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // TODO: write the local sigmoid derivative * dy into dx (uses the saved y)
}

int main() {
    // y = 0.5 (i.e. sigmoid(0)) for all entries; dy = 1.
    // Expected dx = 1 * 0.5 * 0.5 = 0.25 each. Sum = 1.0.
    float h_y[N] = {0.5, 0.5, 0.5, 0.5};
    float h_dy[N] = {1, 1, 1, 1};
    float h_dx[N] = {0};
    float *d_y, *d_dy, *d_dx;
    cudaMalloc(&d_y,  N*sizeof(float));
    cudaMalloc(&d_dy, N*sizeof(float));
    cudaMalloc(&d_dx, N*sizeof(float));
    cudaMemcpy(d_y,  h_y,  N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, h_dy, N*sizeof(float), cudaMemcpyHostToDevice);
    sigmoid_bwd<<<1, N>>>(d_y, d_dy, d_dx);
    cudaMemcpy(h_dx, d_dx, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_dx[i];
    printf("sum=%.4f\n", sum);
    cudaFree(d_y); cudaFree(d_dy); cudaFree(d_dx);
    return 0;
}
