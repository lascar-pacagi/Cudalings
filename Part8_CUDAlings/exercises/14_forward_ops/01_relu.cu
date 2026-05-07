// CUDAlings 14.01 — ReLU forward
//
// y = max(0, x), elementwise. Trivial kernel — but it's the first activation
// of every conv net, so worth committing to muscle memory.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024

__global__ void relu(const float* x, float* y) {
    // TODO: i = global tid; bounds; y[i] = (x[i] > 0) ? x[i] : 0.0f
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = (float)(i - 512);  // half negative
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    relu<<<(N+255)/256, 256>>>(d_in, d_out);
    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    // Sum of relu(i-512) for i=0..1023 = sum_{i=512..1023} (i-512) = 0+1+...+511 = 130816
    printf("sum=%.0f\n", sum);
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
