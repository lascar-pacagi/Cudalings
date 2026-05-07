// CUDAlings 01.04 — The grid-stride loop
//
// Goal: process N=10000 elements with a launch of just 64 blocks of 128
// threads (= 8192 threads). That's fewer threads than data items, so each
// thread must handle multiple elements via a grid-stride loop.
//
// The pattern:
//   for (int i = gid; i < N; i += stride)
// where stride = gridDim.x * blockDim.x. This is the "scalable" launch form
// that decouples problem size from launch config.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 10000

__global__ void doublify(float* x) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // TODO: write a for-loop that visits every i in [0, N) using gid, stride.
    // Each iteration should set x[i] = 2 * x[i].
}

int main() {
    float *h_x = new float[N];
    for (int i = 0; i < N; ++i) h_x[i] = 1.0f;
    float *d_x; cudaMalloc(&d_x, N*sizeof(float));
    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);

    doublify<<<64, 128>>>(d_x);

    cudaMemcpy(h_x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_x[i];
    printf("sum=%.1f\n", sum);   // expected: 20000.0
    cudaFree(d_x); delete[] h_x;
    return 0;
}
