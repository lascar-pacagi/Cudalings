// CUDAlings 01.05 — SAXPY: the canonical BLAS-1 kernel
//
// y[i] = a * x[i] + y[i]    for i in [0, N), a is a scalar.
//
// SAXPY is the "hello world" of high-performance computing because every
// element is independent and memory-bound — the perfect GPU shape.
//
// Goal: implement saxpy as a __global__ function that uses a grid-stride
// loop, then launch it with N=1<<20, threads=256, blocks=128. The driver
// prints the L2 norm; expected ~ 2048.0 (computed analytically below).

// I AM NOT DONE

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 20)

__global__ void saxpy(int n, float a, const float* x, float* y) {
    // TODO: implement the formula above using a grid-stride loop
}

int main() {
    float *h_x = new float[N], *h_y = new float[N];
    for (int i = 0; i < N; ++i) { h_x[i] = 1.0f; h_y[i] = 1.0f; }

    float *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);

    saxpy<<<128, 256>>>(N, 1.0f, d_x, d_y);  // expected y[i] = 2.0 for all i

    cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sumsq = 0;
    for (int i = 0; i < N; ++i) sumsq += (double)h_y[i] * h_y[i];
    // ||y||_2 = sqrt(N * 4) = 2 * sqrt(N) = 2048
    printf("l2norm=%.1f\n", std::sqrt(sumsq));

    cudaFree(d_x); cudaFree(d_y);
    delete[] h_x; delete[] h_y;
    return 0;
}
