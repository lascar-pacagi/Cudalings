// CUDAlings 02.03 — __constant__ memory
//
// Read-only data that every thread reads identically (e.g. small filter
// coefficients) goes in __constant__ memory: cached, broadcast to whole warp
// in 1 cycle when all threads access the same address.
//
// Goal: declare a __constant__ array `c_coeffs[5]`, copy host data into it
// with cudaMemcpyToSymbol, and have the kernel use it.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024

// TODO: declare a constant-memory array of 5 floats called `c_coeffs`

__global__ void apply(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // TODO: write c_coeffs[i % 5] into x[i]
    }
}

int main() {
    float h_coeffs[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    // TODO: copy h_coeffs into the constant-memory symbol

    float *d; cudaMalloc(&d, N*sizeof(float));
    apply<<<4, 256>>>(d);

    float *h = new float[N];
    cudaMemcpy(h, d, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    // sum = (1+2+3+4+5) * (N/5) = 15 * 204 + (1+2+3+4) [for N % 5 = 4]
    // N=1024, 1024 = 5*204 + 4. Sum = 15*204 + (1+2+3+4) = 3060 + 10 = 3070
    printf("sum=%.1f\n", sum);

    cudaFree(d); delete[] h;
    return 0;
}
