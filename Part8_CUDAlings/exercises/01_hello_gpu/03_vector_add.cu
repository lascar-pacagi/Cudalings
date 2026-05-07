// CUDAlings 01.03 — Vector add (small N, exact-fit launch)
//
// Goal: c[i] = a[i] + b[i] for N=1024. Use 1 block of 256 threads in a
// 1D grid: gridDim = ceil_div(N, blockDim).
//
// Validator: the program prints the sum of all c[i]. With a[i]=i and b[i]=2*i,
// the expected sum is 3 * (N-1) * N / 2 = 1572864.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024

__global__ void add(const float* a, const float* b, float* c) {
    // TODO: compute idx and write c[idx]
}

int main() {
    float *h_a = new float[N], *h_b = new float[N], *h_c = new float[N];
    for (int i = 0; i < N; ++i) { h_a[i] = i; h_b[i] = 2.0f * i; }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

    // TODO: launch `add` with enough threads so that every i in [0, N) is covered.
    // Hint: 1 block of 256 threads is too few. Use 4 blocks of 256.

    cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h_c[i];
    printf("sum=%.1f\n", sum);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    delete[] h_a; delete[] h_b; delete[] h_c;
    return 0;
}
