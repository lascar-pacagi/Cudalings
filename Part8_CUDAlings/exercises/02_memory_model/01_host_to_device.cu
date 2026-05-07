// CUDAlings 02.01 — Host→device→host memory roundtrip
//
// Goal: allocate a device buffer, copy a host array in, run a no-op kernel,
// copy back, verify the data is unchanged. The point is to internalize the
// shape of every CUDA program: malloc → memcpy → launch → memcpy → free.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024

__global__ void noop(float* x) { (void)x; }

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = (float)i;

    float *d = nullptr;
    // TODO: cudaMalloc d for N floats
    // TODO: cudaMemcpy h → d
    noop<<<4, 256>>>(d);
    // TODO: cudaMemcpy d → h
    // TODO: cudaFree d

    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.1f\n", sum);    // 0+1+...+1023 = 523776
    delete[] h;
    return 0;
}
