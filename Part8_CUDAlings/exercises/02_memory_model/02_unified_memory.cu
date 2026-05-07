// CUDAlings 02.02 — Unified Memory
//
// cudaMallocManaged returns a pointer accessible from both host and device.
// On Pascal (CC 6.x) and later, page faults migrate pages on demand. No
// explicit cudaMemcpy — the runtime handles it.
//
// Goal: replace the manual host/device dance with a single managed alloc.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 4096

__global__ void scale(float* x, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = a * x[i];
}

int main() {
    float *p = nullptr;
    // TODO: allocate `p` with cudaMallocManaged for N floats.

    for (int i = 0; i < N; ++i) p[i] = 1.0f;

    scale<<<16, 256>>>(p, 3.0f);
    cudaDeviceSynchronize();   // <-- crucial before reading from host!

    double sum = 0;
    for (int i = 0; i < N; ++i) sum += p[i];
    printf("sum=%.1f\n", sum); // expected 12288.0

    // TODO: cudaFree(p)
    return 0;
}
