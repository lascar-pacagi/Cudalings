// CUDAlings 00.06 — Pass values to a kernel
//
// Kernel arguments are passed by value -- no need to wrap scalars in
// device buffers. The launcher copies the arg list into a small
// kernel-arg area before each call.
//
// Goal: write a kernel that takes (int* p, int a, int b) and stores a + b.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_into(int* p, int a, int b) {
    // TODO: *p = a + b;
}

int main() {
    int* d; cudaMalloc(&d, sizeof(int));
    // TODO: add_into<<<1, 1>>>(d, 7, 35);
    int h = 0;
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("v=%d\n", h);    // expected 42
    cudaFree(d);
    return 0;
}
