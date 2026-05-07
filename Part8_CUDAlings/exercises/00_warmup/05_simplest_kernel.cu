// CUDAlings 00.05 — The simplest possible kernel
//
// One thread, one int written to a device location, copied back, printed.
// No threadIdx, no blockIdx -- a kernel running with <<<1, 1>>>.
//
// Goal: write a kernel that sets *p = 42, then launch it once.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void set_42(int* p) {
    // TODO: *p = 42;
}

int main() {
    int* d;
    cudaMalloc(&d, sizeof(int));
    // TODO: launch set_42<<<1, 1>>>(d);
    int h = 0;
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("v=%d\n", h);     // expected 42
    cudaFree(d);
    return 0;
}
