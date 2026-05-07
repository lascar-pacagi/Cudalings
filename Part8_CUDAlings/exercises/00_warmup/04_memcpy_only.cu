// CUDAlings 00.04 — Round-trip a buffer through device memory (no kernel)
//
// Goal: copy host[N] = {10, 20, 30, 40} to device, then back to a fresh
// host buffer. Sum and print. Expected sum = 100.
//
// This is the third leg of the CUDA pyramid: malloc, memcpy, kernel.
// You're not running a kernel yet -- just proving the data path works.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int N = 4;
    int h_in[] = {10, 20, 30, 40};
    int h_out[4] = {0, 0, 0, 0};
    int *d = nullptr;

    // TODO: allocate `d` on the device for N ints
    // TODO: copy h_in to the device buffer
    // TODO: copy the device buffer back into h_out
    // TODO: free the device allocation

    int sum = 0;
    for (int i = 0; i < N; ++i) sum += h_out[i];
    printf("sum=%d\n", sum);    // expected 100
    return 0;
}
