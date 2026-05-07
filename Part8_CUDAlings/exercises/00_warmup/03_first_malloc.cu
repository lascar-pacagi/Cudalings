// CUDAlings 00.03 — cudaMalloc only (no kernel yet)
//
// Allocate 1 MB of device memory, then immediately free it. No kernel,
// no memcpy. The point is to internalize the cudaMalloc/cudaFree pair
// and confirm your CUDA driver is alive.
//
// Validator: program prints "alloc ok" if both calls return cudaSuccess.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    void* d = nullptr;
    size_t bytes = 1 << 20;     // 1 MB
    cudaError_t e1 = cudaSuccess;
    cudaError_t e2 = cudaSuccess;

    // TODO: allocate `bytes` on the device into `d`; capture the rc into e1
    // TODO: free the device allocation; capture the rc into e2

    if (e1 == cudaSuccess && e2 == cudaSuccess) printf("alloc ok\n");
    else printf("FAIL e1=%d e2=%d\n", (int)e1, (int)e2);
    return 0;
}
