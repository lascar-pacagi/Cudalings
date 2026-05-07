// CUDAlings 04.01 — The CUDA_CHECK macro
//
// Every cuda* call returns a cudaError_t. Most CUDA bugs go undiagnosed
// because the return value is ignored. The standard fix is a CUDA_CHECK
// macro that wraps every call.
//
// Goal: implement CUDA_CHECK below so that on failure it prints
// "CUDA error: <name> at <file>:<line>" and exits. Then run a deliberately
// failing memcpy and confirm the program terminates with a clear message.

// I AM NOT DONE

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        /* TODO: if _err != cudaSuccess, print and exit(1).                 \
           Use cudaGetErrorString(_err) and __FILE__ / __LINE__.            */ \
        (void)_err;                                                         \
    } while (0)

int main() {
    void* p;
    // 100 TB allocation -> guaranteed cudaErrorMemoryAllocation.
    cudaError_t e = cudaMalloc(&p, (size_t)100 << 40);
    CUDA_CHECK(e);  // should not be reached when CHECK works correctly
    printf("UNREACHABLE\n");
    return 0;
}
