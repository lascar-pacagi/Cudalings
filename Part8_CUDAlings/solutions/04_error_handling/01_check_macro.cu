#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _err = (call);                                         \
        if (_err != cudaSuccess) {                                         \
            std::printf("CUDA error: %s at %s:%d\n",                       \
                        cudaGetErrorString(_err), __FILE__, __LINE__);     \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

int main() {
    void* p;
    cudaError_t e = cudaMalloc(&p, (size_t)100 << 40);
    CUDA_CHECK(e);
    printf("UNREACHABLE\n");
    return 0;
}
