#include <cstdio>
#include <cuda_runtime.h>
int main() {
    void* d = nullptr;
    cudaError_t e1 = cudaMalloc(&d, 1 << 20);
    cudaError_t e2 = cudaFree(d);
    if (e1 == cudaSuccess && e2 == cudaSuccess) printf("alloc ok\n");
    else printf("FAIL\n");
    return 0;
}
