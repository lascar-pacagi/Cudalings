#include <cstdio>
#include <cuda_runtime.h>
__global__ void noop() {}
int main() {
    noop<<<1, 100000>>>();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) printf("launch error: %s\n", cudaGetErrorString(e));
    return 0;
}
