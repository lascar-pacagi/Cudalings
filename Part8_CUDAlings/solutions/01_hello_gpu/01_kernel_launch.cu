#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello() {
    printf("hello from thread %d\n", threadIdx.x);
}

int main() {
    hello<<<1, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}
