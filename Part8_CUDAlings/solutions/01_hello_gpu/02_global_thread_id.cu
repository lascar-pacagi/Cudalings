#include <cstdio>
#include <cuda_runtime.h>

__global__ void show_gid() {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("gid=%d\n", gid);
}

int main() {
    show_gid<<<4, 8>>>();
    cudaDeviceSynchronize();
    return 0;
}
