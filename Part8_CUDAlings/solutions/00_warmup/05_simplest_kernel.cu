#include <cstdio>
#include <cuda_runtime.h>
__global__ void set_42(int* p) { *p = 42; }
int main() {
    int* d; cudaMalloc(&d, sizeof(int));
    set_42<<<1, 1>>>(d);
    int h = 0;
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("v=%d\n", h);
    cudaFree(d);
    return 0;
}
