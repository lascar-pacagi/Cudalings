#include <cstdio>
#include <cuda_runtime.h>
__global__ void add_into(int* p, int a, int b) { *p = a + b; }
int main() {
    int* d; cudaMalloc(&d, sizeof(int));
    add_into<<<1, 1>>>(d, 7, 35);
    int h = 0;
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("v=%d\n", h);
    cudaFree(d);
    return 0;
}
