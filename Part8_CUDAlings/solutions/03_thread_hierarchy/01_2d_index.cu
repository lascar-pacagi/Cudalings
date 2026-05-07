#include <cstdio>
#include <cuda_runtime.h>
#define H 64
#define W 32
__global__ void fill(int* m) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < W && r < H) m[r * W + c] = r * W + c;
}
int main() {
    int *d; cudaMalloc(&d, H*W*sizeof(int));
    dim3 block(16, 16);
    dim3 grid((W + 15)/16, (H + 15)/16);
    fill<<<grid, block>>>(d);
    int *h = new int[H*W];
    cudaMemcpy(h, d, H*W*sizeof(int), cudaMemcpyDeviceToHost);
    long sum = 0;
    for (int i = 0; i < H*W; ++i) sum += h[i];
    printf("sum=%ld\n", sum);
    cudaFree(d); delete[] h;
    return 0;
}
