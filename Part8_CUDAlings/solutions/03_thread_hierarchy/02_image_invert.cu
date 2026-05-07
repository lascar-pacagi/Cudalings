#include <cstdio>
#include <cuda_runtime.h>
#define H 100
#define W 200
__global__ void invert(const unsigned char* in, unsigned char* out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < W && r < H) {
        int i = r * W + c;
        out[i] = 255 - in[i];
    }
}
int main() {
    unsigned char *h_in = new unsigned char[H*W];
    for (int i = 0; i < H*W; ++i) h_in[i] = (unsigned char)(i % 256);
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in,  H*W);
    cudaMalloc(&d_out, H*W);
    cudaMemcpy(d_in, h_in, H*W, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((W + 15)/16, (H + 15)/16);
    invert<<<grid, block>>>(d_in, d_out);
    unsigned char *h_out = new unsigned char[H*W];
    cudaMemcpy(h_out, d_out, H*W, cudaMemcpyDeviceToHost);
    long sum = 0;
    for (int i = 0; i < H*W; ++i) sum += h_out[i];
    printf("sum=%ld\n", sum);
    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}
