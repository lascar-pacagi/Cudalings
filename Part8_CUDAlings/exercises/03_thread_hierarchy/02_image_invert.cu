// CUDAlings 03.02 — Invert a grayscale image
//
// Goal: given an HxW image of unsigned char in [0, 255], produce
// out[r][c] = 255 - in[r][c]. Use a 2D launch.
//
// A real-world signal-processing kernel — and a chance to reason about
// guards: H and W are NOT multiples of blockDim, so the bounds check matters.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define H 100
#define W 200

__global__ void invert(const unsigned char* in, unsigned char* out) {
    // TODO: 2D index, bounds check, write 255 - in[idx]
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
    // For each i, in = i%256, out = 255 - i%256. H*W=20000.
    // i%256 cycles 0..255 (78 full cycles + remainder 32).
    // sum(0..255) = 32640. 78 * (256*255 - 32640) = 78 * 32640 = 2545920.
    // remainder for i in [78*256, 20000): 32 values 0..31, each 255 - i%256:
    //   sum = sum(255..224) = 32*255 - sum(0..31) = 8160 - 496 = 7664.
    // total = 2545920 + 7664 = 2553584
    printf("sum=%ld\n", sum);

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}
