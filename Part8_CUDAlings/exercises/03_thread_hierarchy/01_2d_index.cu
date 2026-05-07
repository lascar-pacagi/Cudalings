// CUDAlings 03.01 — 2D thread indexing on a matrix
//
// Goal: fill a row-major matrix M[H][W] with M[r][c] = r * W + c. Use a 2D
// grid of 2D blocks. The trick: dim3{x, y} maps to the column then row.
//
// Common bug: swapping row/col when computing the linear offset. We catch
// it with a checksum.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define H 64
#define W 32

__global__ void fill(int* m) {
    // TODO: compute c (column) from x dim, r (row) from y dim
    // TODO: bounds-check (c < W, r < H)
    // TODO: m[r * W + c] = r * W + c
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
    // sum of 0..(H*W - 1) = (H*W)*(H*W - 1)/2
    printf("sum=%ld\n", sum);  // 64*32 = 2048; 2048*2047/2 = 2096128
    cudaFree(d); delete[] h;
    return 0;
}
