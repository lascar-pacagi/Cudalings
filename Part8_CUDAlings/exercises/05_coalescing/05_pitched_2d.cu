// CUDAlings 05.05 — cudaMallocPitch for naturally-aligned 2D access
//
// cudaMallocPitch(&p, &pitch, W*sizeof(float), H) allocates a 2D buffer
// where each row is padded to a multiple of 128 bytes. The driver
// returns `pitch` (in bytes!) which you must use for indexing:
//
//   float* row_ptr = (float*)((char*)p + r * pitch);
//   row_ptr[c] = ...;
//
// This guarantees that `row_ptr[0]` is 128-byte-aligned regardless of W.
// Critical for performance when W is not a multiple of 32.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define H 100
#define W 100        // not a multiple of 32!

__global__ void fill_pitched(float* base, size_t pitch_bytes, int h, int w) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= h || c >= w) return;
    // TODO: index into the pitched row (note: pitch is in BYTES, not elements)
    //       and write 1.0f at column `c`.
}

int main() {
    float* d = nullptr;
    size_t pitch = 0;
    cudaMallocPitch(&d, &pitch, W * sizeof(float), H);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    fill_pitched<<<grid, block>>>(d, pitch, H, W);

    // Pull each row separately using cudaMemcpy2D (handles the pitch).
    float* h_buf = new float[H * W];
    cudaMemcpy2D(h_buf, W * sizeof(float),
                 d, pitch,
                 W * sizeof(float), H,
                 cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < H * W; ++i) sum += h_buf[i];
    printf("sum=%.0f\n", sum);   // H*W = 10000
    cudaFree(d); delete[] h_buf;
    return 0;
}
