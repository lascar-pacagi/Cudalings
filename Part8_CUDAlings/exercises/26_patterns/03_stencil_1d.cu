// CUDAlings 26.03 — 1D stencil with shared-memory halo
//
// y[i] = (x[i-1] + x[i] + x[i+1]) / 3            (radius=1, 3-point average)
//
// Naive: each thread reads 3 elements from global -> 3x bandwidth.
// Tiled: load BLOCK + 2 elements into shared mem (the +2 is the "halo"),
//         then read them locally.
//
// Goal: implement the stencil. With x = ones[N], y should also be ones
// in the interior, and we just sum y to N (boundaries excluded).

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 256
#define BLOCK 64

__global__ void stencil_3pt(const float* x, float* y, int n) {
    __shared__ float tile[BLOCK + 2];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load main element + halo on the edges.
    tile[tid + 1] = (gid < n) ? x[gid] : 0.f;
    if (tid == 0) {
        tile[0]         = (gid > 0)     ? x[gid - 1] : 0.f;
        tile[BLOCK + 1] = (gid + BLOCK < n) ? x[gid + BLOCK] : 0.f;
    }
    __syncthreads();

    // TODO: y[gid] = (tile[tid] + tile[tid + 1] + tile[tid + 2]) / 3.0f
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_x, *d_y;
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMemcpy(d_x, h, N*sizeof(float), cudaMemcpyHostToDevice);
    stencil_3pt<<<N / BLOCK, BLOCK>>>(d_x, d_y, N);
    cudaMemcpy(h, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Sum interior only (positions 1..N-2). Each is (1+1+1)/3 = 1.
    float s = 0;
    for (int i = 1; i < N - 1; ++i) s += h[i];
    printf("sum=%.0f\n", s);   // expected N - 2 = 254
    cudaFree(d_x); cudaFree(d_y); delete[] h;
    return 0;
}
