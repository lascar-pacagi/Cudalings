// CUDAlings 06.03 — Bank conflicts: a column-major access pattern
//
// Shared memory has 32 banks (32-bit each). Accesses by 32 threads to the
// same bank are serialized. A 32x32 float tile placed as `tile[32][32]`
// has column j living entirely in bank j -- a 32-way conflict when 32
// threads read column j. The classic fix: one extra column of padding,
// `tile[32][33]`, which spreads each logical column across all banks.
//
// Goal: implement a column-sum kernel using a padded shared tile so it's
// conflict-free. We don't measure perf here -- just verify correctness.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 32

__global__ void col_sum_padded(const float* in, float* out) {
    __shared__ float tile[N][N + 1];   // +1 padding to avoid 32-way conflicts
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Load
    // TODO: tile[row][col] = in[row * N + col];
    __syncthreads();
    // Each thread (col) sums down the column it owns.
    if (row == 0) {
        float s = 0;
        for (int r = 0; r < N; ++r) s += tile[r][col];
        out[col] = s;
    }
}

int main() {
    float h_in[N*N];
    for (int i = 0; i < N*N; ++i) h_in[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*N*sizeof(float), cudaMemcpyHostToDevice);
    col_sum_padded<<<1, dim3(N, N)>>>(d_in, d_out);
    float h_out[N];
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    double total = 0;
    for (int c = 0; c < N; ++c) total += h_out[c];
    printf("total=%.0f\n", total);   // expected N*N = 1024
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
