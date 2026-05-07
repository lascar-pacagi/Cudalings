// CUDAlings 06.02 — Tiled matrix transpose with bank-conflict padding
//
// Naive transpose: out[c*N + r] = in[r*N + c]. Reads coalesce (consecutive
// threads read a row), but writes do NOT (consecutive threads write a column,
// stride N). Tiled transpose: each block stages a 32x32 tile in shared
// memory, then writes it to the output transposed.
//
// Bank conflicts: a 32x32 float tile has 32 banks, and column j hits bank
// (i*32 + j) % 32 = j — 32 threads writing column j all hit bank j, that's
// fine (broadcast). But 32 threads READING column j on the way out hit the
// same bank serially: 32-way conflict. The classic fix is to pad with a
// dummy column: declare `tile[32][33]`. Now column j of the logical tile
// lives in bank (i*33 + j) % 32, which spreads them across all banks.
//
// Goal: implement `transpose_tiled` with bank-conflict-free shared memory.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>
#define N 64

__global__ void transpose_tiled(const float* in, float* out) {
    // TODO: declare __shared__ float tile[32][33];
    // TODO: stage in[r * N + c] into tile[threadIdx.y][threadIdx.x]
    // TODO: __syncthreads()
    // TODO: write out[c' * N + r'] from tile[threadIdx.x][threadIdx.y]
    //   where c', r' are based on swapping block coordinates.
}

int main() {
    float *h = new float[N*N];
    for (int i = 0; i < N*N; ++i) h[i] = (float)i;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*N*sizeof(float));
    cudaMalloc(&d_out, N*N*sizeof(float));
    cudaMemcpy(d_in, h, N*N*sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(32, 32);
    dim3 grid(N/32, N/32);
    transpose_tiled<<<grid, block>>>(d_in, d_out);
    float *hout = new float[N*N];
    cudaMemcpy(hout, d_out, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Validate: trace of input == trace of output (diag preserved)
    double tr = 0;
    for (int i = 0; i < N; ++i) tr += hout[i*N + i];
    // Trace of input: i*N + i for i in [0,N). N=64. Sum = (N-1)*N/2 + N*(N-1)*N/2
    // = 64*63/2 + 64*64*63/2 = 2016 + 129024 = 131040... let's print and compare.
    // Easier: also check (sum of all out) == (sum of all in).
    double sum = 0;
    for (int i = 0; i < N*N; ++i) sum += hout[i];
    printf("sum=%.0f\n", sum);  // sum of 0..(N*N - 1) = 4095*4096/2 = 8386560
    cudaFree(d_in); cudaFree(d_out); delete[] h; delete[] hout;
    return 0;
}
