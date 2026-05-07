// CUDAlings 11.02 — Multi-block scan via block-local scan + sum offset
//
// For N too big for one block we do scan in two passes:
//   Pass A: each block scans its chunk locally, writes the chunk sum to
//           a per-block array.
//   Pass B: do an exclusive scan on the per-block sums (small array),
//           and have each block add its offset to its local scan.
// We do pass A only here -- the offset addition is a follow-up exercise.
//
// Goal: implement Hillis-Steele scan inside each block on a (blocks * BS)
// input. The validator checks the LAST element of block 0 (= BS, since
// each input is 1).

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define BS 64
#define BLOCKS 4
#define N (BS * BLOCKS)

__global__ void scan_per_block(const float* in, float* out) {
    __shared__ float buf[BS];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BS + tid;
    buf[tid] = in[gid];
    __syncthreads();

    // TODO: same Hillis-Steele scan as 11.01 over `buf`.

    out[gid] = buf[tid];
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    scan_per_block<<<BLOCKS, BS>>>(d_in, d_out);
    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("blk0_last=%.0f\n", h[BS - 1]);    // expected BS = 64
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
