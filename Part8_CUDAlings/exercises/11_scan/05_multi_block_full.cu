// CUDAlings 11.05 — Full multi-block exclusive scan (3 kernels)
//
// 1. block_scan: each block computes inclusive scan of its chunk in shared
//    memory; writes the chunk total to block_sums[blockIdx.x].
// 2. (host-side or 1 block) inclusive-scan block_sums.
// 3. add_offset: block b adds block_sums[b-1] to every element of its chunk
//    (or 0 for block 0).
//
// We do all three on a (BLOCKS * BS) input. With input = ones[N], the final
// output is INCLUSIVE [1..N]. The validator checks the last element = N.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define BS 64
#define BLOCKS 4
#define N (BS * BLOCKS)

__global__ void block_scan(const float* in, float* out, float* block_sums) {
    __shared__ float buf[BS];
    int tid = threadIdx.x;
    int gid = blockIdx.x * BS + tid;
    buf[tid] = in[gid];
    __syncthreads();
    for (int d = 1; d < BS; d <<= 1) {
        float read = (tid >= d) ? buf[tid - d] : 0.0f;
        __syncthreads();
        buf[tid] += read;
        __syncthreads();
    }
    out[gid] = buf[tid];
    if (tid == BS - 1) block_sums[blockIdx.x] = buf[BS - 1];
}

__global__ void scan_block_sums(float* sums, int n) {
    // tiny inclusive scan with one thread (n is at most BLOCKS=4 here)
    int tid = threadIdx.x;
    if (tid != 0) return;
    for (int i = 1; i < n; ++i) sums[i] += sums[i - 1];
}

__global__ void add_offset(float* out, const float* block_sums) {
    int b = blockIdx.x;
    if (b == 0) return;
    int tid = threadIdx.x;
    int gid = b * BS + tid;
    // TODO: add this block's offset (the previous block's prefix sum) to out[gid]
}

int main() {
    float h[N]; for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out, *d_sums;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMalloc(&d_sums, BLOCKS*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);

    block_scan<<<BLOCKS, BS>>>(d_in, d_out, d_sums);
    scan_block_sums<<<1, 1>>>(d_sums, BLOCKS);
    add_offset<<<BLOCKS, BS>>>(d_out, d_sums);

    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("last=%.0f\n", h[N - 1]);    // expected N = 256
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_sums);
    return 0;
}
