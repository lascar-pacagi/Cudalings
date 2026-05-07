// CUDAlings 10.03 — Block reduction = warp shuffle + per-warp reduce
//
// The fastest block reduction uses warp shuffles, not shared memory:
//   1. Each warp reduces to its lane 0 with __shfl_down_sync.
//   2. The (block_size / 32) warp results go through ONE more warp reduce.
// Two warp-reduces beat the log2(block_size) shared-memory tree.
//
// Goal: implement on a block of 256 threads.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 256

__inline__ __device__ float warp_sum(float v) {
    for (int d = 16; d > 0; d >>= 1)
        v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}

__global__ void block_sum(const float* in, float* out) {
    __shared__ float warp_results[8];     // N/32 = 8 warps
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid & 31;

    float v = in[tid];
    // TODO: v = warp_sum(v);
    // TODO: if (lane == 0) warp_results[warp_id] = v;
    __syncthreads();
    if (warp_id == 0) {
        float w = (lane < 8) ? warp_results[lane] : 0.0f;
        // TODO: w = warp_sum(w);
        if (lane == 0) *out = w;
    }
}

int main() {
    float h[N]; for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    block_sum<<<1, N>>>(d_in, d_out);
    float r = 0;
    cudaMemcpy(&r, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum=%.0f\n", r);  // 256
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
