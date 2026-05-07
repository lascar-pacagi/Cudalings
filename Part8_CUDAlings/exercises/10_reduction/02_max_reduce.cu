// CUDAlings 10.02 — Tree reduction with a different operator (max)
//
// Reductions aren't sum-only. The same tree pattern works for any
// associative + commutative operator. Today: per-block max.
//
// Goal: find the global max of the input. Initialize tile with -inf for
// out-of-range threads so they don't affect the max.

// I AM NOT DONE

#include <cstdio>
#include <cfloat>
#include <cuda_runtime.h>

#define N 4096

__global__ void block_max(const float* in, float* out) {
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    tile[tid] = (gid < N) ? in[gid] : -FLT_MAX;
    __syncthreads();
    // TODO: tree reduction with max instead of +
    if (tid == 0) out[blockIdx.x] = tile[0];
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = (float)i;     // max is N-1 = 4095
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, (N/256)*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    block_max<<<N/256, 256, 256*sizeof(float)>>>(d_in, d_out);
    float *hout = new float[N/256];
    cudaMemcpy(hout, d_out, (N/256)*sizeof(float), cudaMemcpyDeviceToHost);
    float global_max = -FLT_MAX;
    for (int i = 0; i < N/256; ++i) if (hout[i] > global_max) global_max = hout[i];
    printf("max=%.0f\n", global_max);
    cudaFree(d_in); cudaFree(d_out); delete[] h; delete[] hout;
    return 0;
}
