// CUDAlings 06.04 — Dynamic shared memory (size set at launch)
//
// Static shared memory must be sized at compile time:
//     __shared__ float tile[256];
// Dynamic shared memory is sized at launch in the third <<<>>> argument:
//     kernel<<<grid, block, BLOCK*sizeof(float)>>>(...)
// and accessed via a single `extern __shared__` declaration:
//     extern __shared__ float tile[];
//
// You can only have ONE extern __shared__ declaration; if you need
// multiple "regions" you split a single buffer manually.
//
// Goal: write `block_sum_dyn` using dynamic shared memory sized at launch.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024

__global__ void block_sum_dyn(const float* in, float* out) {
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    // TODO: stage in[gid] into shared, sync, then have thread 0 emit the block sum
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, (N/256)*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    block_sum_dyn<<<N/256, 256, 256 * sizeof(float)>>>(d_in, d_out);
    float h_out[N/256];
    cudaMemcpy(h_out, d_out, (N/256)*sizeof(float), cudaMemcpyDeviceToHost);
    double total = 0;
    for (int i = 0; i < N/256; ++i) total += h_out[i];
    printf("total=%.0f\n", total);    // 1024
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
