// CUDAlings 11.03 — Blelloch (work-efficient) exclusive scan
//
// Two phases on a power-of-2 array of length N (single block, in shared mem):
//   UP-SWEEP (reduce):
//     for offset in 1, 2, 4, ..., N/2:
//       buf[stride*(2*tid+2) - 1] += buf[stride*(2*tid+1) - 1]
//   Set buf[N-1] = 0 (zero the last element so this is *exclusive*).
//   DOWN-SWEEP:
//     for offset in N/2, N/4, ..., 1:
//       t = buf[left]; buf[left] = buf[right]; buf[right] += t
//
// Goal: input is all ones; the inclusive scan would be 1, 2, ..., N. The
// exclusive scan is 0, 1, ..., N-1. Validator checks the LAST output = N-1.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 32      // small to keep the code legible; use one block of 32 threads but only N/2 active

__global__ void blelloch_scan(const float* in, float* out) {
    __shared__ float buf[N];
    int tid = threadIdx.x;
    if (tid < N) buf[tid] = in[tid];
    __syncthreads();

    // UP-SWEEP
    int offset = 1;
    for (int d = N / 2; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int left  = offset * (2*tid + 1) - 1;
            int right = offset * (2*tid + 2) - 1;
            // TODO: buf[right] += buf[left];
        }
        offset *= 2;
    }
    if (tid == 0) buf[N - 1] = 0.0f;

    // DOWN-SWEEP
    for (int d = 1; d < N; d *= 2) {
        offset /= 2;
        __syncthreads();
        if (tid < d) {
            int left  = offset * (2*tid + 1) - 1;
            int right = offset * (2*tid + 2) - 1;
            // TODO: float t = buf[left]; buf[left] = buf[right]; buf[right] += t;
        }
    }
    __syncthreads();
    if (tid < N) out[tid] = buf[tid];
}

int main() {
    float h_in[N], h_out[N];
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    blelloch_scan<<<1, N>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("last=%.0f\n", h_out[N - 1]);    // expected N - 1 = 31
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
