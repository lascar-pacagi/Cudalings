// CUDAlings 29.03 — Part 3 Milestone: running maximum (max-scan)
//
// Tie together Part 3 (Ch 10-12): reduction, scan, matmul tiling.
//
// Compute the cumulative maximum of an array:
//   out[i] = max(in[0], in[1], ..., in[i])
//
// This is a SCAN (Ch 11) but with `max` instead of `+` (Ch 10's lesson:
// the same tree pattern works for any associative operator). We do single
// block for clarity.
//
// Goal: implement Hillis-Steele scan over `max`.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 32

__global__ void max_scan(const float* in, float* out) {
    __shared__ float buf[N];
    int tid = threadIdx.x;
    if (tid >= N) return;
    buf[tid] = in[tid];
    __syncthreads();
    for (int d = 1; d < N; d <<= 1) {
        float other = (tid >= d) ? buf[tid - d] : -1e30f;
        __syncthreads();
        // TODO: if (other > buf[tid]) buf[tid] = other;
        __syncthreads();
    }
    out[tid] = buf[tid];
}

int main() {
    // Input: 1, 5, 3, 8, 2, 7, ... (alternating up/down)
    float h_in[N];
    for (int i = 0; i < N; ++i) h_in[i] = (i % 4 == 1) ? (float)i + 5 : (float)i;

    float h_out[N];
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    max_scan<<<1, N>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Verify monotonic: out[i] >= out[i-1]
    bool ok = true;
    for (int i = 1; i < N; ++i) if (h_out[i] < h_out[i-1]) { ok = false; break; }
    printf("%s last=%.0f\n", ok ? "ok" : "FAIL", h_out[N-1]);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
