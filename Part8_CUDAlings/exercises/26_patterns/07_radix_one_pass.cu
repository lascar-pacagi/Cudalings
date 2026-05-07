// CUDAlings 26.07 — One pass of radix sort (bit 0 partition)
//
// Radix sort = repeated stable partition by one bit at a time. Per pass:
//   1. Compute flag[i] = (data[i] >> bit) & 1   // 0 → goes to head, 1 → tail
//   2. Compact 0s to the head, 1s to the tail.
//
// We do bit 0 only on N=8. After the pass, all evens (bit0=0) precede all
// odds (bit0=1), preserving relative order within each group.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 8

__global__ void radix_pass_bit0(const int* in, int* out) {
    __shared__ int flags[N];
    __shared__ int falses;
    int tid = threadIdx.x;
    int v = in[tid];
    flags[tid] = (v & 1) ? 1 : 0;     // 1 means "goes to tail"
    __syncthreads();

    if (tid == 0) {
        int cnt = 0;
        for (int i = 0; i < N; ++i) cnt += (flags[i] == 0);
        falses = cnt;
    }
    __syncthreads();

    // count of 0-flags before me (number of even values in [0, tid))
    int t_zero = 0;
    int t_one  = 0;
    for (int i = 0; i < tid; ++i) { if (flags[i] == 0) ++t_zero; else ++t_one; }

    int dest;
    // TODO: dest = (flags[tid] == 0) ? t_zero : (falses + t_one);
    // TODO: out[dest] = v;
}

int main() {
    int h_in[N]  = {3, 8, 1, 4, 1, 5, 9, 2};   // bit0: 1 0 1 0 1 1 1 0
    int h_out[N] = {0};
    int *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(int));
    cudaMalloc(&d_out, N*sizeof(int));
    cudaMemcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice);
    radix_pass_bit0<<<1, N>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);

    // Validate ordering: bit0=0 elements before bit0=1 elements.
    bool seen_one = false;
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if ((h_out[i] & 1) == 0 && seen_one) { ok = false; break; }
        if ((h_out[i] & 1) == 1) seen_one = true;
    }
    printf("%s\n", ok ? "ok" : "FAIL");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
