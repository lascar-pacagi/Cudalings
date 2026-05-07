#include <cstdio>
#include <cuda_runtime.h>
#define N 8
__global__ void radix_pass_bit0(const int* in, int* out) {
    __shared__ int flags[N];
    __shared__ int falses;
    int tid = threadIdx.x;
    int v = in[tid];
    flags[tid] = (v & 1) ? 1 : 0;
    __syncthreads();
    if (tid == 0) {
        int cnt = 0;
        for (int i = 0; i < N; ++i) cnt += (flags[i] == 0);
        falses = cnt;
    }
    __syncthreads();
    int t_zero = 0, t_one = 0;
    for (int i = 0; i < tid; ++i) { if (flags[i] == 0) ++t_zero; else ++t_one; }
    int dest = (flags[tid] == 0) ? t_zero : (falses + t_one);
    out[dest] = v;
}
int main() {
    int h_in[N]  = {3, 8, 1, 4, 1, 5, 9, 2};
    int h_out[N] = {0};
    int *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(int));
    cudaMalloc(&d_out, N*sizeof(int));
    cudaMemcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice);
    radix_pass_bit0<<<1, N>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);
    bool seen_one = false, ok = true;
    for (int i = 0; i < N; ++i) {
        if ((h_out[i] & 1) == 0 && seen_one) { ok = false; break; }
        if ((h_out[i] & 1) == 1) seen_one = true;
    }
    printf("%s\n", ok ? "ok" : "FAIL");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
