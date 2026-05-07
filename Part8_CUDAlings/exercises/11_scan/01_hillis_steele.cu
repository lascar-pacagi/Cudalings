// CUDAlings 11.01 — Hillis-Steele inclusive scan (one block)
//
// Inclusive scan: out[i] = in[0] + in[1] + ... + in[i].
// Hillis-Steele: at offset d=1,2,4,...,blockDim/2:
//   if (tid >= d) buf[tid] += buf[tid - d];
//   __syncthreads();
// O(N log N) work but only log N rounds — perfect for small (single-block) N.
//
// Goal: implement on N=128 ones; final out[127] should be 128.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 128

__global__ void scan(const float* in, float* out) {
    __shared__ float buf[N];
    int tid = threadIdx.x;
    buf[tid] = in[tid];
    __syncthreads();

    // TODO: implement the doubling-offset loop described above
    //        (read the neighbor before writing -- two syncs per round)
    out[tid] = buf[tid];
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    scan<<<1, N>>>(d_in, d_out);
    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("last=%.0f\n", h[N-1]);   // expected 128
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
