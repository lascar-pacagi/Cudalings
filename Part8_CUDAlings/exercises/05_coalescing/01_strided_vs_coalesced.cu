// CUDAlings 05.01 — Coalesced vs strided global access
//
// Coalesced: thread t reads x[t]. The 32 threads in a warp issue ONE 128-byte
// memory transaction. Bandwidth-bound code stays at peak.
// Strided: thread t reads x[t * STRIDE]. With STRIDE=32, that's 32 separate
// transactions — 32x less bandwidth.
//
// Goal: implement `copy_coalesced` so that the OUTPUT (sum) is correct, then
// imagine running the strided version and observing the bandwidth drop in a
// profiler. We only validate correctness here; speed is a separate exercise.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 20)

__global__ void copy_coalesced(const float* in, float* out) {
    // TODO: coalesced read+write: out[gid] = in[gid].
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    copy_coalesced<<<(N+255)/256, 256>>>(d_in, d_out);
    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.0f\n", sum);   // expected 1048576
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
