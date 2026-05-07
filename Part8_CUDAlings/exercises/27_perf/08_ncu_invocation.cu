// CUDAlings 27.08 — A reference kernel for nsight-compute profiling
//
// This exercise is meta: it builds a deliberately memory-bound kernel
// (just elementwise scaling) and prints the ncu command you'd run on the
// resulting binary. The validator only checks that the printed command
// mentions "ncu" and the binary name.
//
// Try it manually after the runner passes:
//   ncu --set basic ./08_ncu_invocation
// You'll see Memory Throughput at ~70-80% of peak and Compute Throughput
// near 0%. Roofline-classify it as memory-bound, exactly as 27.06 predicted.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 24)

__global__ void scale(const float* in, float* out, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a * in[i];
}

int main() {
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemset(d_in, 0, N*sizeof(float));

    // run a few times so ncu has something to attribute
    for (int i = 0; i < 5; ++i) scale<<<(N+255)/256, 256>>>(d_in, d_out, 2.0f);
    cudaDeviceSynchronize();

    // TODO: printf("ncu --set basic ./08_ncu_invocation\n");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
