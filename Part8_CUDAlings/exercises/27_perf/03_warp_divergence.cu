// CUDAlings 27.03 — Warp divergence: uniform vs checkerboard branch
//
// When threads in a warp take different branches, the SM serializes them.
// Two kernels do the same total work; one diverges, one doesn't.
//   uniform:    all threads in a warp go through the SAME branch
//                (decided per-warp via tid / 32)
//   checker:    threads alternate (tid % 2) -- maximum divergence
//
// We just measure both with cudaEvents and print the ratio. The diverging
// kernel should be ~2x slower on Pascal.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 22)

__global__ void uniform_branch(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int warp = i / 32;
    if (warp & 1) x[i] = sinf(x[i]); else x[i] = cosf(x[i]);
}

__global__ void checker_branch(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (i & 1) x[i] = sinf(x[i]); else x[i] = cosf(x[i]);
}

float time_kernel(void (*f)(float*), float* d) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    f<<<(N+255)/256, 256>>>(d);     // warm up
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < 10; ++i) f<<<(N+255)/256, 256>>>(d);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / 10.0f;
}

int main() {
    float* d; cudaMalloc(&d, N*sizeof(float));
    cudaMemset(d, 0, N*sizeof(float));

    float t_uniform = 0.f, t_checker = 0.f;
    // TODO: time both kernels via the helper and store the per-launch ms

    // We don't print the ratio (which depends on hw), just the fact that
    // both finished. Validator: stdout contains both labels.
    printf("uniform_ms=%.3f\n", t_uniform);
    printf("checker_ms=%.3f\n", t_checker);
    cudaFree(d);
    return 0;
}
