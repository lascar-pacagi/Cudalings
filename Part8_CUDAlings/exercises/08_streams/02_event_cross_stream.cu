// CUDAlings 08.02 — Synchronize one stream against another with events
//
// cudaStreamWaitEvent makes stream B block until an event recorded on
// stream A has fired. That's how you express "kernel B depends on
// kernel A" without serializing the whole device.
//
//   sA: kernel1 → eventRecord(e, sA)
//   sB:                                streamWaitEvent(sB, e) → kernel2

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 1024

__global__ void scale(float* x, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] *= a;
}

int main() {
    float *d_x; cudaMalloc(&d_x, N*sizeof(float));
    float h[N]; for (int i = 0; i < N; ++i) h[i] = 1.0f;
    cudaMemcpy(d_x, h, N*sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t sA, sB;
    cudaStreamCreate(&sA);
    cudaStreamCreate(&sB);
    cudaEvent_t e;
    cudaEventCreate(&e);

    scale<<<(N+255)/256, 256, 0, sA>>>(d_x, 2.0f);     // ×2 on stream A
    // TODO: cudaEventRecord(e, sA);
    // TODO: cudaStreamWaitEvent(sB, e, 0);
    scale<<<(N+255)/256, 256, 0, sB>>>(d_x, 3.0f);     // ×3 on stream B

    cudaDeviceSynchronize();
    cudaMemcpy(h, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    // 1 * 2 * 3 = 6 per element. Sum = N * 6 = 6144.
    printf("sum=%.0f\n", sum);
    cudaEventDestroy(e);
    cudaStreamDestroy(sA); cudaStreamDestroy(sB);
    cudaFree(d_x);
    return 0;
}
