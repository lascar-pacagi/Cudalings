// CUDAlings 08.01 — Overlapping H2D copy with kernel work via streams
//
// Default stream serializes everything. Two non-default streams can overlap:
// one stream's H2D copy can run while another stream's kernel computes,
// provided the host memory is pinned (cudaMallocHost).
//
// Goal: launch two kernels into two streams and synchronize on the device.
// We don't measure overlap here — that's a profiler exercise. We just
// want correctness with non-default streams.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N 4096

__global__ void scale(float* x, float a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) x[i] = a * x[i];
}

int main() {
    float *d1, *d2;
    cudaMalloc(&d1, N*sizeof(float));
    cudaMalloc(&d2, N*sizeof(float));

    cudaStream_t s1, s2;
    // TODO: cudaStreamCreate s1, s2

    cudaMemsetAsync(d1, 0, N*sizeof(float), /*stream*/ 0);
    cudaMemsetAsync(d2, 0, N*sizeof(float), /*stream*/ 0);
    // TODO: launch scale on s1 with a=2.0f, on s2 with a=3.0f
    // TODO: cudaDeviceSynchronize()

    float *h1 = new float[N], *h2 = new float[N];
    cudaMemcpy(h1, d1, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h2, d2, N*sizeof(float), cudaMemcpyDeviceToHost);

    // After scale*0, both buffers are still 0 — pass condition is rc==0.
    printf("done\n");

    // TODO: cudaStreamDestroy s1, s2
    cudaFree(d1); cudaFree(d2); delete[] h1; delete[] h2;
    return 0;
}
