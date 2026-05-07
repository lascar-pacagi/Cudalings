// CUDAlings 29.01 — Part 1 Milestone: a complete SAXPY benchmark
//
// Tie together Part 1 (Ch 01-04): kernel launch, memory model, thread
// hierarchy, error handling.
//
// Build a SAXPY (y = a*x + y) that:
//   1. Allocates host + device buffers, fills x and y on host.
//   2. Copies to device.
//   3. Launches the kernel with a grid-stride loop.
//   4. Times it with cudaEvent.
//   5. Checks every cuda call with CUDA_CHECK.
//   6. Copies back, computes the L2 error vs. reference, prints "ms=" + "err=".
//
// This is the program you'd actually write to benchmark a new GPU --
// every line is a Part 1 chapter put to use.

// I AM NOT DONE

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @ %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); std::exit(1);}} while(0)

#define N (1 << 20)

__global__ void saxpy(int n, float a, const float* x, float* y) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    // TODO: implement SAXPY (y = a*x + y) via a grid-stride loop
}

int main() {
    float *h_x = new float[N], *h_y = new float[N];
    for (int i = 0; i < N; ++i) { h_x[i] = 1.0f; h_y[i] = 1.0f; }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));

    saxpy<<<128, 256>>>(N, 2.0f, d_x, d_y);   // warmup
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(a));
    saxpy<<<128, 256>>>(N, 2.0f, d_x, d_y);    // y becomes 2*x+y = 2+1 = 3 ; then 2+3 = 5 after warmup
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));

    CUDA_CHECK(cudaMemcpy(h_y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));
    // After 2 saxpy calls (warmup + timed), each with a=2: y_new = 2*x + y_old.
    // y starts at 1; first call makes 2+1=3; second call makes 2+3=5.
    double err = 0;
    for (int i = 0; i < N; ++i) { double d = h_y[i] - 5.0; err += d*d; }
    err = std::sqrt(err / N);

    printf("ms=%.3f\n", ms);
    printf("err=%.6f\n", err);

    delete[] h_x; delete[] h_y;
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    return 0;
}
