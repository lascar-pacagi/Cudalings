// CUDAlings 12.04 — Vectorized loads with float4
//
// A single 128-bit load (`float4`) issues fewer instructions than four
// 32-bit loads. When K is a multiple of 4 and pointers are 16-byte
// aligned, you can load four floats per instruction:
//
//   float4 a4 = *reinterpret_cast<const float4*>(&A[row*K + k]);
//
// Then unpack a4.x, a4.y, a4.z, a4.w. This typically halves the load
// instruction count for memory-bound copies.
//
// Goal: implement a vectorized COPY first (matmul vectorization is the
// next exercise's spirit). The validator checks the sum of the output.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define N (1 << 16)     // 65536, divisible by 4

__global__ void copy_vec4(const float4* in, float4* out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        // TODO: float4 v = in[i]; out[i] = v;
    }
}

int main() {
    float *h = new float[N];
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h, N*sizeof(float), cudaMemcpyHostToDevice);
    int n4 = N / 4;
    copy_vec4<<<(n4+127)/128, 128>>>(
        reinterpret_cast<const float4*>(d_in),
        reinterpret_cast<float4*>(d_out), n4);
    cudaMemcpy(h, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += h[i];
    printf("sum=%.0f\n", sum);    // expected 65536
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
