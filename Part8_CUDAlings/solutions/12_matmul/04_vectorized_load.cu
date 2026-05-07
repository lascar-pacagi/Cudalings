#include <cstdio>
#include <cuda_runtime.h>
#define N (1 << 16)
__global__ void copy_vec4(const float4* in, float4* out, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) out[i] = in[i];
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
    printf("sum=%.0f\n", sum);
    cudaFree(d_in); cudaFree(d_out); delete[] h;
    return 0;
}
