#include <cstdio>
#include <cuda_runtime.h>
#define T 4
#define Dh 2
__global__ void attn_av(const float* att, const float* v, float* out) {
    int d = threadIdx.x;
    int i = blockIdx.x;
    if (d >= Dh) return;
    float acc = 0.f;
    for (int j = 0; j < T; ++j) acc += att[i*T + j] * v[j*Dh + d];
    out[i*Dh + d] = acc;
}
int main() {
    float h_att[T*T], h_v[T*Dh], h_out[T*Dh];
    for (int i = 0; i < T; ++i)
        for (int j = 0; j < T; ++j)
            h_att[i*T + j] = (j <= i) ? 1.0f / (i + 1) : 0.0f;
    for (int i = 0; i < T*Dh; ++i) h_v[i] = 1.0f;
    float *d_att, *d_v, *d_out;
    cudaMalloc(&d_att, sizeof(h_att));
    cudaMalloc(&d_v,   sizeof(h_v));
    cudaMalloc(&d_out, sizeof(h_out));
    cudaMemcpy(d_att, h_att, sizeof(h_att), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v,   h_v,   sizeof(h_v),   cudaMemcpyHostToDevice);
    attn_av<<<T, Dh>>>(d_att, d_v, d_out);
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < T*Dh; ++i) sum += h_out[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d_att); cudaFree(d_v); cudaFree(d_out);
    return 0;
}
