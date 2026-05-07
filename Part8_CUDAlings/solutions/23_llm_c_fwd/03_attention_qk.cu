#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#define T 4
#define Dh 2
__global__ void attn_qk(const float* q, const float* k, float* logits) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= T || i >= T) return;
    if (j > i) { logits[i * T + j] = -INFINITY; return; }
    float dot = 0.f;
    for (int d = 0; d < Dh; ++d) dot += q[i*Dh + d] * k[j*Dh + d];
    logits[i*T + j] = dot / sqrtf((float)Dh);
}
int main() {
    float h_q[T*Dh], h_k[T*Dh], h_l[T*T];
    for (int i = 0; i < T*Dh; ++i) { h_q[i] = 1.0f; h_k[i] = 1.0f; }
    float *d_q, *d_k, *d_l;
    cudaMalloc(&d_q, sizeof(h_q));
    cudaMalloc(&d_k, sizeof(h_k));
    cudaMalloc(&d_l, sizeof(h_l));
    cudaMemcpy(d_q, h_q, sizeof(h_q), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, sizeof(h_k), cudaMemcpyHostToDevice);
    dim3 block(8, 8);
    dim3 grid((T+7)/8, (T+7)/8);
    attn_qk<<<grid, block>>>(d_q, d_k, d_l);
    cudaMemcpy(h_l, d_l, sizeof(h_l), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < T; ++i)
        for (int j = 0; j <= i; ++j) sum += h_l[i * T + j];
    printf("sum=%.3f\n", sum);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_l);
    return 0;
}
