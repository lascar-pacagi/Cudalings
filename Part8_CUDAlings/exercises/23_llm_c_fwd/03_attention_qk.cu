// CUDAlings 23.03 — Attention QK kernel: scores = Q @ K^T / sqrt(Dh) (causal mask)
//
// For T=4, Dh=2 (single batch, single head):
//   q (T, Dh), k (T, Dh)
//   logits (T, T): logits[i, j] = q[i] . k[j] / sqrt(Dh)  for j <= i, else -inf
//
// Goal: implement the kernel. We use q = k = ones, so non-masked logits
// equal Dh / sqrt(Dh) = sqrt(Dh). Sum of valid logits = T*(T+1)/2 * sqrt(Dh).
// With T=4, Dh=2: 10 * sqrt(2) ≈ 14.142.

// I AM NOT DONE

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define T 4
#define Dh 2

__global__ void attn_qk(const float* q, const float* k, float* logits) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;     // key position
    int i = blockIdx.y * blockDim.y + threadIdx.y;     // query position
    if (j >= T || i >= T) return;
    if (j > i) {
        logits[i * T + j] = -INFINITY;
        return;
    }
    float dot = 0.f;
    // TODO: dot product q[i] · k[j] across Dh, then write the scaled value into logits[i, j]
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
        for (int j = 0; j <= i; ++j)
            sum += h_l[i * T + j];
    printf("sum=%.3f\n", sum);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_l);
    return 0;
}
