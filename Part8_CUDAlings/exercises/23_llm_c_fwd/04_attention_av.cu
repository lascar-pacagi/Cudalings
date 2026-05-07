// CUDAlings 23.04 — Attention AV kernel: out = att @ v
//
// att (T, T): each row sums to 1 (already softmaxed).
// v (T, Dh).
// out (T, Dh): out[i, d] = sum_j att[i, j] * v[j, d]
//
// Goal: implement the kernel for T=4, Dh=2. With att = uniform 1/T below
// the diagonal (just for the test) and v = ones, each output row equals 1.
// Sum of output = T * Dh * 1 = 8.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define T 4
#define Dh 2

__global__ void attn_av(const float* att, const float* v, float* out) {
    int d = threadIdx.x;
    int i = blockIdx.x;
    if (d >= Dh) return;
    float acc = 0.f;
    // TODO: for j in 0..T: acc += att[i*T + j] * v[j*Dh + d]
    out[i*Dh + d] = acc;
}

int main() {
    float h_att[T*T], h_v[T*Dh], h_out[T*Dh];
    // att[i, j] = 1/(i+1) for j in [0..i], else 0. Each row sums to 1.
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
