// CUDAlings 13.03 — Broadcasting via stride=0
//
// Broadcasting is "free" if you store stride=0 along broadcast dims.
// For (B, T, E) + (E,) bias:
//   bias.shape  = (E,)
//   bias.stride = (0, 0, 1)    <- treat as if it had shape (B, T, E)
//   out[b,t,e] = x[b,t,e] + bias[0*0 + 0*0 + e*1] = x[b,t,e] + bias[e]
//
// No memory copy, no kernel rewrite. The same kernel that does plain
// elementwise add handles broadcast add as long as it computes offsets
// with the right strides.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define B 2
#define T 3
#define E 4

__global__ void broadcast_add(const float* x, const float* bias, float* y,
                              int sb_x, int st_x, int se_x,
                              int sb_b, int st_b, int se_b) {
    int e = threadIdx.x;
    int t = blockIdx.x;
    int b = blockIdx.y;
    if (e >= E) return;
    // TODO: compute x_idx and b_idx via the strides
    // y[b * T*E + t*E + e] = x[x_idx] + bias[b_idx];
}

int main() {
    float h_x[B*T*E], h_b[E], h_y[B*T*E];
    for (int i = 0; i < B*T*E; ++i) h_x[i] = 1.0f;
    for (int i = 0; i < E;     ++i) h_b[i] = (float)(i + 1);    // [1, 2, 3, 4]

    float *d_x, *d_b, *d_y;
    cudaMalloc(&d_x, B*T*E*sizeof(float));
    cudaMalloc(&d_b, E*sizeof(float));
    cudaMalloc(&d_y, B*T*E*sizeof(float));
    cudaMemcpy(d_x, h_x, B*T*E*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, E*sizeof(float),     cudaMemcpyHostToDevice);

    // Strides: x is contiguous (T*E, E, 1); bias broadcasts (0, 0, 1).
    broadcast_add<<<dim3(T, B), E>>>(d_x, d_b, d_y, T*E, E, 1, 0, 0, 1);
    cudaMemcpy(h_y, d_y, B*T*E*sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < B*T*E; ++i) sum += h_y[i];
    // sum = (1+1) + (1+2) + (1+3) + (1+4) per (b,t) = 14, times B*T = 6 → 84
    printf("sum=%.0f\n", sum);
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_y);
    return 0;
}
