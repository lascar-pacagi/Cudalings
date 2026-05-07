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
    int xi = b * sb_x + t * st_x + e * se_x;
    int bi = b * sb_b + t * st_b + e * se_b;
    y[b * T * E + t * E + e] = x[xi] + bias[bi];
}
int main() {
    float h_x[B*T*E], h_b[E], h_y[B*T*E];
    for (int i = 0; i < B*T*E; ++i) h_x[i] = 1.0f;
    for (int i = 0; i < E;     ++i) h_b[i] = (float)(i + 1);
    float *d_x, *d_b, *d_y;
    cudaMalloc(&d_x, B*T*E*sizeof(float));
    cudaMalloc(&d_b, E*sizeof(float));
    cudaMalloc(&d_y, B*T*E*sizeof(float));
    cudaMemcpy(d_x, h_x, B*T*E*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, E*sizeof(float),     cudaMemcpyHostToDevice);
    broadcast_add<<<dim3(T, B), E>>>(d_x, d_b, d_y, T*E, E, 1, 0, 0, 1);
    cudaMemcpy(h_y, d_y, B*T*E*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < B*T*E; ++i) sum += h_y[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d_x); cudaFree(d_b); cudaFree(d_y);
    return 0;
}
