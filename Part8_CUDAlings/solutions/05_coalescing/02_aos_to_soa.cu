#include <cstdio>
#include <cuda_runtime.h>
#define N 1024
struct P { float x, y, z; };
__global__ void to_soa(const P* aos, float* xs, float* ys, float* zs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { xs[i] = aos[i].x; ys[i] = aos[i].y; zs[i] = aos[i].z; }
}
int main() {
    P *h = new P[N];
    for (int i = 0; i < N; ++i) { h[i].x = 1; h[i].y = 2; h[i].z = 3; }
    P *d_aos; float *d_x, *d_y, *d_z;
    cudaMalloc(&d_aos, N*sizeof(P));
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMalloc(&d_z, N*sizeof(float));
    cudaMemcpy(d_aos, h, N*sizeof(P), cudaMemcpyHostToDevice);
    to_soa<<<(N+255)/256, 256>>>(d_aos, d_x, d_y, d_z);
    float *hx = new float[N], *hy = new float[N], *hz = new float[N];
    cudaMemcpy(hx, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hy, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hz, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < N; ++i) sum += hx[i] + hy[i] + hz[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d_aos); cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    delete[] h; delete[] hx; delete[] hy; delete[] hz;
    return 0;
}
