#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#define V 4
__global__ void softmax_xent_bwd(const float* logits, int target, float* dlogits, float scale) {
    __shared__ float buf[V];
    __shared__ float row_max, row_sum;
    int i = threadIdx.x;
    if (i >= V) return;
    buf[i] = logits[i];
    __syncthreads();
    if (i == 0) {
        float m = buf[0];
        for (int k = 1; k < V; ++k) if (buf[k] > m) m = buf[k];
        row_max = m;
    }
    __syncthreads();
    float e = expf(buf[i] - row_max);
    buf[i] = e;
    __syncthreads();
    if (i == 0) {
        float s = 0;
        for (int k = 0; k < V; ++k) s += buf[k];
        row_sum = s;
    }
    __syncthreads();
    float p = e / row_sum;
    float t = (i == target) ? 1.0f : 0.0f;
    dlogits[i] = (p - t) * scale;
}
int main() {
    float h_logits[V] = {1.0f, 2.0f, 3.0f, 4.0f};
    int target = 2;
    float h_dl[V] = {0};
    float *d_l, *d_dl;
    cudaMalloc(&d_l, V*sizeof(float));
    cudaMalloc(&d_dl, V*sizeof(float));
    cudaMemcpy(d_l, h_logits, V*sizeof(float), cudaMemcpyHostToDevice);
    softmax_xent_bwd<<<1, V>>>(d_l, target, d_dl, 1.0f);
    cudaMemcpy(h_dl, d_dl, V*sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < V; ++i) sum += h_dl[i];
    printf("sum=%.4f\n", sum);
    cudaFree(d_l); cudaFree(d_dl);
    return 0;
}
