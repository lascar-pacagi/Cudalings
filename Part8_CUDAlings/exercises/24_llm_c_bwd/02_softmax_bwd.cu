// CUDAlings 24.02 — Softmax backward kernel
//
// Forward:   p = softmax(z)
// Backward:  dx_i = p_i * (dy_i - sum_j p_j * dy_j)
//
// Goal: implement on a single row of length C=8. The reduction is small
// enough that thread 0 can do it inline; for production you'd use a warp
// shuffle reduction (chapter 09).

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

#define C 8

__global__ void softmax_bwd(const float* p, const float* dy, float* dx) {
    __shared__ float sdot;
    int i = threadIdx.x;
    if (i >= C) return;
    if (i == 0) {
        float s = 0;
        for (int j = 0; j < C; ++j) s += p[j] * dy[j];
        sdot = s;
    }
    __syncthreads();
    // TODO: write the per-element softmax-backward formula from the header into dx
}

int main() {
    // Probabilities sum to 1; choose p = [1/8] * 8 (uniform).
    // Choose dy = [1, 0, 0, 0, 0, 0, 0, 0]. Then sum_j p_j*dy_j = 1/8.
    // dx_0 = (1/8) * (1 - 1/8) = 7/64,  dx_j>0 = (1/8)*(0 - 1/8) = -1/64
    // sum dx = 7/64 - 7/64 = 0.
    float h_p[C], h_dy[C], h_dx[C];
    for (int i = 0; i < C; ++i) { h_p[i] = 1.0f/C; h_dy[i] = (i == 0) ? 1.0f : 0.0f; }

    float *d_p, *d_dy, *d_dx;
    cudaMalloc(&d_p,  C*sizeof(float));
    cudaMalloc(&d_dy, C*sizeof(float));
    cudaMalloc(&d_dx, C*sizeof(float));
    cudaMemcpy(d_p,  h_p,  C*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dy, h_dy, C*sizeof(float), cudaMemcpyHostToDevice);
    softmax_bwd<<<1, C>>>(d_p, d_dy, d_dx);
    cudaMemcpy(h_dx, d_dx, C*sizeof(float), cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < C; ++i) sum += h_dx[i];
    printf("sum=%.4f\n", sum);     // expected 0.0
    cudaFree(d_p); cudaFree(d_dy); cudaFree(d_dx);
    return 0;
}
