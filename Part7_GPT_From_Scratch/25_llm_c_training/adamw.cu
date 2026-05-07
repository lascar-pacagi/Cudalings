// adamw.cu -- the AdamW optimizer step as a CUDA kernel.
//
// One thread per parameter element. The kernel is identical for every
// parameter buffer (W, b, gamma, beta, ...) -- the dispatcher loops over
// (param_ptr, grad_ptr, m_ptr, v_ptr) tuples and launches once each.

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>


__global__ void adamw_kernel(
    float* __restrict__ param,
    float* __restrict__ m,
    float* __restrict__ v,
    const float* __restrict__ grad,
    int n,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float beta1_correction, float beta2_correction)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi / beta1_correction;       // bias-corrected first moment
    float v_hat = vi / beta2_correction;       // bias-corrected second moment
    // decoupled weight decay: subtract wd * p separately, NOT inside the moment.
    param[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param[i]);
}


// Convenience wrapper -- keeps the launch config in one place.
void adamw_step(float* param, float* m, float* v, const float* grad, int n,
                float lr, float beta1, float beta2, float eps, float wd,
                int step_t)
{
    int block = 256;
    int grid  = (n + block - 1) / block;
    float beta1_correction = 1.0f - powf(beta1, (float)step_t);
    float beta2_correction = 1.0f - powf(beta2, (float)step_t);
    adamw_kernel<<<grid, block>>>(param, m, v, grad, n,
                                  lr, beta1, beta2, eps, wd,
                                  beta1_correction, beta2_correction);
}


// ---------------------------------------------------------------------------
// Tiny self-test: optimize a 1D parabola y = (p - 3)^2 to see the param
// march toward 3.0. Useful to confirm signs and bias correction are right
// before plugging into the full GPT trainer.
// ---------------------------------------------------------------------------
int main() {
    int n = 1;
    float h_p = 0.f, h_m = 0.f, h_v = 0.f, h_g = 0.f;
    float *d_p, *d_m, *d_v, *d_g;
    cudaMalloc(&d_p, sizeof(float));
    cudaMalloc(&d_m, sizeof(float));
    cudaMalloc(&d_v, sizeof(float));
    cudaMalloc(&d_g, sizeof(float));
    cudaMemcpy(d_p, &h_p, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &h_m, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, &h_v, sizeof(float), cudaMemcpyHostToDevice);
    for (int step = 1; step <= 1000; ++step) {
        cudaMemcpy(&h_p, d_p, sizeof(float), cudaMemcpyDeviceToHost);
        h_g = 2.0f * (h_p - 3.0f);                 // dL/dp for L = (p-3)^2
        cudaMemcpy(d_g, &h_g, sizeof(float), cudaMemcpyHostToDevice);
        adamw_step(d_p, d_m, d_v, d_g, n,
                   /*lr*/0.05f, /*b1*/0.9f, /*b2*/0.999f,
                   /*eps*/1e-8f, /*wd*/0.0f, step);
    }
    cudaMemcpy(&h_p, d_p, sizeof(float), cudaMemcpyDeviceToHost);
    printf("final p=%.4f (target 3.0)\n", h_p);
    cudaFree(d_p); cudaFree(d_m); cudaFree(d_v); cudaFree(d_g);
    return 0;
}
