// CUDAlings 25.01 -- AdamW step kernel
//
// Per-element:
//   m = b1*m + (1-b1)*g
//   v = b2*v + (1-b2)*g^2
//   m_hat = m / (1 - b1^t)
//   v_hat = v / (1 - b2^t)
//   p = p - lr * (m_hat / (sqrt(v_hat) + eps) + wd*p)
//
// Goal: optimize p toward 3.0 by taking 1000 steps with grad = 2*(p-3).

// I AM NOT DONE

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void adamw(float* p, float* m, float* v, float g,
                      int n, float lr, float b1, float b2, float eps, float wd,
                      float bc1, float bc2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // TODO: implement the formulas above for p[i], m[i], v[i] using grad g.
}

int main() {
    int n = 1;
    float h_p = 0.f, h_m = 0.f, h_v = 0.f;
    float *d_p, *d_m, *d_v;
    cudaMalloc(&d_p, sizeof(float));
    cudaMalloc(&d_m, sizeof(float));
    cudaMalloc(&d_v, sizeof(float));
    cudaMemcpy(d_p, &h_p, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, &h_m, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, &h_v, sizeof(float), cudaMemcpyHostToDevice);

    float b1 = 0.9f, b2 = 0.999f;
    for (int t = 1; t <= 1000; ++t) {
        cudaMemcpy(&h_p, d_p, sizeof(float), cudaMemcpyDeviceToHost);
        float g = 2.0f * (h_p - 3.0f);
        adamw<<<1, 1>>>(d_p, d_m, d_v, g, n, /*lr*/0.05f, b1, b2,
                        /*eps*/1e-8f, /*wd*/0.0f,
                        1.f - powf(b1, (float)t), 1.f - powf(b2, (float)t));
    }
    cudaMemcpy(&h_p, d_p, sizeof(float), cudaMemcpyDeviceToHost);
    printf("p=%.4f\n", h_p);    // expected ~3.0
    cudaFree(d_p); cudaFree(d_m); cudaFree(d_v);
    return 0;
}
