#include <cstdio>
#include <cuda_runtime.h>
#define H 100
#define W 100
__global__ void fill_pitched(float* base, size_t pitch_bytes, int h, int w) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= h || c >= w) return;
    char* row = (char*)base + r * pitch_bytes;
    reinterpret_cast<float*>(row)[c] = 1.0f;
}
int main() {
    float* d = nullptr;
    size_t pitch = 0;
    cudaMallocPitch(&d, &pitch, W * sizeof(float), H);
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    fill_pitched<<<grid, block>>>(d, pitch, H, W);
    float* h_buf = new float[H * W];
    cudaMemcpy2D(h_buf, W * sizeof(float), d, pitch,
                 W * sizeof(float), H, cudaMemcpyDeviceToHost);
    double sum = 0;
    for (int i = 0; i < H * W; ++i) sum += h_buf[i];
    printf("sum=%.0f\n", sum);
    cudaFree(d); delete[] h_buf;
    return 0;
}
