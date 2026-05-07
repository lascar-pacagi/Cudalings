#include <cstdio>
#include <cuda_runtime.h>
#define H 8
#define W 8
#define OH (H/2)
#define OW (W/2)
__global__ void maxpool2x2(const float* in, float* out) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    int orr = blockIdx.y * blockDim.y + threadIdx.y;
    if (oc >= OW || orr >= OH) return;
    int ic = oc * 2;
    int irr = orr * 2;
    float m = -1e30f;
    for (int dr = 0; dr < 2; ++dr)
        for (int dc = 0; dc < 2; ++dc) {
            float v = in[(irr+dr) * W + (ic+dc)];
            if (v > m) m = v;
        }
    out[orr * OW + oc] = m;
}
int main() {
    float h_in[H*W];
    for (int r = 0; r < H; ++r) for (int c = 0; c < W; ++c) h_in[r*W + c] = (float)(r*W + c);
    float h_out[OH*OW];
    float *d_in, *d_out;
    cudaMalloc(&d_in,  H*W*sizeof(float));
    cudaMalloc(&d_out, OH*OW*sizeof(float));
    cudaMemcpy(d_in, h_in, H*W*sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(8, 8);
    dim3 grid((OW+7)/8, (OH+7)/8);
    maxpool2x2<<<grid, block>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, OH*OW*sizeof(float), cudaMemcpyDeviceToHost);
    double s = 0;
    for (int i = 0; i < OH*OW; ++i) s += h_out[i];
    printf("sum=%.0f\n", s);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
