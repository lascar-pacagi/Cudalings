// CUDAlings 14.07 — Max-pooling 2x2, stride 2
//
// Reduces (H, W) -> (H/2, W/2) by taking the max of each 2x2 window.
// Used everywhere in CNNs to halve resolution. Backward needs to remember
// the argmax position so the gradient flows only to the winning input.
//
// Goal: forward only. Input is identity-ish: x[r,c] = r*W + c. Each 2x2
// window's max is at its bottom-right corner. We sum the output and check.

// I AM NOT DONE

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
    // TODO: loop dr, dc in {0, 1}: m = max(m, in[(irr+dr) * W + (ic+dc)])
    out[orr * OW + oc] = m;
}

int main() {
    float h_in[H*W];
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) h_in[r*W + c] = (float)(r * W + c);
    float h_out[OH*OW];
    float *d_in, *d_out;
    cudaMalloc(&d_in,  H*W*sizeof(float));
    cudaMalloc(&d_out, OH*OW*sizeof(float));
    cudaMemcpy(d_in, h_in, H*W*sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(8, 8);
    dim3 grid((OW+7)/8, (OH+7)/8);
    maxpool2x2<<<grid, block>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, OH*OW*sizeof(float), cudaMemcpyDeviceToHost);
    // Sum of bottom-right corners of all 2x2 windows of an 8x8 grid:
    //   for r in {1,3,5,7}, c in {1,3,5,7}: r*W + c
    //   = sum_{r in {1,3,5,7}} sum_{c in {1,3,5,7}} (8r + c)
    //   = 4*8*(1+3+5+7) + 4*(1+3+5+7) = 4*8*16 + 4*16 = 512 + 64 = 576.
    double s = 0;
    for (int i = 0; i < OH*OW; ++i) s += h_out[i];
    printf("sum=%.0f\n", s);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
