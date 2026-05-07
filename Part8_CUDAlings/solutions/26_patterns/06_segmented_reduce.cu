#include <cstdio>
#include <cuda_runtime.h>
#define N 6
#define S 3
__global__ void seg_sum(const int* data, const int* seg, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&out[seg[i]], data[i]);
}
int main() {
    int h_data[N] = {10, 20, 30, 40, 50, 60};
    int h_seg[N]  = { 0,  0,  1,  1,  2,  2};
    int h_out[S]  = {0};
    int *d_data, *d_seg, *d_out;
    cudaMalloc(&d_data, N*sizeof(int));
    cudaMalloc(&d_seg,  N*sizeof(int));
    cudaMalloc(&d_out,  S*sizeof(int));
    cudaMemcpy(d_data, h_data, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seg,  h_seg,  N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, S*sizeof(int));
    seg_sum<<<1, N>>>(d_data, d_seg, d_out, N);
    cudaMemcpy(h_out, d_out, S*sizeof(int), cudaMemcpyDeviceToHost);
    int sum = h_out[0] + h_out[1] + h_out[2];
    printf("total=%d\n", sum);
    cudaFree(d_data); cudaFree(d_seg); cudaFree(d_out);
    return 0;
}
