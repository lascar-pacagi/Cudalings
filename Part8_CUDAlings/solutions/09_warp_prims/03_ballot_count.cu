#include <cstdio>
#include <cuda_runtime.h>
__global__ void count_pred(int* out) {
    int tid = threadIdx.x;
    int pred = (tid >= 16);
    unsigned mask = __ballot_sync(0xffffffff, pred);
    if (tid == 0) *out = __popc(mask);
}
int main() {
    int *d_out; cudaMalloc(&d_out, sizeof(int));
    count_pred<<<1, 32>>>(d_out);
    int h = 0;
    cudaMemcpy(&h, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("count=%d\n", h);
    cudaFree(d_out);
    return 0;
}
