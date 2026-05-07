#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int N = 4;
    int h_in[] = {10, 20, 30, 40};
    int h_out[4] = {0, 0, 0, 0};
    int* d;
    cudaMalloc(&d, N * sizeof(int));
    cudaMemcpy(d, h_in,  N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h_out, d, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    int sum = 0;
    for (int i = 0; i < N; ++i) sum += h_out[i];
    printf("sum=%d\n", sum);
    return 0;
}
