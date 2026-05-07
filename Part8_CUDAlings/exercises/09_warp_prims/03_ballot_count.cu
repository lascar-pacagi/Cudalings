// CUDAlings 09.03 — __ballot_sync + __popc to count predicates per warp
//
// __ballot_sync(0xffffffff, pred) returns a uint32 where bit i = (pred from lane i).
// __popc counts set bits.
//
// Goal: launch one warp; each lane sets pred = (tid >= 16). __popc on the
// ballot should yield 16. Lane 0 prints the count.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void count_pred(int* out) {
    int tid = threadIdx.x;
    int pred = (tid >= 16);
    // TODO: unsigned mask = __ballot_sync(0xffffffff, pred);
    // TODO: if (tid == 0) *out = __popc(mask);
}

int main() {
    int *d_out; cudaMalloc(&d_out, sizeof(int));
    count_pred<<<1, 32>>>(d_out);
    int h = 0;
    cudaMemcpy(&h, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("count=%d\n", h);     // expected 16
    cudaFree(d_out);
    return 0;
}
