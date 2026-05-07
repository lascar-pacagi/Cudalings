// CUDAlings 01.02 — Compute a global thread index
//
// Goal: launch 4 blocks of 8 threads (32 threads total) and have each
// thread print its global id (0..31), without duplicates.
//
// The lesson: the formula `blockIdx.x * blockDim.x + threadIdx.x` is the
// most-used identity in all of CUDA. Wire it into your fingertips.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void show_gid() {
    // TODO: compute `gid` from blockIdx.x, blockDim.x, threadIdx.x
    int gid = 0;
    printf("gid=%d\n", gid);
}

int main() {
    // TODO: launch with 4 blocks of 8 threads, then synchronize.
    return 0;
}
