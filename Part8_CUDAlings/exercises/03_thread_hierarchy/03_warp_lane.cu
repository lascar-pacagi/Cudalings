// CUDAlings 03.03 — Warps and lane IDs
//
// A warp is 32 threads scheduled together by the SM. Within a warp, the
// "lane id" is threadIdx.x % 32 (for 1D blocks). Knowing where a thread
// sits inside its warp is essential for shuffle, ballot, and reduction.
//
// Goal: launch 1 block of 64 threads. Each thread prints its
// (warpId, laneId). Expected output contains both warps fully.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void warp_lane() {
    int tid = threadIdx.x;
    // TODO: derive `warp` and `lane` from tid (one warp = 32 threads)
    int warp = -1, lane = -1;
    if (lane == 0)                    // print only the lane-0 of each warp
        printf("warp=%d lane0_tid=%d\n", warp, tid);
}

int main() {
    warp_lane<<<1, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}
