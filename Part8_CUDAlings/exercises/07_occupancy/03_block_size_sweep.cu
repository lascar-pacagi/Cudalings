// CUDAlings 07.03 — Sweep block sizes to find the highest-occupancy one
//
// Block size is a knob: 64, 128, 256, 512, 1024 are all valid for most
// kernels. The sweet spot depends on register & shared-mem usage. The
// reliable way to pick: query occupancy for each candidate.
//
// Goal: print the block size in {64, 128, 256, 512, 1024} that gives the
// largest blocks_per_sm * block_size product (= active threads/SM).

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void k() {
    __shared__ float buf[64];
    if (threadIdx.x < 64) buf[threadIdx.x] = (float)threadIdx.x;
    __syncthreads();
}

int main() {
    int sizes[] = {64, 128, 256, 512, 1024};
    int best_size = 0, best_threads = 0;
    for (int s : sizes) {
        int blocks = 0;
        // TODO: cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k, s, 0);
        int active = blocks * s;
        if (active > best_threads) { best_threads = active; best_size = s; }
    }
    printf("best=%d threads=%d\n", best_size, best_threads);
    return 0;
}
