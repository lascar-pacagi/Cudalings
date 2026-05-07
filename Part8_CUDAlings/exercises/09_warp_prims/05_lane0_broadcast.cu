// CUDAlings 09.05 — Broadcast lane 0's value to every lane in the warp
//
// __shfl_sync(mask, v, src_lane) returns lane `src_lane`'s `v` to every
// participating lane. Use src_lane=0 to broadcast lane 0.
//
// Goal: lane 0 has value 42.0; after broadcast every lane should have 42.
// We sum across the warp -- expect 32 * 42 = 1344.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void broadcast(float* sum_out) {
    int lane = threadIdx.x & 31;
    float v = (lane == 0) ? 42.0f : 0.0f;
    // TODO: v = __shfl_sync(0xffffffff, v, 0);

    // sum across warp
    for (int d = 16; d > 0; d >>= 1)
        v += __shfl_down_sync(0xffffffff, v, d);
    if (lane == 0) *sum_out = v;
}

int main() {
    float* d_out; cudaMalloc(&d_out, sizeof(float));
    broadcast<<<1, 32>>>(d_out);
    float h = 0;
    cudaMemcpy(&h, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum=%.0f\n", h);
    cudaFree(d_out);
    return 0;
}
