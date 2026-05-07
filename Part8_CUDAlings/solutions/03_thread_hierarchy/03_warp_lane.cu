#include <cstdio>
#include <cuda_runtime.h>
__global__ void warp_lane() {
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;
    if (lane == 0) printf("warp=%d lane0_tid=%d\n", warp, tid);
}
int main() {
    warp_lane<<<1, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}
