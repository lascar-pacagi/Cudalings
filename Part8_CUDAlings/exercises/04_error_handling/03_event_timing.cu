// CUDAlings 04.03 — Time a kernel with cudaEvent
//
// cudaEventRecord/Synchronize/ElapsedTime is the canonical GPU timer. Use
// it instead of std::chrono — it accounts for async submission lag and
// returns wall time on the GPU's clock.
//
// Goal: time the noop kernel below over 100 launches and print
// "elapsed=<float>". Any positive number passes.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void noop() {}

int main() {
    cudaEvent_t start, stop;
    // TODO: cudaEventCreate both
    // TODO: cudaEventRecord(start)
    for (int i = 0; i < 100; ++i) noop<<<1, 32>>>();
    // TODO: cudaEventRecord(stop), cudaEventSynchronize(stop)
    float ms = 0.f;
    // TODO: cudaEventElapsedTime(&ms, start, stop)
    printf("elapsed=%.3f\n", ms);
    return 0;
}
