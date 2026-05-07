// CUDAlings 04.02 — Catching async kernel errors
//
// Kernel launches return immediately without an error. The error becomes
// visible only via cudaGetLastError() or the next synchronizing call.
// Goal: launch a kernel with an obviously bad config (1024+ threads/block on
// hardware where 1024 is the limit), then detect the launch failure and
// print "launch error: <name>".

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void noop() {}

int main() {
    noop<<<1, 100000>>>();   // > maxThreadsPerBlock => invalid configuration
    // TODO: query the last error and, if non-success, print `launch error: <name>`
    return 0;
}
