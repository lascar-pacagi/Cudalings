// CUDAlings 01.01 — Launch your first kernel
//
// Goal: print "hello from thread 0" through "hello from thread 7", one line
// per thread, in any order. The kernel is launched once with a single block
// of 8 threads.
//
// What to do:
//   1) Inside `hello`, use printf and threadIdx.x to emit the line.
//   2) Inside `main`, launch `hello` with <<<1, 8>>> and synchronize.
//
// Why we synchronize: device-side printf output is buffered and only flushed
// when the device finishes work. Without cudaDeviceSynchronize() the program
// can return before any output appears.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello() {
    // TODO: printf("hello from thread %d\n", ...)
}

int main() {
    // TODO: launch `hello`, then call cudaDeviceSynchronize().
    return 0;
}
