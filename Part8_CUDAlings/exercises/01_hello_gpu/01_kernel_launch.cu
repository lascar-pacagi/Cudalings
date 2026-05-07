// CUDAlings 01.01 — Launch your first kernel
//
// Goal: print "hello from thread 0" through "hello from thread 7", one line
// per thread, in any order. The kernel is launched once with a single block
// of 8 threads.
//
// Why we synchronize: device-side printf output is buffered and only flushed
// when the device finishes work. Without a synchronize, the program can
// return before any output appears.
//
// (Hint file: `./cudalings hint 01_hello_gpu/01_kernel_launch`.)

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello() {
    // TODO: emit one line per thread containing its thread index
}

int main() {
    // TODO: launch `hello` (one block, 8 threads) then wait for it to finish
    return 0;
}
