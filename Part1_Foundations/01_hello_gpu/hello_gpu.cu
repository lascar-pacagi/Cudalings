/*
 * hello_gpu.cu -- Your Very First CUDA Program
 * ==============================================
 *
 * This program does something simple but profound: it runs code on the GPU.
 * We'll launch a "kernel" (a function that runs on the GPU) with different
 * configurations to see how threads and blocks work.
 *
 * Key concepts introduced:
 *   - __global__ keyword (marks a function as a GPU kernel)
 *   - <<<blocks, threads>>> launch syntax
 *   - threadIdx.x, blockIdx.x, blockDim.x built-in variables
 *   - cudaDeviceSynchronize() (wait for GPU to finish)
 *
 * Compile: nvcc -arch=sm_61 -O2 -o hello_gpu hello_gpu.cu
 * Run:     ./hello_gpu
 */

#include <stdio.h>

// =============================================================================
// KERNEL DEFINITIONS
// =============================================================================

/*
 * __global__ tells the compiler:
 *   "This function runs on the GPU but is called from the CPU."
 *
 * Rules for __global__ functions:
 *   - Must return void (cannot return a value)
 *   - Cannot be called like a normal function; must use <<<>>> syntax
 *   - Executes asynchronously (CPU does NOT wait for it to finish)
 *   - Each thread that runs this function has its own threadIdx, blockIdx, etc.
 */
__global__ void hello_from_gpu() {
    // printf works inside CUDA kernels (since CC 2.0+).
    // Each thread executes this printf independently.
    // Note: GPU printf buffers output; you may see threads print out of order.
    printf("Hello from GPU! I am thread %d in block %d\n",
           threadIdx.x,   // This thread's index within its block (0, 1, 2, ...)
           blockIdx.x);   // This block's index within the grid (0, 1, 2, ...)
}

/*
 * A slightly more detailed kernel that shows the global thread index.
 *
 * The global index is how you map threads to data elements:
 *
 *   Block 0         Block 1         Block 2
 *   [T0 T1 T2 T3]  [T0 T1 T2 T3]  [T0 T1 T2 T3]
 *    |  |  |  |      |  |  |  |      |  |  |  |
 *    0  1  2  3      4  5  6  7      8  9 10 11    <-- global index
 *
 *   global_id = blockIdx.x * blockDim.x + threadIdx.x
 */
__global__ void hello_with_index() {
    // blockDim.x = number of threads per block (set at launch time)
    // blockIdx.x = which block this thread belongs to
    // threadIdx.x = this thread's position within the block
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of threads in the entire grid
    int total_threads = gridDim.x * blockDim.x;

    printf("Thread %2d (block %d, local thread %2d) | total threads = %d\n",
           global_id, blockIdx.x, threadIdx.x, total_threads);
}


// =============================================================================
// MAIN -- runs on the CPU (host)
// =============================================================================

int main() {
    // =========================================================================
    // Demo 1: Single thread -- <<<1, 1>>>
    // =========================================================================
    //
    // This launches 1 block with 1 thread. Basically sequential execution
    // on the GPU. Not useful in practice, but a good starting point.
    //
    //   Grid:  [ Block 0 ]
    //   Block: [ Thread 0 ]
    //
    printf("=== Demo 1: <<<1, 1>>> -- one block, one thread ===\n");

    hello_from_gpu<<<1, 1>>>();

    // IMPORTANT: Kernel launches are ASYNCHRONOUS. The CPU continues executing
    // immediately after launching the kernel. We must call cudaDeviceSynchronize()
    // to wait for the GPU to finish before we can see the output.
    //
    //   CPU timeline:  [launch kernel]---[continues here immediately]---[sync]---
    //   GPU timeline:  .................[kernel runs]......................[done]
    //
    cudaDeviceSynchronize();

    printf("\n");

    // =========================================================================
    // Demo 2: One block, 32 threads -- <<<1, 32>>>
    // =========================================================================
    //
    // This launches 1 block with 32 threads. 32 is the warp size -- this means
    // all threads execute in lockstep (same instruction at the same time).
    //
    //   Grid:  [ Block 0                               ]
    //   Block: [ T0 T1 T2 T3 T4 ... T29 T30 T31       ]
    //                    (one warp)
    //
    printf("=== Demo 2: <<<1, 32>>> -- one block, 32 threads (one warp) ===\n");

    hello_from_gpu<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n");

    // =========================================================================
    // Demo 3: Two blocks, 32 threads each -- <<<2, 32>>>
    // =========================================================================
    //
    // This launches 2 blocks, each with 32 threads = 64 threads total.
    // The two blocks may run on different SMs (Streaming Multiprocessors).
    //
    //   Grid:  [ Block 0              ] [ Block 1              ]
    //          [ T0 T1 ... T31        ] [ T0 T1 ... T31        ]
    //
    // Note: Block execution order is NOT guaranteed! Block 1 might print
    // before Block 0. This is fundamental to GPU programming -- blocks
    // are independent and can run in any order.
    //
    printf("=== Demo 3: <<<2, 32>>> -- two blocks, 32 threads each ===\n");

    hello_with_index<<<2, 32>>>();
    cudaDeviceSynchronize();

    printf("\n");

    // =========================================================================
    // Demo 4: Larger launch -- <<<4, 64>>>
    // =========================================================================
    //
    // 4 blocks x 64 threads = 256 threads total.
    // Each block has 2 warps (64 / 32 = 2).
    //
    // With 256 threads, we could process 256 array elements in parallel.
    // On your Quadro P4000 with 14 SMs, the 4 blocks would be distributed
    // across the SMs by the hardware scheduler.
    //
    printf("=== Demo 4: <<<4, 64>>> -- four blocks, 64 threads each (256 total) ===\n");
    printf("    (showing global thread IDs with block info)\n");

    hello_with_index<<<4, 64>>>();
    cudaDeviceSynchronize();

    printf("\n");

    // =========================================================================
    // Error checking -- always a good idea!
    // =========================================================================
    //
    // CUDA errors are asynchronous. If a kernel launch fails, the error is
    // stored internally. cudaGetLastError() retrieves and clears it.
    //
    // Common errors:
    //   - Invalid launch configuration (too many threads per block)
    //   - Kernel accessed invalid memory
    //   - Device not available
    //
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // cudaGetErrorString converts the error code to a human-readable string
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("All demos completed successfully!\n");
    printf("\nKey takeaways:\n");
    printf("  - __global__ functions run on the GPU\n");
    printf("  - <<<blocks, threads>>> sets the launch configuration\n");
    printf("  - threadIdx.x = thread index within block\n");
    printf("  - blockIdx.x  = block index within grid\n");
    printf("  - global_id   = blockIdx.x * blockDim.x + threadIdx.x\n");
    printf("  - Block execution order is NOT guaranteed\n");
    printf("  - Always call cudaDeviceSynchronize() to wait for completion\n");

    return 0;
}
