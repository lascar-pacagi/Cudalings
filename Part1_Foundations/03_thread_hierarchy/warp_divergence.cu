/**
 * =============================================================================
 * Chapter 03: Warp Divergence -- Demonstration and Measurement
 * =============================================================================
 *
 * This program demonstrates warp divergence and its performance impact.
 *
 * A warp is 32 threads that execute in lockstep (SIMT). When threads in the
 * same warp take different branches, the warp must execute ALL branches
 * sequentially, masking inactive threads. This wastes execution resources.
 *
 * Hardware: Quadro P4200 (CC 6.1, 18 SMs)
 *
 * =============================================================================
 *
 * WARP DIVERGENCE DIAGRAM
 * =======================
 *
 * NO DIVERGENCE (all threads take the same path):
 *
 *   if (blockIdx.x > 0) {   // entire block goes one way
 *       path_A();            // or entire block goes the other
 *   }
 *
 *   Warp 0: [A][A][A][A][A][A][A][A] ... [A]   32/32 active = 100% efficient
 *   Warp 1: [A][A][A][A][A][A][A][A] ... [A]   32/32 active = 100% efficient
 *
 *
 * MILD DIVERGENCE (half the threads diverge):
 *
 *   if (threadIdx.x % 2 == 0) {
 *       path_A();   // even threads
 *   } else {
 *       path_B();   // odd threads
 *   }
 *
 *   Pass 1 (path A): [A][-][A][-][A][-][A][-] ... [-]  16/32 active = 50%
 *   Pass 2 (path B): [-][B][-][B][-][B][-][B] ... [B]  16/32 active = 50%
 *   Total: 2 passes needed -> ~2x slower
 *
 *
 * HEAVY DIVERGENCE (many different paths):
 *
 *   switch (threadIdx.x % 8) {
 *       case 0: path_0(); break;
 *       case 1: path_1(); break;
 *       ...
 *       case 7: path_7(); break;
 *   }
 *
 *   Pass 1: [0][-][-][-][-][-][-][-][0][-]...   4/32 active = 12.5%
 *   Pass 2: [-][1][-][-][-][-][-][-][-][1]...   4/32 active = 12.5%
 *   Pass 3: [-][-][2][-][-][-][-][-][-][-]...   4/32 active = 12.5%
 *   ...
 *   Pass 8: [-][-][-][-][-][-][-][7][-][-]...   4/32 active = 12.5%
 *   Total: 8 passes needed -> ~8x slower
 *
 *
 * NO DIVERGENCE (divergence at warp boundaries):
 *
 *   if (threadIdx.x < 32) {   // entire warp 0 goes one way
 *       path_A();
 *   } else {                  // entire warp 1+ goes another
 *       path_B();
 *   }
 *
 *   Warp 0: [A][A][A][A][A][A][A][A] ... [A]  32/32 active = 100%
 *   Warp 1: [B][B][B][B][B][B][B][B] ... [B]  32/32 active = 100%
 *   No divergence! Each warp is uniform.
 *
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


/* ---------------------------------------------------------------------------
 * Configuration
 * ---------------------------------------------------------------------------*/
#define N          (1 << 22)   /* ~4 million elements (4,194,304) */
#define BLOCK_SIZE 256         /* 256 threads = 8 warps per block */
#define ITERATIONS 100         /* Repeat to get stable timing */


/* ---------------------------------------------------------------------------
 * Kernel 1: NO divergence
 * ---------------------------------------------------------------------------
 * All threads execute the same arithmetic. No branches at all.
 * This is our baseline for comparison.
 * ---------------------------------------------------------------------------*/
__global__ void kernel_no_divergence(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        /*
         * All threads do the same work: a series of math operations.
         * No if/else, no branching within a warp.
         *
         * Warp execution:
         *   All 32 lanes: LOAD -> MUL -> ADD -> SIN -> ADD -> STORE
         *   100% lane utilization
         */
        float val = data[idx];
        val = val * 2.0f + 1.0f;
        val = sinf(val) + 0.5f;
        val = val * val + 3.0f;
        data[idx] = val;
    }
}


/* ---------------------------------------------------------------------------
 * Kernel 2: MILD divergence (even/odd threads take different paths)
 * ---------------------------------------------------------------------------
 * threadIdx.x % 2 splits every warp in half.
 * Each warp must execute both paths sequentially.
 *
 * Within each warp:
 *   Lanes 0,2,4,...,30 -> path A (16 threads)
 *   Lanes 1,3,5,...,31 -> path B (16 threads)
 *
 * Both paths do the same AMOUNT of work, but the warp serializes them.
 * ---------------------------------------------------------------------------*/
__global__ void kernel_mild_divergence(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];

        /*
         * This branch causes divergence within every warp:
         *   Even lanes (0,2,4,...): path A
         *   Odd  lanes (1,3,5,...): path B
         *
         * The hardware runs path A first (16 active lanes),
         * then path B (16 active lanes).
         */
        if (threadIdx.x % 2 == 0) {
            /* Path A: even threads */
            val = val * 2.0f + 1.0f;
            val = sinf(val) + 0.5f;
            val = val * val + 3.0f;
        } else {
            /* Path B: odd threads (same amount of work, different ops) */
            val = val * 3.0f - 1.0f;
            val = cosf(val) + 0.5f;
            val = val * val + 2.0f;
        }

        data[idx] = val;
    }
}


/* ---------------------------------------------------------------------------
 * Kernel 3: HEAVY divergence (8-way branch)
 * ---------------------------------------------------------------------------
 * threadIdx.x % 8 creates 8 different paths within each warp.
 * Each path has only 4 active threads out of 32.
 *
 * The warp scheduler must serialize all 8 paths.
 * ---------------------------------------------------------------------------*/
__global__ void kernel_heavy_divergence(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];

        /*
         * 8-way divergence: threadIdx.x % 8 gives values 0-7
         * Within each warp of 32 threads:
         *   4 threads take case 0
         *   4 threads take case 1
         *   ...
         *   4 threads take case 7
         *
         * Hardware must execute 8 passes, each with only 4/32 lanes active.
         */
        switch (threadIdx.x % 8) {
            case 0:
                val = val * 2.0f + 1.0f;
                val = sinf(val) + 0.5f;
                val = val * val + 3.0f;
                break;
            case 1:
                val = val * 3.0f - 1.0f;
                val = cosf(val) + 0.5f;
                val = val * val + 2.0f;
                break;
            case 2:
                val = val * 1.5f + 2.0f;
                val = sinf(val) - 0.5f;
                val = val * val + 1.0f;
                break;
            case 3:
                val = val * 2.5f - 0.5f;
                val = cosf(val) + 1.5f;
                val = val * val + 4.0f;
                break;
            case 4:
                val = val * 1.0f + 3.0f;
                val = sinf(val) + 2.5f;
                val = val * val + 5.0f;
                break;
            case 5:
                val = val * 3.5f - 2.0f;
                val = cosf(val) - 1.5f;
                val = val * val + 6.0f;
                break;
            case 6:
                val = val * 0.5f + 4.0f;
                val = sinf(val) + 3.5f;
                val = val * val + 7.0f;
                break;
            case 7:
                val = val * 4.0f - 3.0f;
                val = cosf(val) - 2.5f;
                val = val * val + 8.0f;
                break;
        }

        data[idx] = val;
    }
}


/* ---------------------------------------------------------------------------
 * Kernel 4: Divergence at WARP boundaries (no actual divergence!)
 * ---------------------------------------------------------------------------
 * We branch based on (threadIdx.x / 32), which is the warp ID.
 * All threads within the same warp take the same path -> NO divergence.
 *
 * This demonstrates that branching is fine as long as entire warps agree.
 * ---------------------------------------------------------------------------*/
__global__ void kernel_warp_aligned(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];

        /*
         * Branch on warp ID (threadIdx.x / 32):
         *   Warp 0 (threads 0-31):   path A
         *   Warp 1 (threads 32-63):  path B
         *   Warp 2 (threads 64-95):  path A
         *   Warp 3 (threads 96-127): path B
         *   ...
         *
         * Every thread in each warp follows the same branch.
         * NO warp divergence! Performance should match kernel_no_divergence.
         */
        int warp_id = threadIdx.x / 32;

        if (warp_id % 2 == 0) {
            /* Path A: even warps */
            val = val * 2.0f + 1.0f;
            val = sinf(val) + 0.5f;
            val = val * val + 3.0f;
        } else {
            /* Path B: odd warps (same amount of work) */
            val = val * 3.0f - 1.0f;
            val = cosf(val) + 0.5f;
            val = val * val + 2.0f;
        }

        data[idx] = val;
    }
}


/* ---------------------------------------------------------------------------
 * Timing helper using CUDA events
 * ---------------------------------------------------------------------------*/
float time_kernel(void (*kernel)(float*, int), float *d_data, float *d_source,
                  int n, int iterations) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    /* Warm-up run */
    CUDA_CHECK(cudaMemcpy(d_data, d_source, n * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    kernel<<<blocks, BLOCK_SIZE>>>(d_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed runs */
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        /* Reset data each iteration to keep values in a reasonable range */
        CUDA_CHECK(cudaMemcpy(d_data, d_source, n * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        kernel<<<blocks, BLOCK_SIZE>>>(d_data, n);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iterations;  /* Average time per iteration */
}


/* ---------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------------*/
int main() {
    printf("==========================================================\n");
    printf("  Chapter 03: Warp Divergence Demo\n");
    printf("==========================================================\n");
    printf("Array size: %d elements (%.1f MB)\n",
           N, (float)N * sizeof(float) / (1024 * 1024));
    printf("Block size: %d threads (%d warps per block)\n",
           BLOCK_SIZE, BLOCK_SIZE / 32);
    printf("Grid size:  %d blocks\n", (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    printf("Iterations: %d (for timing stability)\n", ITERATIONS);

    /* -----------------------------------------------------------------------
     * Allocate and initialize data
     * -----------------------------------------------------------------------*/
    float *h_data = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i % 100) / 100.0f;  /* Values in [0, 1) */
    }

    float *d_data, *d_source;
    CUDA_CHECK(cudaMalloc(&d_data,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_source, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_source, h_data, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* -----------------------------------------------------------------------
     * Time each kernel
     * -----------------------------------------------------------------------*/
    printf("\n----------------------------------------------------------\n");
    printf("Timing Results (average over %d iterations):\n", ITERATIONS);
    printf("----------------------------------------------------------\n\n");

    float t_none = time_kernel(kernel_no_divergence, d_data, d_source,
                               N, ITERATIONS);
    printf("  1. No divergence:         %8.3f ms\n", t_none);

    float t_mild = time_kernel(kernel_mild_divergence, d_data, d_source,
                               N, ITERATIONS);
    printf("  2. Mild divergence (2x):  %8.3f ms  (%.2fx slower)\n",
           t_mild, t_mild / t_none);

    float t_heavy = time_kernel(kernel_heavy_divergence, d_data, d_source,
                                N, ITERATIONS);
    printf("  3. Heavy divergence (8x): %8.3f ms  (%.2fx slower)\n",
           t_heavy, t_heavy / t_none);

    float t_aligned = time_kernel(kernel_warp_aligned, d_data, d_source,
                                  N, ITERATIONS);
    printf("  4. Warp-aligned branch:   %8.3f ms  (%.2fx vs baseline)\n",
           t_aligned, t_aligned / t_none);

    /* -----------------------------------------------------------------------
     * Analysis
     * -----------------------------------------------------------------------*/
    printf("\n----------------------------------------------------------\n");
    printf("Analysis:\n");
    printf("----------------------------------------------------------\n");
    printf("\n");
    printf("Kernel 1 (no divergence):\n");
    printf("  All 32 lanes active -> maximum throughput.\n\n");
    printf("Kernel 2 (mild divergence, even/odd):\n");
    printf("  Each warp serializes 2 paths. Expect ~1.5-2x slowdown.\n");
    printf("  (Not exactly 2x because the compiler may optimize.)\n\n");
    printf("Kernel 3 (heavy divergence, 8-way):\n");
    printf("  Each warp serializes up to 8 paths. Expect significant\n");
    printf("  slowdown, though hardware can sometimes mask some cost.\n\n");
    printf("Kernel 4 (warp-aligned):\n");
    printf("  Although there is an if/else, entire warps go one way.\n");
    printf("  Should be nearly as fast as kernel 1 (no divergence).\n\n");

    printf("----------------------------------------------------------\n");
    printf("HOW TO MINIMIZE WARP DIVERGENCE:\n");
    printf("----------------------------------------------------------\n");
    printf("\n");
    printf("1. Avoid branching on threadIdx.x within a warp (mod 32).\n");
    printf("   BAD:  if (threadIdx.x %% 2 == 0)  // splits every warp\n");
    printf("   GOOD: if (threadIdx.x / 32 < 4)   // warp-level branch\n\n");
    printf("2. Use math instead of branches when possible:\n");
    printf("   BAD:  if (x > 0) y = x; else y = 0;\n");
    printf("   GOOD: y = fmaxf(x, 0.0f);\n\n");
    printf("3. Reorganize data so threads in the same warp follow\n");
    printf("   the same path (Structure of Arrays helps).\n\n");
    printf("4. Use predication: short if/else bodies (1-2 instructions)\n");
    printf("   are often compiled to predicated instructions, avoiding\n");
    printf("   actual divergence at the hardware level.\n\n");
    printf("5. Profile with nvprof or Nsight Compute to check:\n");
    printf("     nvprof --metrics branch_efficiency ./warp_divergence\n\n");

    /* Cleanup */
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_source));

    printf("==========================================================\n");
    printf("  Done!\n");
    printf("==========================================================\n");

    return 0;
}
