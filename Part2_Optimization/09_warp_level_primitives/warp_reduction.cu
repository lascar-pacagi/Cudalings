/* ============================================================================
 * warp_reduction.cu  --  Chapter 09: Warp-Level and Block-Level Reduction
 * ============================================================================
 *
 * This program demonstrates the MOST IMPORTANT application of warp shuffles:
 * parallel reduction (sum of N values).
 *
 * We build three versions:
 *   1. Warp-level reduction using __shfl_down_sync  (32 elements → 1)
 *   2. Shared memory reduction for comparison        (32 elements → 1)
 *   3. Block-level reduction: warp reduce + shared memory + final warp reduce
 *
 * Then we benchmark warp shuffle vs shared memory for a large reduction.
 *
 * Target GPU : Quadro P4200  (CC 6.1, Pascal, 18 SMs)
 * CUDA       : 11.7
 * Compile    : nvcc -arch=sm_61 -O2 -lineinfo -ccbin g++-11 warp_reduction.cu -o warp_reduction
 * ========================================================================= */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define FULL_MASK 0xFFFFFFFF

/* ---------------------------------------------------------------------------
 * CUDA error checking macro.
 * Wraps every CUDA call to catch errors early.
 * ------------------------------------------------------------------------ */
#define CHECK_CUDA(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ===========================================================================
 * warpReduceSum: reduce 32 values within a single warp to 1 sum.
 *
 * This is the core building block.  It uses __shfl_down_sync to perform
 * a tree reduction in 5 steps (log2(32) = 5).
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  REDUCTION TREE (showing 8 lanes for clarity; full warp has 32)       │
 * │                                                                        │
 * │  Initial values:                                                       │
 * │  Lane:  0    1    2    3    4    5    6    7                           │
 * │  Val:   v0   v1   v2   v3   v4   v5   v6   v7                        │
 * │                                                                        │
 * │  Step 1: __shfl_down by 4  (each lane adds value from lane+4)         │
 * │  Lane:  0         1         2         3         4    5    6    7       │
 * │  Val:   v0+v4     v1+v5     v2+v6     v3+v7     v4   v5   v6   v7    │
 * │         ↑ got v4  ↑ got v5  ↑ got v6  ↑ got v7                       │
 * │                                                                        │
 * │  Step 2: __shfl_down by 2  (each lane adds value from lane+2)         │
 * │  Lane:  0              1              2         3         ...         │
 * │  Val:   v0+v4+v2+v6    v1+v5+v3+v7   (stale)   (stale)              │
 * │         ↑ got v2+v6    ↑ got v3+v7                                    │
 * │                                                                        │
 * │  Step 3: __shfl_down by 1  (each lane adds value from lane+1)         │
 * │  Lane:  0                                                              │
 * │  Val:   v0+v1+v2+v3+v4+v5+v6+v7    ← FINAL SUM!                     │
 * │         ↑ got v1+v5+v3+v7                                              │
 * │                                                                        │
 * │  For full 32-lane warp: 5 steps with deltas 16, 8, 4, 2, 1           │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * IMPORTANT: after this function, ONLY lane 0 holds the correct sum.
 * All other lanes hold intermediate (partially reduced) values.
 * ========================================================================= */
__device__ float warpReduceSum(float val) {
    /*
     * Step 1: delta = 16
     *   Lane i gets the value from lane (i + 16).
     *   Lanes 0-15 accumulate partial sums; lanes 16-31 are now "done".
     */
    val += __shfl_down_sync(FULL_MASK, val, 16);

    /*
     * Step 2: delta = 8
     *   Lane i gets the value from lane (i + 8).
     *   Lanes 0-7 accumulate; lanes 8-15 are done.
     */
    val += __shfl_down_sync(FULL_MASK, val, 8);

    /*
     * Step 3: delta = 4
     *   Lanes 0-3 accumulate from lanes 4-7.
     */
    val += __shfl_down_sync(FULL_MASK, val, 4);

    /*
     * Step 4: delta = 2
     *   Lanes 0-1 accumulate from lanes 2-3.
     */
    val += __shfl_down_sync(FULL_MASK, val, 2);

    /*
     * Step 5: delta = 1
     *   Lane 0 accumulates from lane 1.  DONE!
     */
    val += __shfl_down_sync(FULL_MASK, val, 1);

    return val;   // only lane 0 has the correct total sum
}

/* ===========================================================================
 * Kernel 1: Demonstrate warp reduction on 32 elements
 *
 * One block, one warp (32 threads).  Each lane has a value; reduce to sum.
 * ========================================================================= */
__global__ void warpReduceDemo(const float* input, float* output) {
    int lane = threadIdx.x;                    // 0-31
    float val = input[lane];                   // each lane loads one element

    float sum = warpReduceSum(val);            // reduce within the warp

    if (lane == 0) {
        output[0] = sum;                       // lane 0 writes the result
    }
}

/* ===========================================================================
 * Kernel 2: Shared memory reduction for comparison (32 elements)
 *
 * Traditional approach: write to shared memory, synchronize, tree reduce.
 * This is slower because:
 *   - Each step requires a shared memory write + read (~20-30 cycles each)
 *   - __syncthreads() barriers between steps
 *   - Uses shared memory that could be used for tiling, etc.
 * ========================================================================= */
__global__ void sharedMemReduceDemo(const float* input, float* output) {
    __shared__ float sdata[32];                // 32 floats in shared memory

    int tid = threadIdx.x;
    sdata[tid] = input[tid];                   // load into shared memory
    __syncthreads();                           // make sure all threads have loaded

    /* Tree reduction in shared memory, 5 steps */

    /* Step 1: stride 16 */
    if (tid < 16) sdata[tid] += sdata[tid + 16];
    __syncthreads();

    /* Step 2: stride 8 */
    if (tid < 8) sdata[tid] += sdata[tid + 8];
    __syncthreads();

    /* Step 3: stride 4 */
    if (tid < 4) sdata[tid] += sdata[tid + 4];
    __syncthreads();

    /* Step 4: stride 2 */
    if (tid < 2) sdata[tid] += sdata[tid + 2];
    __syncthreads();

    /* Step 5: stride 1 */
    if (tid == 0) {
        sdata[0] += sdata[1];
        output[0] = sdata[0];
    }
}

/* ===========================================================================
 * blockReduceSum: reduce all values in a block to a single sum.
 *
 * Strategy:
 *   Phase 1 -- Each warp reduces its 32 values using warpReduceSum
 *   Phase 2 -- Lane 0 of each warp writes to shared memory
 *   Phase 3 -- The first warp loads the partial sums and reduces them
 *
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  Block with 256 threads = 8 warps                               │
 *   │                                                                  │
 *   │  Phase 1: Warp-level reduction (all warps in parallel)          │
 *   │  ┌──────────────┐  ┌──────────────┐      ┌──────────────┐      │
 *   │  │ Warp 0       │  │ Warp 1       │      │ Warp 7       │      │
 *   │  │ 32 vals → S0 │  │ 32 vals → S1 │ ...  │ 32 vals → S7 │      │
 *   │  └──────┬───────┘  └──────┬───────┘      └──────┬───────┘      │
 *   │         │                 │                      │              │
 *   │  Phase 2: Write partial sums to shared memory                   │
 *   │         ↓                 ↓                      ↓              │
 *   │  shared[0] = S0   shared[1] = S1  ...   shared[7] = S7         │
 *   │                                                                  │
 *   │  Phase 3: First warp reduces shared[0..7]                       │
 *   │  ┌──────────────────────────────────────────────────┐           │
 *   │  │ Warp 0: load shared[0..7], warpReduceSum → TOTAL│           │
 *   │  └──────────────────────────────────────────────────┘           │
 *   └──────────────────────────────────────────────────────────────────┘
 * ========================================================================= */
__device__ float blockReduceSum(float val) {
    /* Shared memory for inter-warp communication.
     * We need at most 32 entries (one per warp, max 1024 threads / 32 = 32 warps).
     */
    __shared__ float warpSums[32];

    int lane   = threadIdx.x % 32;            // lane within the warp (0-31)
    int warpId = threadIdx.x / 32;            // which warp in the block

    /* Phase 1: reduce within each warp */
    val = warpReduceSum(val);

    /* Phase 2: lane 0 of each warp writes its partial sum to shared memory */
    if (lane == 0) {
        warpSums[warpId] = val;
    }

    /* Barrier: all warps must finish writing before the first warp reads */
    __syncthreads();

    /* Phase 3: first warp reduces the partial sums.
     * How many warps do we have?  blockDim.x / 32.
     * First warp loads warpSums[0..numWarps-1].
     * Lanes beyond numWarps load 0 (identity for sum).
     */
    int numWarps = blockDim.x / 32;
    if (warpId == 0) {
        val = (lane < numWarps) ? warpSums[lane] : 0.0f;
        val = warpReduceSum(val);
    }

    return val;   // only thread 0 of the block has the correct sum
}

/* ===========================================================================
 * Kernel 3: Block-level reduction (full-size blocks)
 *
 * Each block reduces blockDim.x elements.  The per-block sums are written
 * to output[], and a second pass (or atomic) combines them.
 *
 * For simplicity, we use atomicAdd to accumulate across blocks.
 * (In production code, you might do a two-pass reduction instead.)
 * ========================================================================= */
__global__ void blockReduceKernel(const float* input, float* output, int N) {
    /* Global thread ID */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load element (or 0 if out of bounds) */
    float val = (tid < N) ? input[tid] : 0.0f;

    /* Reduce within this block */
    val = blockReduceSum(val);

    /* Thread 0 of each block atomically adds its block's sum to output */
    if (threadIdx.x == 0) {
        atomicAdd(output, val);
    }
}

/* ===========================================================================
 * Kernel 4: Shared-memory-only block reduction (for benchmark comparison)
 *
 * Classic shared memory tree reduction.  No warp shuffles.
 * ========================================================================= */
__global__ void sharedMemBlockReduceKernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];

    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int locTid = threadIdx.x;

    /* Load (or 0 if OOB) */
    sdata[locTid] = (tid < N) ? input[tid] : 0.0f;
    __syncthreads();

    /* Tree reduction in shared memory */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (locTid < stride) {
            sdata[locTid] += sdata[locTid + stride];
        }
        __syncthreads();
    }

    /* Thread 0 writes result */
    if (locTid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

/* ===========================================================================
 * main()
 * ========================================================================= */
int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Chapter 09: Warp-Level and Block-Level Reduction\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    /* ── Part 1: Demonstrate warp reduction on 32 elements ─────────────── */
    {
        printf("── Part 1: Warp Reduction (32 elements) ──\n\n");

        /* Prepare 32 floats: values 1, 2, 3, ..., 32 */
        float h_in[32];
        for (int i = 0; i < 32; i++) h_in[i] = (float)(i + 1);

        /* Expected sum: 32*33/2 = 528 */
        float expected = 32.0f * 33.0f / 2.0f;

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in,  32 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in, 32 * sizeof(float), cudaMemcpyHostToDevice));

        /* Warp shuffle reduction */
        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
        warpReduceDemo<<<1, 32>>>(d_in, d_out);
        float h_out;
        CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  Warp shuffle reduction:   sum = %.1f  (expected %.1f) %s\n",
               h_out, expected, fabsf(h_out - expected) < 0.1f ? "[PASS]" : "[FAIL]");

        /* Shared memory reduction */
        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
        sharedMemReduceDemo<<<1, 32>>>(d_in, d_out);
        CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  Shared mem reduction:     sum = %.1f  (expected %.1f) %s\n\n",
               h_out, expected, fabsf(h_out - expected) < 0.1f ? "[PASS]" : "[FAIL]");

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
    }

    /* ── Part 2: Block-level reduction on large array ──────────────────── */
    {
        printf("── Part 2: Block-Level Reduction (large array) ──\n\n");

        const int N = 1 << 20;  // 1M elements
        const int blockSize = 256;
        const int gridSize  = (N + blockSize - 1) / blockSize;

        /* Prepare data: all 1.0f → sum should be N */
        float* h_in = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) h_in[i] = 1.0f;
        float expected = (float)N;

        float *d_in, *d_out;
        CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        /* Warp shuffle block reduction */
        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
        blockReduceKernel<<<gridSize, blockSize>>>(d_in, d_out, N);
        float h_out;
        CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  Warp shuffle block reduction: sum = %.0f  (expected %.0f) %s\n",
               h_out, expected, fabsf(h_out - expected) < 1.0f ? "[PASS]" : "[FAIL]");

        /* Shared memory block reduction */
        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
        sharedMemBlockReduceKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, N);
        CHECK_CUDA(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        printf("  Shared mem block reduction:   sum = %.0f  (expected %.0f) %s\n\n",
               h_out, expected, fabsf(h_out - expected) < 1.0f ? "[PASS]" : "[FAIL]");

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
        free(h_in);
    }

    /* ── Part 3: Benchmark warp shuffle vs shared memory ───────────────── */
    {
        printf("── Part 3: Benchmark (warp shuffle vs shared memory) ──\n\n");

        const int N = 1 << 24;   // 16M elements
        const int blockSize = 256;
        const int gridSize  = (N + blockSize - 1) / blockSize;
        const int nIter     = 100;

        /* Allocate and initialize */
        float* d_in;
        float* d_out;
        CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));

        /* Fill with 1.0f */
        float* h_in = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) h_in[i] = 1.0f;
        CHECK_CUDA(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

        /* Create CUDA events for timing */
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        /* ── Benchmark: warp shuffle reduction ── */
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < nIter; i++) {
            CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
            blockReduceKernel<<<gridSize, blockSize>>>(d_in, d_out, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_shuffle;
        CHECK_CUDA(cudaEventElapsedTime(&ms_shuffle, start, stop));
        ms_shuffle /= nIter;

        /* ── Benchmark: shared memory reduction ── */
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < nIter; i++) {
            CHECK_CUDA(cudaMemset(d_out, 0, sizeof(float)));
            sharedMemBlockReduceKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_in, d_out, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms_shared;
        CHECK_CUDA(cudaEventElapsedTime(&ms_shared, start, stop));
        ms_shared /= nIter;

        /* Report */
        printf("  Array size: %d elements (%.1f MB)\n", N, N * sizeof(float) / (1024.0f * 1024.0f));
        printf("  Block size: %d threads (%d warps per block)\n", blockSize, blockSize / 32);
        printf("  Grid size:  %d blocks\n", gridSize);
        printf("  Iterations: %d\n\n", nIter);
        printf("  ┌──────────────────────────────────────┐\n");
        printf("  │  Method              │  Time (ms)     │\n");
        printf("  ├──────────────────────┼────────────────┤\n");
        printf("  │  Warp shuffle        │  %8.3f       │\n", ms_shuffle);
        printf("  │  Shared memory       │  %8.3f       │\n", ms_shared);
        printf("  ├──────────────────────┼────────────────┤\n");
        printf("  │  Speedup (shfl)      │  %8.2fx      │\n", ms_shared / ms_shuffle);
        printf("  └──────────────────────┴────────────────┘\n\n");

        /* Effective bandwidth */
        float gbytes = N * sizeof(float) / 1.0e9f;
        printf("  Effective bandwidth:\n");
        printf("    Warp shuffle:   %.1f GB/s\n", gbytes / (ms_shuffle * 1e-3f));
        printf("    Shared memory:  %.1f GB/s\n\n", gbytes / (ms_shared * 1e-3f));

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_out));
        free(h_in);
    }

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Key takeaway: warp shuffle reduction avoids shared memory\n");
    printf("  entirely for the intra-warp phase, reducing latency and\n");
    printf("  freeing shared memory for other uses (tiling, etc.).\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}
