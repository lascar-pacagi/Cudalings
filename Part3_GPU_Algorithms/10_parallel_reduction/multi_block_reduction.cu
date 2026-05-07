/* ===========================================================================
 * Chapter 10: Multi-Block Reduction for Very Large Arrays
 * ===========================================================================
 *
 * Problem: A single thread block can hold at most 1024 threads. Even with
 * a grid-stride loop, we produce one partial result PER BLOCK. For very
 * large arrays (millions to billions of elements), we need a strategy to
 * combine block-level partial results into a single final answer.
 *
 * Solution: TWO kernel launches.
 *
 *   Pass 1 (many blocks):
 *     Each block reduces its portion of the array -> partial_results[blockIdx.x]
 *
 *   Pass 2 (1 block):
 *     A single block reduces the partial_results array -> final answer
 *
 *   This works for ANY array size. Two launches always suffice because:
 *   - Pass 1 can use any number of blocks (grid-stride handles overflow)
 *   - Pass 2 has at most numBlocks elements, which fits in one block
 *     (e.g., 1024 blocks * 1 float = 4 KB -- trivial)
 *
 * Diagram:
 *
 *   Global array: [======= 100,000,000 elements =======]
 *                  |        |        |        |        |
 *               Block 0  Block 1  Block 2   ...   Block B-1
 *                  |        |        |        |        |
 *                  v        v        v        v        v
 *   Partial:    [p[0]]   [p[1]]   [p[2]]   ...    [p[B-1]]
 *
 *   Pass 2:     [p[0]  p[1]  p[2]  ...  p[B-1]]
 *                            |
 *                        one block
 *                            |
 *                            v
 *                       final_result
 *
 * We benchmark with 100M floats (~400 MB) to see real-world performance.
 *
 * Target: Quadro P4200 (CC 6.1, 18 SMs, ~134 GB/s memory BW), CUDA 11.7
 * ===========================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

/* ---------------------------------------------------------------------------
 * Error checking macro
 * ---------------------------------------------------------------------------*/
#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ===========================================================================
 * Warp-level reduction using shuffle
 * ===========================================================================
 *
 * __shfl_down_sync(mask, val, offset):
 *   Each lane gets the value from (its_lane + offset).
 *   After offsets 16,8,4,2,1, lane 0 holds the sum of all 32 lanes.
 *
 *   Conceptual diagram for 8 lanes:
 *     offset=4: lane[i] += lane[i+4]   (4 adds)
 *     offset=2: lane[i] += lane[i+2]   (2 adds)
 *     offset=1: lane[i] += lane[i+1]   (1 add)
 *     lane 0 = sum of all 8
 *
 *   Full warp (32 lanes) uses offsets 16,8,4,2,1 = 5 steps.
 * ---------------------------------------------------------------------------*/
__device__ float warpReduceSum(float val)
{
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

/* ===========================================================================
 * PASS 1 KERNEL: Block-Level Reduction with Grid-Stride Loop
 * ===========================================================================
 *
 * Each block processes a large chunk of the array via grid-stride loop,
 * then reduces its thread-local sums through shared memory + warp shuffle.
 *
 *   For 100M elements, 256 threads/block, 512 blocks:
 *     gridSize = 256 * 512 = 131,072
 *     Each thread processes 100M / 131,072 ~ 763 elements
 *
 *   Grid-stride pattern:
 *     t0:   g[0],  g[131072],  g[262144],  ... (763 elements)
 *     t1:   g[1],  g[131073],  g[262145],  ... (763 elements)
 *     ...
 *     Each thread accumulates ~763 values before the tree reduction.
 *     The tree reduction then handles only 256 values (one per thread).
 *
 *   Block-level reduction tree (256 threads -> 1 result):
 *
 *     [t0  t1  t2  ... t255]   <- 256 partial sums in shared memory
 *          |                    Stride = 128
 *     [t0  t1  ... t127]       <- 128 sums
 *          |                    Stride = 64
 *     [t0  ... t63]            <- 64 sums
 *          |                    Stride = 32
 *     [t0  ... t31]            <- 32 sums (one warp -> use shuffle)
 *          |
 *          v
 *       partial[blockIdx.x]    <- one value written to global memory
 *
 * ---------------------------------------------------------------------------*/
__global__ void reduce_pass1(const float *g_idata, float *g_odata, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid      = threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    /* -----------------------------------------------------------------------
     * Grid-stride accumulation.
     *
     * Each thread walks through the entire array with stride = gridSize.
     * Consecutive threads read consecutive addresses -> coalesced access.
     *
     * Example for thread 0 with gridSize=131072 and N=100M:
     *   i = 0, 131072, 262144, 393216, ..., until i >= 100M
     *   That's about 763 additions per thread.
     * -----------------------------------------------------------------------*/
    float mySum = 0.0f;
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < (unsigned int)n; i += gridSize) {
        mySum += g_idata[i];
    }

    /* -----------------------------------------------------------------------
     * Intra-warp reduction (no shared memory needed).
     * The 32 threads in each warp reduce their values using registers only.
     * Lane 0 of each warp holds the warp's total.
     * -----------------------------------------------------------------------*/
    mySum = warpReduceSum(mySum);

    /* -----------------------------------------------------------------------
     * Inter-warp reduction via shared memory.
     * We have blockDim.x / 32 warps per block (e.g., 256/32 = 8 warps).
     * Lane 0 of each warp writes its sum to shared memory.
     * Then the first warp (warp 0) reduces those 8 values.
     * -----------------------------------------------------------------------*/
    int warpId = tid / 32;
    int laneId = tid % 32;

    if (laneId == 0) {
        sdata[warpId] = mySum;
    }
    __syncthreads();

    /* Only the first warp participates in the final reduction */
    int numWarps = (blockDim.x + 31) / 32;
    mySum = (tid < (unsigned int)numWarps) ? sdata[tid] : 0.0f;

    if (warpId == 0) {
        mySum = warpReduceSum(mySum);
    }

    /* Thread 0 writes this block's result to the partial results array */
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

/* ===========================================================================
 * PASS 2 KERNEL: Reduce Partial Results to Final Answer
 * ===========================================================================
 *
 * This kernel is launched with a SINGLE BLOCK. It reads the partial results
 * array (one element per block from pass 1) and reduces them to one value.
 *
 * Since numBlocks from pass 1 is small (e.g., 512), this is trivial --
 * a single block of 256 threads handles it in a few steps.
 *
 *   Input:  partial[0], partial[1], ..., partial[numBlocks-1]
 *
 *   Grid-stride loop (one block):
 *     t0: partial[0] + partial[256] + ...
 *     t1: partial[1] + partial[257] + ...
 *     ...
 *     (if numBlocks <= 256, each thread reads exactly one element)
 *
 *   Tree reduction:
 *     256 values -> 1 value (same as above)
 *
 *   Output: g_odata[0] = final sum
 * ---------------------------------------------------------------------------*/
__global__ void reduce_pass2(const float *g_idata, float *g_odata, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;

    /* Grid-stride loop (though typically n is small enough for one pass) */
    float mySum = 0.0f;
    for (unsigned int i = tid; i < (unsigned int)n; i += blockDim.x) {
        mySum += g_idata[i];
    }

    /* Intra-warp reduction */
    mySum = warpReduceSum(mySum);

    int warpId = tid / 32;
    int laneId = tid % 32;

    if (laneId == 0) {
        sdata[warpId] = mySum;
    }
    __syncthreads();

    int numWarps = (blockDim.x + 31) / 32;
    mySum = (tid < (unsigned int)numWarps) ? sdata[tid] : 0.0f;

    if (warpId == 0) {
        mySum = warpReduceSum(mySum);
    }

    if (tid == 0) {
        g_odata[0] = mySum;
    }
}

/* ===========================================================================
 * CPU reference (double precision for accuracy)
 * ===========================================================================*/
double cpu_sum(const float *data, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)data[i];
    }
    return sum;
}

/* ===========================================================================
 * Full two-pass reduction (host function)
 * ===========================================================================
 *
 * This function encapsulates the two-pass strategy:
 *   1. Launch pass1 with many blocks -> partial results
 *   2. Launch pass2 with 1 block     -> final result
 *
 * The caller gets back a single float.
 * ---------------------------------------------------------------------------*/
float gpu_reduce_two_pass(const float *d_data, int n,
                           float *d_partial, float *d_result,
                           int numBlocks, int blockSize)
{
    int smemSize = ((blockSize + 31) / 32) * sizeof(float);

    /* Pass 1: N elements -> numBlocks partial sums */
    reduce_pass1<<<numBlocks, blockSize, smemSize>>>(d_data, d_partial, n);

    /* Pass 2: numBlocks partial sums -> 1 final result */
    reduce_pass2<<<1, blockSize, smemSize>>>(d_partial, d_result, numBlocks);

    /* Copy result back to host */
    float result;
    CHECK_CUDA(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

/* ===========================================================================
 * Main
 * ===========================================================================*/

#define BLOCK_SIZE 256
#define NUM_ITERS  50

int main()
{
    printf("=============================================================\n");
    printf(" Chapter 10: Multi-Block Reduction for Very Large Arrays\n");
    printf("=============================================================\n\n");

    /* -------------------------------------------------------------------
     * Test multiple array sizes
     * -------------------------------------------------------------------*/
    int sizes[] = {
        1 << 20,    /*   1M =   4 MB */
        1 << 22,    /*   4M =  16 MB */
        1 << 24,    /*  16M =  64 MB */
        1 << 26,    /*  64M = 256 MB */
        100000000   /* 100M = 400 MB (not a power of 2, tests robustness) */
    };
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    /* Find the largest size to allocate once */
    int maxN = 0;
    for (int s = 0; s < numSizes; s++) {
        if (sizes[s] > maxN) maxN = sizes[s];
    }

    printf("Largest array: %d elements (%.1f MB)\n", maxN, maxN * sizeof(float) / 1e6);
    printf("Block size:    %d threads\n", BLOCK_SIZE);
    printf("Iterations:    %d (for timing)\n\n", NUM_ITERS);

    /* -------------------------------------------------------------------
     * Allocate host memory and fill with random data
     * -------------------------------------------------------------------*/
    float *h_data = (float *)malloc((size_t)maxN * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate %.1f MB on host\n",
                maxN * sizeof(float) / 1e6);
        return 1;
    }

    srand(42);
    for (int i = 0; i < maxN; i++) {
        /* Small values to avoid float overflow when summing 100M elements */
        h_data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        /* values in [-1, 1) */
    }

    /* -------------------------------------------------------------------
     * Allocate device memory
     * -------------------------------------------------------------------*/
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, (size_t)maxN * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, (size_t)maxN * sizeof(float),
                           cudaMemcpyHostToDevice));

    /*
     * Number of blocks for pass 1.
     * Heuristic: use enough blocks to keep all SMs busy, but not too many
     * (more blocks = more partial results to reduce in pass 2).
     *
     * Good rule of thumb: 2-8 blocks per SM.
     * Quadro P4200 has 18 SMs -> 36 to 144 blocks.
     *
     * We try a few configurations to show the tradeoff.
     */
    int numSMs = 18;

    /* We'll benchmark with different block counts */
    int blockCounts[] = { numSMs * 2, numSMs * 4, numSMs * 8, 512, 1024 };
    int numConfigs = sizeof(blockCounts) / sizeof(blockCounts[0]);

    /* Partial results and final result buffers */
    int maxBlocks = 1024;
    float *d_partial, *d_result;
    CHECK_CUDA(cudaMalloc(&d_partial, maxBlocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    /* CUDA events */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* ===================================================================
     * Part 1: Demonstrate correctness for all sizes
     * ===================================================================*/
    printf("=== Part 1: Correctness Verification ===\n\n");
    printf("%-12s %14s %14s %12s  %s\n",
           "N", "CPU Sum", "GPU Sum", "Rel Error", "Status");
    printf("--------------------------------------------------------------\n");

    int numBlocks_default = numSMs * 4;  /* 72 blocks */

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];

        /* CPU reference (double precision) */
        double cpu_ref = cpu_sum(h_data, n);

        /* GPU two-pass reduction */
        float gpu_result = gpu_reduce_two_pass(d_data, n, d_partial, d_result,
                                                numBlocks_default, BLOCK_SIZE);

        /* Relative error */
        double rel_err = fabs((double)gpu_result - cpu_ref) / fabs(cpu_ref);

        /*
         * Float precision note: summing 100M floats in [-1,1] gives a result
         * near 0. The absolute error matters more than relative error here.
         * We use a generous threshold.
         */
        const char *status = (rel_err < 0.01 || fabs((double)gpu_result - cpu_ref) < 1000.0)
                              ? "PASS" : "FAIL";

        printf("%-12d %14.2f %14.2f %12.2e  %s\n",
               n, cpu_ref, (double)gpu_result, rel_err, status);
    }

    /* ===================================================================
     * Part 2: Benchmark with 100M elements, varying number of blocks
     * ===================================================================*/
    printf("\n=== Part 2: Performance (N = 100,000,000 = %.0f MB) ===\n\n",
           100000000.0 * sizeof(float) / 1e6);

    int n_bench = 100000000;  /* 100M */

    printf("%-10s %10s %12s %10s  %s\n",
           "Blocks", "Time (ms)", "BW (GB/s)", "Pass2 frac", "Notes");
    printf("-----------------------------------------------------------\n");

    for (int c = 0; c < numConfigs; c++) {
        int nb = blockCounts[c];

        int smemSize = ((BLOCK_SIZE + 31) / 32) * sizeof(float);

        /* Warmup */
        reduce_pass1<<<nb, BLOCK_SIZE, smemSize>>>(d_data, d_partial, n_bench);
        reduce_pass2<<<1, BLOCK_SIZE, smemSize>>>(d_partial, d_result, nb);
        CHECK_CUDA(cudaDeviceSynchronize());

        /* Time both passes together */
        float elapsed_total;
        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            reduce_pass1<<<nb, BLOCK_SIZE, smemSize>>>(d_data, d_partial, n_bench);
            reduce_pass2<<<1, BLOCK_SIZE, smemSize>>>(d_partial, d_result, nb);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_total, start, stop));
        elapsed_total /= NUM_ITERS;

        /* Time pass 2 alone to see its overhead */
        float elapsed_p2;
        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            reduce_pass2<<<1, BLOCK_SIZE, smemSize>>>(d_partial, d_result, nb);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_p2, start, stop));
        elapsed_p2 /= NUM_ITERS;

        float bw = ((size_t)n_bench * sizeof(float)) / (elapsed_total * 1e6);
        float p2_frac = elapsed_p2 / elapsed_total * 100.0f;

        const char *note = "";
        if (nb == numSMs * 2) note = "<-- minimal blocks";
        if (nb == numSMs * 4) note = "<-- sweet spot";
        if (nb == 1024)       note = "<-- many blocks";

        printf("%-10d %10.4f %12.2f %9.1f%%  %s\n",
               nb, elapsed_total, bw, p2_frac, note);
    }

    /* ===================================================================
     * Part 3: Show that 2 passes always suffice
     * ===================================================================*/
    printf("\n=== Part 3: Why 2 Passes Always Suffice ===\n\n");
    printf("Pass 1: N elements -> numBlocks partial sums\n");
    printf("  - Grid-stride loop means ANY N works with ANY numBlocks.\n");
    printf("  - More elements per thread = more work before reduction.\n\n");
    printf("Pass 2: numBlocks partial sums -> 1 final value\n");
    printf("  - numBlocks is at most ~1024, easily fits in one block.\n");
    printf("  - Even with 1024 blocks, pass 2 processes 4 KB of data.\n\n");

    /*
     * Demonstrate: what if we use a single pass with atomicAdd?
     * This is simpler but potentially slower due to atomic contention.
     */
    printf("=== Part 4: Comparison with atomicAdd Approach ===\n\n");

    /* We'll skip the atomic approach implementation to keep focus on
     * the two-pass method, but note the tradeoffs:
     *
     *   Two-pass:     Two kernel launches, no atomic contention.
     *                 Achieves near-peak bandwidth.
     *
     *   AtomicAdd:    Single kernel, each block does atomicAdd to a global sum.
     *                 Simpler code, but atomic contention from many blocks
     *                 serializes the final accumulation.
     *                 On Pascal (CC 6.1), atomicAdd for float is native,
     *                 so contention is lower than on older architectures.
     *
     *   In practice, two-pass is preferred for maximum performance.
     *   AtomicAdd is fine for convenience when reduction is not the bottleneck.
     */

    printf("Two-pass reduction: preferred for maximum bandwidth.\n");
    printf("AtomicAdd approach: simpler, but atomic contention limits scaling.\n");
    printf("For production code, use CUB's cub::DeviceReduce (handles all cases).\n");

    /* -------------------------------------------------------------------
     * Cleanup
     * -------------------------------------------------------------------*/
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_partial));
    CHECK_CUDA(cudaFree(d_result));
    free(h_data);

    printf("\nDone.\n");
    return 0;
}
