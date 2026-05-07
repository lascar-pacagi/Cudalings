/* ===========================================================================
 * Chapter 10: Parallel Reduction -- Max and Argmax
 * ===========================================================================
 *
 * This file demonstrates that the reduction pattern works for ANY
 * associative operator, not just addition. We implement:
 *
 *   1. Max reduction:    find the maximum value in an array
 *   2. Argmax reduction: find the INDEX of the maximum value
 *   3. Generic template: a single kernel that works for sum, max, min, etc.
 *
 * We use the best optimization level (Level 7: grid-stride + warp shuffle)
 * from the main reduction.cu file.
 *
 * Key insight: Reduction works for any ASSOCIATIVE binary operator:
 *   - sum:     (a + b) + c = a + (b + c)
 *   - max:     max(max(a,b), c) = max(a, max(b,c))
 *   - min:     min(min(a,b), c) = min(a, min(b,c))
 *   - product: (a * b) * c = a * (b * c)
 *   - bitwise: (a | b) | c = a | (b | c)     (OR, AND, XOR)
 *
 * The IDENTITY ELEMENT changes with each operator:
 *   - sum:     identity = 0        (x + 0 = x)
 *   - max:     identity = -INF     (max(x, -INF) = x)
 *   - min:     identity = +INF     (min(x, +INF) = x)
 *   - product: identity = 1        (x * 1 = x)
 *
 * Target: Quadro P4200 (CC 6.1, 18 SMs), CUDA 11.7
 * ===========================================================================
 */

#include <cstdio>
#include <cstdlib>
#include <cfloat>
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
 * Warp-level reduction helpers using shuffle
 * ===========================================================================
 *
 * __shfl_down_sync(mask, val, offset):
 *   Lane i receives the value from lane (i + offset).
 *   After 5 rounds (offset = 16, 8, 4, 2, 1), lane 0 holds the result.
 *
 * For MAX:
 *   val = fmaxf(val, __shfl_down_sync(mask, val, 16))
 *   val = fmaxf(val, __shfl_down_sync(mask, val, 8))
 *   ... etc
 *
 * Same tree structure as sum, just replace '+' with 'fmaxf'.
 * ---------------------------------------------------------------------------*/

/* Warp-level max reduction */
__device__ float warpReduceMax(float val)
{
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 8));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 4));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 2));
    val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, 1));
    return val;
}

/* Warp-level sum reduction (for comparison / generic use) */
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
 * KERNEL: Max Reduction (Level 7 -- grid-stride + warp shuffle)
 * ===========================================================================
 *
 *   Grid-stride loop pattern:
 *     t0: max(g[0], g[4], g[8],  ...) -> local_max_0
 *     t1: max(g[1], g[5], g[9],  ...) -> local_max_1
 *     t2: max(g[2], g[6], g[10], ...) -> local_max_2
 *     t3: max(g[3], g[7], g[11], ...) -> local_max_3
 *
 *   Warp shuffle tree (for a 4-thread warp, conceptually):
 *     offset=2: lane[i] = max(lane[i], lane[i+2])
 *     offset=1: lane[i] = max(lane[i], lane[i+1])
 *     lane 0 = max of all
 *
 *   Inter-warp: shared memory to collect warp results, then first warp reduces.
 * ---------------------------------------------------------------------------*/
__global__ void reduce_max(const float *g_idata, float *g_odata, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid      = threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    /* Phase 1: Grid-stride loop -- each thread finds max of its elements */
    float myMax = -FLT_MAX;   /* Identity element for max */
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < (unsigned int)n; i += gridSize) {
        myMax = fmaxf(myMax, g_idata[i]);
    }

    /* Phase 2: Intra-warp reduction via shuffle */
    myMax = warpReduceMax(myMax);

    /* Phase 3: Inter-warp reduction via shared memory */
    int warpId = tid / 32;
    int laneId = tid % 32;

    if (laneId == 0) {
        sdata[warpId] = myMax;
    }
    __syncthreads();

    int numWarps = (blockDim.x + 31) / 32;
    myMax = (tid < (unsigned int)numWarps) ? sdata[tid] : -FLT_MAX;

    if (warpId == 0) {
        myMax = warpReduceMax(myMax);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = myMax;
    }
}

/* ===========================================================================
 * KERNEL: Argmax Reduction (find index of maximum value)
 * ===========================================================================
 *
 * Finding argmax is trickier because we need to carry both the VALUE and
 * the INDEX through the reduction tree.
 *
 * Strategy: each thread maintains a (value, index) pair. At each reduction
 * step, we compare values and keep the pair with the larger value.
 *
 * For warp shuffle, we shuffle both value and index, then compare.
 *
 *   Tree for 4 elements:
 *     t0: (v0, i0)    t1: (v1, i1)    t2: (v2, i2)    t3: (v3, i3)
 *
 *     offset=2:
 *       t0: compare (v0,i0) vs (v2,i2) -> keep the larger
 *       t1: compare (v1,i1) vs (v3,i3) -> keep the larger
 *
 *     offset=1:
 *       t0: compare its pair vs t1's pair -> lane 0 has global argmax
 *
 * ---------------------------------------------------------------------------*/

/* Warp-level argmax reduction */
__device__ void warpReduceArgmax(float &val, int &idx)
{
    /* At each step, shuffle both value and index from lane + offset */
    for (int offset = 16; offset >= 1; offset >>= 1) {
        float otherVal = __shfl_down_sync(0xFFFFFFFF, val, offset);
        int   otherIdx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
        /* Keep the pair with the larger value (ties: keep lower index) */
        if (otherVal > val || (otherVal == val && otherIdx < idx)) {
            val = otherVal;
            idx = otherIdx;
        }
    }
}

__global__ void reduce_argmax(const float *g_idata,
                               float *g_odata_val,
                               int   *g_odata_idx,
                               int n)
{
    /* Shared memory: interleaved (value, index) pairs, one per warp */
    extern __shared__ char shared_raw[];
    float *sval = (float *)shared_raw;
    int   *sidx = (int   *)(shared_raw + ((blockDim.x + 31) / 32) * sizeof(float));

    unsigned int tid      = threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    /* Phase 1: Grid-stride loop -- find local max and its index */
    float myVal = -FLT_MAX;
    int   myIdx = -1;

    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < (unsigned int)n; i += gridSize) {
        float v = g_idata[i];
        if (v > myVal || (v == myVal && (int)i < myIdx)) {
            myVal = v;
            myIdx = (int)i;
        }
    }

    /* Phase 2: Intra-warp reduction */
    warpReduceArgmax(myVal, myIdx);

    /* Phase 3: Inter-warp reduction */
    int warpId = tid / 32;
    int laneId = tid % 32;

    if (laneId == 0) {
        sval[warpId] = myVal;
        sidx[warpId] = myIdx;
    }
    __syncthreads();

    int numWarps = (blockDim.x + 31) / 32;
    if (tid < (unsigned int)numWarps) {
        myVal = sval[tid];
        myIdx = sidx[tid];
    } else {
        myVal = -FLT_MAX;
        myIdx = -1;
    }

    if (warpId == 0) {
        warpReduceArgmax(myVal, myIdx);
    }

    if (tid == 0) {
        g_odata_val[blockIdx.x] = myVal;
        g_odata_idx[blockIdx.x] = myIdx;
    }
}

/* ===========================================================================
 * GENERIC TEMPLATE APPROACH
 * ===========================================================================
 *
 * We can write ONE reduction kernel that works for any operator by using
 * a functor (struct with operator()). The compiler inlines the functor,
 * so there is ZERO runtime overhead compared to the hand-written versions.
 *
 * Example functors:
 *   struct SumOp  { __device__ float operator()(float a, float b) { return a + b; }
 *                   static constexpr float identity = 0.0f; };
 *   struct MaxOp  { __device__ float operator()(float a, float b) { return fmaxf(a,b); }
 *                   static constexpr float identity = -FLT_MAX; };
 *   struct MinOp  { __device__ float operator()(float a, float b) { return fminf(a,b); }
 *                   static constexpr float identity = FLT_MAX; };
 *
 * The kernel is parameterized: reduce_generic<MaxOp><<<...>>>(...)
 * ---------------------------------------------------------------------------*/

/* --- Functor definitions --- */

struct SumOp {
    __device__ __host__ float operator()(float a, float b) const { return a + b; }
    static __device__ __host__ float identity() { return 0.0f; }
};

struct MaxOp {
    __device__ __host__ float operator()(float a, float b) const { return fmaxf(a, b); }
    static __device__ __host__ float identity() { return -FLT_MAX; }
};

struct MinOp {
    __device__ __host__ float operator()(float a, float b) const { return fminf(a, b); }
    static __device__ __host__ float identity() { return FLT_MAX; }
};

struct ProductOp {
    __device__ __host__ float operator()(float a, float b) const { return a * b; }
    static __device__ __host__ float identity() { return 1.0f; }
};

/* --- Generic warp-level reduction --- */
template <typename Op>
__device__ float warpReduceGeneric(float val, Op op)
{
    val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 16));
    val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 8));
    val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 4));
    val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 2));
    val = op(val, __shfl_down_sync(0xFFFFFFFF, val, 1));
    return val;
}

/* --- Generic reduction kernel (Level 7 pattern) --- */
/*
 *   Generic tree diagram:
 *
 *   Grid-stride loop:
 *     t[i]: fold g[i], g[i+gridSize], g[i+2*gridSize], ...
 *            using op() with identity as initial value
 *
 *   Warp shuffle reduction:
 *     offset=16: lane[i] = op(lane[i], lane[i+16])
 *     offset=8:  lane[i] = op(lane[i], lane[i+8])
 *     ...
 *     offset=1:  lane[i] = op(lane[i], lane[i+1])
 *     -> lane 0 has warp result
 *
 *   Inter-warp (shared memory):
 *     warp 0's lane 0 writes to sdata[0]
 *     warp 1's lane 0 writes to sdata[1]
 *     ...
 *     First warp reduces sdata[0..numWarps-1]
 *     -> thread 0 has block result
 */
template <typename Op>
__global__ void reduce_generic(const float *g_idata, float *g_odata,
                                int n, Op op)
{
    extern __shared__ float sdata[];

    unsigned int tid      = threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    /* Phase 1: Grid-stride accumulation */
    float myVal = Op::identity();
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < (unsigned int)n; i += gridSize) {
        myVal = op(myVal, g_idata[i]);
    }

    /* Phase 2: Intra-warp shuffle reduction */
    myVal = warpReduceGeneric(myVal, op);

    /* Phase 3: Inter-warp shared memory reduction */
    int warpId = tid / 32;
    int laneId = tid % 32;

    if (laneId == 0) {
        sdata[warpId] = myVal;
    }
    __syncthreads();

    int numWarps = (blockDim.x + 31) / 32;
    myVal = (tid < (unsigned int)numWarps) ? sdata[tid] : Op::identity();

    if (warpId == 0) {
        myVal = warpReduceGeneric(myVal, op);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = myVal;
    }
}

/* ===========================================================================
 * CPU reference implementations
 * ===========================================================================*/
float cpu_max(const float *data, int n)
{
    float mx = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (data[i] > mx) mx = data[i];
    }
    return mx;
}

int cpu_argmax(const float *data, int n)
{
    float mx = -FLT_MAX;
    int   mi = -1;
    for (int i = 0; i < n; i++) {
        if (data[i] > mx) {
            mx = data[i];
            mi = i;
        }
    }
    return mi;
}

float cpu_min(const float *data, int n)
{
    float mn = FLT_MAX;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
    }
    return mn;
}

double cpu_sum(const float *data, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += data[i];
    return s;
}

/* ===========================================================================
 * Main
 * ===========================================================================*/

#define N          (1 << 22)  /* ~4M elements */
#define BLOCK_SIZE 256
#define NUM_ITERS  100

int main()
{
    printf("=============================================================\n");
    printf(" Chapter 10: Reduction for Max, Argmax, and Generic Operators\n");
    printf("=============================================================\n");
    printf("Array size: %d elements (%.1f MB)\n\n", N, N * sizeof(float) / 1e6);

    /* -------------------------------------------------------------------
     * Initialize data with some known pattern
     * -------------------------------------------------------------------*/
    float *h_data = (float *)malloc(N * sizeof(float));
    srand(123);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)rand() / (float)RAND_MAX * 100.0f - 50.0f;
        /* values in [-50, 50) */
    }

    /* Plant a known maximum at a specific location for verification */
    int planted_idx = N / 3 + 7;
    h_data[planted_idx] = 999.0f;

    /* CPU references */
    float ref_max = cpu_max(h_data, N);
    int   ref_argmax = cpu_argmax(h_data, N);
    float ref_min = cpu_min(h_data, N);
    double ref_sum = cpu_sum(h_data, N);

    printf("CPU reference:\n");
    printf("  max    = %.2f  (at index %d)\n", ref_max, ref_argmax);
    printf("  min    = %.2f\n", ref_min);
    printf("  sum    = %.2f\n\n", ref_sum);

    /* -------------------------------------------------------------------
     * Device allocation
     * -------------------------------------------------------------------*/
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    int numSMs = 18;
    int numBlocks = numSMs * 2;  /* 36 blocks for grid-stride kernels */

    float *d_partial_val;
    int   *d_partial_idx;
    CHECK_CUDA(cudaMalloc(&d_partial_val, numBlocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_partial_idx, numBlocks * sizeof(int)));

    int smemSize = ((BLOCK_SIZE + 31) / 32) * sizeof(float);

    /* CUDA events */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* ===================================================================
     * TEST 1: Max Reduction (dedicated kernel)
     * ===================================================================*/
    printf("--- Test 1: Max Reduction (dedicated kernel) ---\n");
    {
        /* Warmup */
        reduce_max<<<numBlocks, BLOCK_SIZE, smemSize>>>(d_data, d_partial_val, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        /* Time it */
        float elapsed;
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < NUM_ITERS; i++) {
            reduce_max<<<numBlocks, BLOCK_SIZE, smemSize>>>(d_data, d_partial_val, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        elapsed /= NUM_ITERS;

        /* Reduce partial results on CPU */
        float *h_partial = (float *)malloc(numBlocks * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_partial, d_partial_val, numBlocks * sizeof(float),
                               cudaMemcpyDeviceToHost));
        float gpu_max = -FLT_MAX;
        for (int i = 0; i < numBlocks; i++) {
            gpu_max = fmaxf(gpu_max, h_partial[i]);
        }
        free(h_partial);

        float bw = (N * sizeof(float)) / (elapsed * 1e6);
        printf("  GPU max = %.2f  (expected %.2f)  %s\n",
               gpu_max, ref_max, (gpu_max == ref_max) ? "PASS" : "FAIL");
        printf("  Time: %.4f ms, BW: %.2f GB/s\n\n", elapsed, bw);
    }

    /* ===================================================================
     * TEST 2: Argmax Reduction
     * ===================================================================*/
    printf("--- Test 2: Argmax Reduction ---\n");
    {
        /* Shared memory: float values + int indices for each warp */
        int numWarps = (BLOCK_SIZE + 31) / 32;
        int smemArgmax = numWarps * sizeof(float) + numWarps * sizeof(int);

        /* Warmup */
        reduce_argmax<<<numBlocks, BLOCK_SIZE, smemArgmax>>>(
            d_data, d_partial_val, d_partial_idx, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        /* Time it */
        float elapsed;
        CHECK_CUDA(cudaEventRecord(start));
        for (int i = 0; i < NUM_ITERS; i++) {
            reduce_argmax<<<numBlocks, BLOCK_SIZE, smemArgmax>>>(
                d_data, d_partial_val, d_partial_idx, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        elapsed /= NUM_ITERS;

        /* Reduce partial results on CPU */
        float *h_pval = (float *)malloc(numBlocks * sizeof(float));
        int   *h_pidx = (int   *)malloc(numBlocks * sizeof(int));
        CHECK_CUDA(cudaMemcpy(h_pval, d_partial_val, numBlocks * sizeof(float),
                               cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_pidx, d_partial_idx, numBlocks * sizeof(int),
                               cudaMemcpyDeviceToHost));

        float gpu_max = -FLT_MAX;
        int   gpu_idx = -1;
        for (int i = 0; i < numBlocks; i++) {
            if (h_pval[i] > gpu_max) {
                gpu_max = h_pval[i];
                gpu_idx = h_pidx[i];
            }
        }
        free(h_pval);
        free(h_pidx);

        float bw = (N * sizeof(float)) / (elapsed * 1e6);
        printf("  GPU max = %.2f at index %d  (expected %.2f at %d)  %s\n",
               gpu_max, gpu_idx, ref_max, ref_argmax,
               (gpu_max == ref_max && gpu_idx == ref_argmax) ? "PASS" : "FAIL");
        printf("  Time: %.4f ms, BW: %.2f GB/s\n\n", elapsed, bw);
    }

    /* ===================================================================
     * TEST 3: Generic Template -- Max, Min, Sum all from one kernel
     * ===================================================================*/
    printf("--- Test 3: Generic Template Reduction ---\n");
    {
        float *h_partial = (float *)malloc(numBlocks * sizeof(float));

        /* --- Generic Max --- */
        reduce_generic<<<numBlocks, BLOCK_SIZE, smemSize>>>(d_data, d_partial_val, N, MaxOp());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_partial, d_partial_val, numBlocks * sizeof(float),
                               cudaMemcpyDeviceToHost));
        float gmax = -FLT_MAX;
        for (int i = 0; i < numBlocks; i++) gmax = fmaxf(gmax, h_partial[i]);
        printf("  Generic Max: %.2f  (expected %.2f)  %s\n",
               gmax, ref_max, (gmax == ref_max) ? "PASS" : "FAIL");

        /* --- Generic Min --- */
        reduce_generic<<<numBlocks, BLOCK_SIZE, smemSize>>>(d_data, d_partial_val, N, MinOp());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_partial, d_partial_val, numBlocks * sizeof(float),
                               cudaMemcpyDeviceToHost));
        float gmin = FLT_MAX;
        for (int i = 0; i < numBlocks; i++) gmin = fminf(gmin, h_partial[i]);
        printf("  Generic Min: %.2f  (expected %.2f)  %s\n",
               gmin, ref_min, (gmin == ref_min) ? "PASS" : "FAIL");

        /* --- Generic Sum --- */
        reduce_generic<<<numBlocks, BLOCK_SIZE, smemSize>>>(d_data, d_partial_val, N, SumOp());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_partial, d_partial_val, numBlocks * sizeof(float),
                               cudaMemcpyDeviceToHost));
        double gsum = 0.0;
        for (int i = 0; i < numBlocks; i++) gsum += h_partial[i];
        printf("  Generic Sum: %.2f  (expected %.2f)  %s\n",
               gsum, ref_sum, (fabs(gsum - ref_sum) < 100.0) ? "PASS" : "FAIL");
        /* Note: float sum of millions of values has limited precision,
         * so we allow a generous tolerance for the sum check. */

        free(h_partial);
    }

    printf("\n--- Summary ---\n");
    printf("The generic template kernel produces the same results as the\n");
    printf("dedicated kernels, with zero overhead -- the functor is inlined\n");
    printf("by the compiler. This is the power of C++ templates on the GPU.\n");

    /* -------------------------------------------------------------------
     * Cleanup
     * -------------------------------------------------------------------*/
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_partial_val));
    CHECK_CUDA(cudaFree(d_partial_idx));
    free(h_data);

    return 0;
}
