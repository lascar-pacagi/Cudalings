/* ===========================================================================
 * Chapter 10: Parallel Reduction -- All 7 Optimization Levels
 * ===========================================================================
 *
 * This file implements the classic Mark Harris / NVIDIA parallel reduction
 * case study. We progress through 7 optimization levels, each fixing a
 * performance problem from the previous one.
 *
 * Target: Quadro P4200 (CC 6.1, 18 SMs), CUDA 11.7
 *
 * The reduction operation: sum N floats into a single value.
 * This is memory-bound -- peak performance = peak memory bandwidth.
 *
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
 * KERNEL 1: Interleaved Addressing with Divergent Branching
 * ===========================================================================
 *
 * The simplest possible tree reduction. Each thread decides whether to
 * participate based on:  if (tid % (2 * stride) == 0)
 *
 * Tree diagram for 8 elements (one block, 8 threads):
 *
 *   Shared mem:  [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]
 *
 *   Step 1 (stride=1):
 *     t0: s[0] += s[1]     t2: s[2] += s[3]     t4: s[4] += s[5]     t6: s[6] += s[7]
 *          |                     |                     |                     |
 *   Result: [a0+a1]  [a1]  [a2+a3]  [a3]  [a4+a5]  [a5]  [a6+a7]  [a7]
 *
 *   Step 2 (stride=2):
 *     t0: s[0] += s[2]                           t4: s[4] += s[6]
 *          |                                           |
 *   Result: [sum0..3]  ...                    [sum4..7]  ...
 *
 *   Step 3 (stride=4):
 *     t0: s[0] += s[4]
 *          |
 *   Result: [sum0..7]   <-- final answer
 *
 * PROBLEM: The modulo test causes warp divergence. In step 1, threads
 * 0,2,4,6 are active and threads 1,3,5,7 are idle within the SAME warp.
 * Both paths serialize, halving throughput.
 * ---------------------------------------------------------------------------*/
__global__ void reduce_interleaved_divergent(const float *g_idata,
                                              float *g_odata, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    /* Load one element from global memory into shared memory */
    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    /* Tree reduction with interleaved addressing */
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        /* DIVERGENT: threads with tid % (2*stride) != 0 do nothing */
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    /* Thread 0 writes this block's result */
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/* ===========================================================================
 * KERNEL 2: Interleaved Addressing, Non-Divergent
 * ===========================================================================
 *
 * Fix the divergence by computing a strided index so that the FIRST N/2
 * threads are active (contiguous within a warp), not every other thread.
 *
 *   Step 1 (stride=1):
 *     tid=0 -> idx=0: s[0] += s[1]
 *     tid=1 -> idx=2: s[2] += s[3]
 *     tid=2 -> idx=4: s[4] += s[5]
 *     tid=3 -> idx=6: s[6] += s[7]
 *     (tids 0-3 active, 4-7 idle -- less divergence within warps)
 *
 *   Step 2 (stride=2):
 *     tid=0 -> idx=0: s[0] += s[2]
 *     tid=1 -> idx=4: s[4] += s[6]
 *
 *   Step 3 (stride=4):
 *     tid=0 -> idx=0: s[0] += s[4]
 *
 * PROBLEM: The interleaved access pattern (indices 0,2,4,6) can cause
 * shared memory BANK CONFLICTS on some architectures.
 * ---------------------------------------------------------------------------*/
__global__ void reduce_interleaved_bank_free(const float *g_idata,
                                              float *g_odata, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    /* Compute index to avoid branch divergence */
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < (int)blockDim.x) {
            sdata[index] += sdata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/* ===========================================================================
 * KERNEL 3: Sequential Addressing
 * ===========================================================================
 *
 * Reverse the loop: start with stride = blockDim/2, halve each step.
 * Now the FIRST `stride` threads are active -- they are contiguous, so
 * NO warp divergence (within a warp, all threads take the same branch).
 * Sequential addressing also eliminates bank conflicts.
 *
 *   Shared mem:  [a0] [a1] [a2] [a3] [a4] [a5] [a6] [a7]
 *
 *   Step 1 (stride=4):
 *     t0: s[0] += s[4]   t1: s[1] += s[5]   t2: s[2] += s[6]   t3: s[3] += s[7]
 *
 *   Result: [a0+a4] [a1+a5] [a2+a6] [a3+a7]  [a4]  [a5]  [a6]  [a7]
 *
 *   Step 2 (stride=2):
 *     t0: s[0] += s[2]   t1: s[1] += s[3]
 *
 *   Result: [sum0246] [sum1357]  ...
 *
 *   Step 3 (stride=1):
 *     t0: s[0] += s[1]
 *
 *   Result: [total]  <-- final answer
 *
 * This is a clean, bank-conflict-free, divergence-free reduction.
 * But we still launch N threads per block and half are idle from step 1.
 * ---------------------------------------------------------------------------*/
__global__ void reduce_sequential(const float *g_idata,
                                   float *g_odata, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    /* Sequential addressing: stride starts at half block, shrinks */
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/* ===========================================================================
 * KERNEL 4: First Add During Load
 * ===========================================================================
 *
 * Observation: in Kernel 3, half the threads are idle from step 1.
 * We launched them just to load data, then threw them away.
 *
 * Fix: each thread loads TWO elements and adds them during the load.
 * We launch half as many blocks, but each block covers twice the data.
 *
 *   Global:  [a0  a1  a2  a3 | a4  a5  a6  a7]
 *               block loads (half the threads):
 *     t0: s[0] = g[0] + g[4] = a0+a4
 *     t1: s[1] = g[1] + g[5] = a1+a5
 *     t2: s[2] = g[2] + g[6] = a2+a6
 *     t3: s[3] = g[3] + g[7] = a3+a7
 *
 *   Shared:  [a0+a4] [a1+a5] [a2+a6] [a3+a7]
 *
 *   Then sequential reduction on 4 elements (same as Kernel 3):
 *     Step 1 (stride=2): t0: s[0]+=s[2], t1: s[1]+=s[3]
 *     Step 2 (stride=1): t0: s[0]+=s[1]
 *
 * The first addition is "free" -- it happens during the global memory load,
 * which is the bottleneck anyway. We halve the number of blocks.
 * ---------------------------------------------------------------------------*/
__global__ void reduce_first_add_during_load(const float *g_idata,
                                              float *g_odata, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    /* Each block processes 2 * blockDim.x elements */
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    /* Load two elements and add them (first reduction step is free) */
    float mySum = 0.0f;
    if (i < n)                  mySum  = g_idata[i];
    if (i + blockDim.x < n)    mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    /* Sequential addressing reduction on the remaining elements */
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/* ===========================================================================
 * KERNEL 5: Unroll Last Warp
 * ===========================================================================
 *
 * When stride <= 32 (warp size), all remaining active threads are in ONE
 * warp. Threads in a warp execute in lockstep (SIMD), so __syncthreads()
 * is unnecessary for the last 5 steps (stride 32,16,8,4,2,1).
 *
 * We unroll the last warp into straight-line code:
 *
 *   if (tid < 32) {
 *       volatile float *smem = sdata;    <-- volatile prevents register caching
 *       if (blockDim.x >= 64)  smem[tid] += smem[tid + 32];
 *       smem[tid] += smem[tid + 16];
 *       smem[tid] += smem[tid + 8];
 *       smem[tid] += smem[tid + 4];
 *       smem[tid] += smem[tid + 2];
 *       smem[tid] += smem[tid + 1];
 *   }
 *
 * The `volatile` keyword forces every write to go through to shared memory
 * (not cached in a register). This is critical because other threads in the
 * warp read the updated value on the very next instruction.
 *
 * NOTE: On CC >= 7.0 (Volta+), independent thread scheduling means warps
 * are NOT guaranteed lockstep. Use __syncwarp() there. On our CC 6.1
 * (Pascal), warps ARE lockstep, so volatile is sufficient.
 *
 *   Tree for the last warp (32 elements in shared memory):
 *
 *   stride=32: t0..t31 all active
 *     t[i]: s[i] += s[i+32]   (merge upper 32 into lower 32)
 *
 *   stride=16: t0..t15 (but we let all 32 run -- extra adds are harmless)
 *     t[i]: s[i] += s[i+16]
 *
 *   stride=8:  t[i]: s[i] += s[i+8]
 *   stride=4:  t[i]: s[i] += s[i+4]
 *   stride=2:  t[i]: s[i] += s[i+2]
 *   stride=1:  t[i]: s[i] += s[i+1]
 *
 *   s[0] = total
 * ---------------------------------------------------------------------------*/

/* Helper: unroll the final warp using volatile shared memory */
__device__ void warpReduce_volatile(volatile float *sdata, unsigned int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce_unroll_last_warp(const float *g_idata,
                                         float *g_odata, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    /* First add during load (from Kernel 4) */
    float mySum = 0.0f;
    if (i < n)                  mySum  = g_idata[i];
    if (i + blockDim.x < n)    mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    /* Tree reduction with __syncthreads -- stop at stride > 32 */
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    /* Unroll the last warp -- no __syncthreads needed */
    if (tid < 32) {
        warpReduce_volatile(sdata, tid);
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/* ===========================================================================
 * KERNEL 6: Completely Unrolled (Template on Block Size)
 * ===========================================================================
 *
 * If the block size is known at compile time, every `if (blockSize >= X)`
 * is a compile-time constant. The compiler eliminates dead branches entirely.
 * The result is a fully unrolled reduction with ZERO loop overhead.
 *
 *   template <unsigned int blockSize>
 *   __global__ void reduce6(...) {
 *       ...
 *       if (blockSize >= 512) { if (tid < 256) s[tid] += s[tid+256]; sync; }
 *       if (blockSize >= 256) { if (tid < 128) s[tid] += s[tid+128]; sync; }
 *       if (blockSize >= 128) { if (tid <  64) s[tid] += s[tid+ 64]; sync; }
 *       // warp unroll for last 32 ...
 *   }
 *
 *   For blockSize=256, the compiler sees:
 *       if (256 >= 512) -> FALSE -> eliminated
 *       if (256 >= 256) -> TRUE  -> kept (s[tid]+=s[tid+128])
 *       if (256 >= 128) -> TRUE  -> kept (s[tid]+=s[tid+64])
 *       // warp unroll
 *
 * No loop, no loop counter, no branch overhead at runtime.
 * ---------------------------------------------------------------------------*/
template <unsigned int blockSize>
__global__ void reduce_completely_unrolled(const float *g_idata,
                                            float *g_odata, int n)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    /* First add during load */
    float mySum = 0.0f;
    if (i < n)                  mySum  = g_idata[i];
    if (i + blockDim.x < n)    mySum += g_idata[i + blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    /* Fully unrolled tree reduction -- compile-time branch elimination */
    if (blockSize >= 1024) {
        if (tid < 512) sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    /* Unroll last warp */
    if (tid < 32) {
        volatile float *smem = sdata;
        if (blockSize >= 64) smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/* ===========================================================================
 * KERNEL 7: Multiple Elements per Thread (Grid-Stride + Warp Shuffle)
 * ===========================================================================
 *
 * The ultimate optimization. Each thread processes MANY elements via a
 * grid-stride loop, accumulating a partial sum. Then the partial sums
 * are reduced in shared memory. The final warp uses warp shuffle
 * (__shfl_down_sync) instead of shared memory -- no bank conflicts,
 * no shared memory needed for the last 32 values.
 *
 *   Grid-stride loop (conceptual for 1 block, 4 threads, 16 elements):
 *
 *     t0: g[0] + g[4] + g[8]  + g[12]  -> partial_sum_0
 *     t1: g[1] + g[5] + g[9]  + g[13]  -> partial_sum_1
 *     t2: g[2] + g[6] + g[10] + g[14]  -> partial_sum_2
 *     t3: g[3] + g[7] + g[11] + g[15]  -> partial_sum_3
 *
 *   Shared memory tree on 4 partial sums:
 *     stride=2: t0: p0 += p2,  t1: p1 += p3
 *     stride=1: t0: p0 += p1  -> total
 *
 *   Warp shuffle for final intra-warp reduction:
 *
 *     Lane 0 has value V0, lane 1 has V1, ..., lane 31 has V31
 *
 *     offset=16: each lane adds the value from lane+16
 *       lane[i] += lane[i+16]  (via register, no shared mem)
 *
 *     offset=8:  lane[i] += lane[i+8]
 *     offset=4:  lane[i] += lane[i+4]
 *     offset=2:  lane[i] += lane[i+2]
 *     offset=1:  lane[i] += lane[i+1]
 *
 *     lane 0 = total of all 32 lanes
 *
 * WHY this is the fastest:
 * - Grid-stride loop: each thread does lots of useful work before reduction.
 * - Far fewer blocks needed (2x SM count is typical), reducing overhead.
 * - Warp shuffle: no shared memory for last warp, no bank conflicts.
 * - Memory coalescing: consecutive threads read consecutive addresses.
 * ---------------------------------------------------------------------------*/

/* Warp-level reduction using shuffle instructions */
__device__ float warpReduceSum(float val)
{
    /* __shfl_down_sync: each lane receives the value from (lane + offset) */
    /* mask 0xFFFFFFFF means all 32 lanes participate */
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;  /* lane 0 has the warp's total */
}

__global__ void reduce_multi_element(const float *g_idata,
                                      float *g_odata, int n)
{
    /*
     * Shared memory only needed for inter-warp reduction.
     * We need at most (blockDim.x / 32) floats = one per warp.
     */
    extern __shared__ float sdata[];

    unsigned int tid      = threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    /* -----------------------------------------------------------------------
     * Phase 1: Grid-stride loop -- each thread accumulates many elements.
     *
     * This is where most of the work happens. Each thread walks through
     * global memory with stride = total number of threads in the grid.
     * Memory accesses are coalesced because consecutive threads read
     * consecutive addresses within each "stride step."
     * -----------------------------------------------------------------------*/
    float mySum = 0.0f;
    for (unsigned int i = blockIdx.x * blockDim.x + tid; i < (unsigned int)n; i += gridSize) {
        mySum += g_idata[i];
    }

    /* -----------------------------------------------------------------------
     * Phase 2: Intra-warp reduction using warp shuffle.
     * Each warp of 32 threads reduces its 32 partial sums to 1 value
     * in lane 0 of that warp. No shared memory needed here.
     * -----------------------------------------------------------------------*/
    mySum = warpReduceSum(mySum);

    /* -----------------------------------------------------------------------
     * Phase 3: Inter-warp reduction using shared memory.
     * Lane 0 of each warp writes its sum to shared memory.
     * Then the first warp reduces those warp sums.
     * -----------------------------------------------------------------------*/
    int warpId = tid / 32;   /* which warp am I in? */
    int laneId = tid % 32;   /* which lane within the warp? */

    /* Lane 0 of each warp stores to shared memory */
    if (laneId == 0) {
        sdata[warpId] = mySum;
    }
    __syncthreads();

    /* First warp reads all warp sums and reduces them */
    int numWarps = (blockDim.x + 31) / 32;
    mySum = (tid < (unsigned int)numWarps) ? sdata[tid] : 0.0f;

    if (warpId == 0) {
        mySum = warpReduceSum(mySum);
    }

    /* Thread 0 writes the block's final result */
    if (tid == 0) {
        g_odata[blockIdx.x] = mySum;
    }
}

/* ===========================================================================
 * CPU reference implementation
 * ===========================================================================*/
double cpu_reduce_sum(const float *data, int n)
{
    /* Use double for the accumulator to get a more accurate reference
     * (float accumulation of millions of values loses precision) */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)data[i];
    }
    return sum;
}

/* ===========================================================================
 * Host helper: reduce partial block results on GPU (second pass)
 * ===========================================================================
 * After the first kernel launch, we have one partial sum per block.
 * This function reduces those partial sums to a single value.
 * For simplicity, if few enough partials, we reduce on CPU.
 * ---------------------------------------------------------------------------*/
float reduce_partials_on_cpu(float *d_partials, int numBlocks)
{
    float *h_partials = (float *)malloc(numBlocks * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_partials, d_partials, numBlocks * sizeof(float),
                           cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        sum += (double)h_partials[i];
    }
    free(h_partials);
    return (float)sum;
}

/* ===========================================================================
 * Benchmark runner
 * ===========================================================================*/

/* Number of timing iterations */
#define NUM_ITERS 100

/* Block size for kernels 1-6 */
#define BLOCK_SIZE 256

/* Array size for benchmarking */
#define N (1 << 22)   /* ~4 million elements = 16 MB */

int main()
{
    printf("=============================================================\n");
    printf(" Chapter 10: Parallel Reduction -- 7 Optimization Levels\n");
    printf("=============================================================\n");
    printf("Array size:  %d elements (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("Block size:  %d threads\n", BLOCK_SIZE);
    printf("Iterations:  %d (for timing)\n\n", NUM_ITERS);

    /* -------------------------------------------------------------------
     * Allocate and initialize host data
     * -------------------------------------------------------------------*/
    float *h_data = (float *)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) {
        /* Random values in [0, 1) to keep partial sums in float range */
        h_data[i] = (float)rand() / (float)RAND_MAX;
    }

    /* CPU reference sum (using double precision for accuracy) */
    double cpu_sum = cpu_reduce_sum(h_data, N);
    printf("CPU reference sum (double): %.6f\n\n", cpu_sum);

    /* -------------------------------------------------------------------
     * Allocate device memory
     * -------------------------------------------------------------------*/
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Partial results arrays (one element per block) */
    /* Kernels 1-3: numBlocks = ceil(N / BLOCK_SIZE) */
    int numBlocks_full = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    /* Kernels 4-6: half as many blocks (first add during load) */
    int numBlocks_half = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    /* For kernel 7: use fewer blocks -- 2 per SM is a good heuristic */
    int numSMs = 18;  /* Quadro P4200 */
    int numBlocks_k7 = numSMs * 2;

    /* Allocate partial results for the largest case */
    int maxBlocks = numBlocks_full;
    float *d_partial;
    CHECK_CUDA(cudaMalloc(&d_partial, maxBlocks * sizeof(float)));

    /* Shared memory size */
    int smemSize = BLOCK_SIZE * sizeof(float);

    /* CUDA events for timing */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Results table header */
    printf("%-5s %-42s %10s %10s %10s %8s\n",
           "Level", "Optimization", "Time (ms)", "BW (GB/s)", "Sum", "Speedup");
    printf("----------------------------------------------------------------------"
           "-------------------\n");

    float time_level1 = 0.0f;  /* baseline for speedup calculation */

    /* ===================================================================
     * KERNEL 1: Interleaved Addressing with Divergent Branching
     * ===================================================================*/
    {
        float gpu_sum;
        float elapsed = 0.0f;

        /* Warmup */
        reduce_interleaved_divergent<<<numBlocks_full, BLOCK_SIZE, smemSize>>>(
            d_data, d_partial, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        /* Timed iterations */
        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            reduce_interleaved_divergent<<<numBlocks_full, BLOCK_SIZE, smemSize>>>(
                d_data, d_partial, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        elapsed /= NUM_ITERS;

        gpu_sum = reduce_partials_on_cpu(d_partial, numBlocks_full);
        float bw = (N * sizeof(float)) / (elapsed * 1e6);  /* GB/s */
        time_level1 = elapsed;

        printf("%-5d %-42s %10.4f %10.2f %10.2f %8.2fx\n",
               1, "Interleaved, divergent", elapsed, bw, gpu_sum, 1.0f);
    }

    /* ===================================================================
     * KERNEL 2: Interleaved Addressing, Non-Divergent
     * ===================================================================*/
    {
        float gpu_sum;
        float elapsed = 0.0f;

        reduce_interleaved_bank_free<<<numBlocks_full, BLOCK_SIZE, smemSize>>>(
            d_data, d_partial, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            reduce_interleaved_bank_free<<<numBlocks_full, BLOCK_SIZE, smemSize>>>(
                d_data, d_partial, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        elapsed /= NUM_ITERS;

        gpu_sum = reduce_partials_on_cpu(d_partial, numBlocks_full);
        float bw = (N * sizeof(float)) / (elapsed * 1e6);
        float speedup = time_level1 / elapsed;

        printf("%-5d %-42s %10.4f %10.2f %10.2f %8.2fx\n",
               2, "Interleaved, non-divergent", elapsed, bw, gpu_sum, speedup);
    }

    /* ===================================================================
     * KERNEL 3: Sequential Addressing
     * ===================================================================*/
    {
        float gpu_sum;
        float elapsed = 0.0f;

        reduce_sequential<<<numBlocks_full, BLOCK_SIZE, smemSize>>>(
            d_data, d_partial, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            reduce_sequential<<<numBlocks_full, BLOCK_SIZE, smemSize>>>(
                d_data, d_partial, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        elapsed /= NUM_ITERS;

        gpu_sum = reduce_partials_on_cpu(d_partial, numBlocks_full);
        float bw = (N * sizeof(float)) / (elapsed * 1e6);
        float speedup = time_level1 / elapsed;

        printf("%-5d %-42s %10.4f %10.2f %10.2f %8.2fx\n",
               3, "Sequential addressing", elapsed, bw, gpu_sum, speedup);
    }

    /* ===================================================================
     * KERNEL 4: First Add During Load
     * ===================================================================*/
    {
        float gpu_sum;
        float elapsed = 0.0f;

        reduce_first_add_during_load<<<numBlocks_half, BLOCK_SIZE, smemSize>>>(
            d_data, d_partial, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            reduce_first_add_during_load<<<numBlocks_half, BLOCK_SIZE, smemSize>>>(
                d_data, d_partial, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        elapsed /= NUM_ITERS;

        gpu_sum = reduce_partials_on_cpu(d_partial, numBlocks_half);
        float bw = (N * sizeof(float)) / (elapsed * 1e6);
        float speedup = time_level1 / elapsed;

        printf("%-5d %-42s %10.4f %10.2f %10.2f %8.2fx\n",
               4, "First add during load", elapsed, bw, gpu_sum, speedup);
    }

    /* ===================================================================
     * KERNEL 5: Unroll Last Warp
     * ===================================================================*/
    {
        float gpu_sum;
        float elapsed = 0.0f;

        reduce_unroll_last_warp<<<numBlocks_half, BLOCK_SIZE, smemSize>>>(
            d_data, d_partial, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            reduce_unroll_last_warp<<<numBlocks_half, BLOCK_SIZE, smemSize>>>(
                d_data, d_partial, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        elapsed /= NUM_ITERS;

        gpu_sum = reduce_partials_on_cpu(d_partial, numBlocks_half);
        float bw = (N * sizeof(float)) / (elapsed * 1e6);
        float speedup = time_level1 / elapsed;

        printf("%-5d %-42s %10.4f %10.2f %10.2f %8.2fx\n",
               5, "Unroll last warp", elapsed, bw, gpu_sum, speedup);
    }

    /* ===================================================================
     * KERNEL 6: Completely Unrolled (Template)
     * ===================================================================*/
    {
        float gpu_sum;
        float elapsed = 0.0f;

        reduce_completely_unrolled<BLOCK_SIZE><<<numBlocks_half, BLOCK_SIZE, smemSize>>>(
            d_data, d_partial, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            reduce_completely_unrolled<BLOCK_SIZE><<<numBlocks_half, BLOCK_SIZE, smemSize>>>(
                d_data, d_partial, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        elapsed /= NUM_ITERS;

        gpu_sum = reduce_partials_on_cpu(d_partial, numBlocks_half);
        float bw = (N * sizeof(float)) / (elapsed * 1e6);
        float speedup = time_level1 / elapsed;

        printf("%-5d %-42s %10.4f %10.2f %10.2f %8.2fx\n",
               6, "Completely unrolled (template)", elapsed, bw, gpu_sum, speedup);
    }

    /* ===================================================================
     * KERNEL 7: Multi-Element per Thread (Grid-Stride + Warp Shuffle)
     * ===================================================================*/
    {
        float gpu_sum;
        float elapsed = 0.0f;

        /* Shared memory: one float per warp in the block */
        int smemSize_k7 = ((BLOCK_SIZE + 31) / 32) * sizeof(float);

        reduce_multi_element<<<numBlocks_k7, BLOCK_SIZE, smemSize_k7>>>(
            d_data, d_partial, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start));
        for (int iter = 0; iter < NUM_ITERS; iter++) {
            reduce_multi_element<<<numBlocks_k7, BLOCK_SIZE, smemSize_k7>>>(
                d_data, d_partial, N);
        }
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        elapsed /= NUM_ITERS;

        gpu_sum = reduce_partials_on_cpu(d_partial, numBlocks_k7);
        float bw = (N * sizeof(float)) / (elapsed * 1e6);
        float speedup = time_level1 / elapsed;

        printf("%-5d %-42s %10.4f %10.2f %10.2f %8.2fx\n",
               7, "Grid-stride + warp shuffle", elapsed, bw, gpu_sum, speedup);
    }

    printf("----------------------------------------------------------------------"
           "-------------------\n");
    printf("\nNote: BW = effective bandwidth (bytes read / time).\n");
    printf("      Peak memory BW for Quadro P4200 is ~134 GB/s.\n");
    printf("      Kernel 7 should approach this limit.\n");

    /* -------------------------------------------------------------------
     * Cleanup
     * -------------------------------------------------------------------*/
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_partial));
    free(h_data);

    return 0;
}
