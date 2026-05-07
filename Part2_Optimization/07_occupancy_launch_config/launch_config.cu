/*
 * launch_config.cu
 * ================
 * Finding optimal launch configurations for CUDA kernels.
 *
 * This demo shows:
 *   1. cudaOccupancyMaxPotentialBlockSize — the CUDA auto-tuner for block size
 *   2. Performance comparison across block sizes: 32, 64, 128, 256, 512, 1024
 *   3. __launch_bounds__ to control compiler behavior
 *   4. How the "best" block size depends on the kernel
 *
 * Key insight: The optimal block size balances:
 *   - Occupancy (enough warps to hide latency)
 *   - Register pressure (more threads = fewer regs/thread)
 *   - Shared memory usage
 *   - Instruction-level parallelism within each thread
 *
 * Target: Quadro P4200 (CC 6.1, 18 SMs)
 * Build:  nvcc -arch=sm_61 -O2 -lineinfo -ccbin g++-11 launch_config.cu -o launch_config
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * ERROR CHECKING
 * ═══════════════════════════════════════════════════════════════════════════ */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL A: Memory-bound (simple element-wise operation)
 * ─────────────────────────────────────────────────────────────────────────
 * This kernel does very little compute per byte loaded. It is entirely
 * limited by memory bandwidth. For memory-bound kernels:
 *   - Higher occupancy helps (more warps to hide memory latency)
 *   - Block size matters mainly for occupancy
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_memory_bound(const float* __restrict__ input,
                                    float* __restrict__ output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        /* ~1 FLOP per 4 bytes loaded + 4 bytes stored.
         * Arithmetic intensity ≈ 1/8 FLOP/byte → very memory-bound. */
        output[idx] = input[idx] * 2.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL B: Compute-bound (heavy math per element)
 * ─────────────────────────────────────────────────────────────────────────
 * This kernel does a LOT of computation per element. It is limited by
 * ALU throughput, not memory. For compute-bound kernels:
 *   - Occupancy beyond ~50% gives diminishing returns
 *   - Sometimes lower occupancy with more ILP per thread is better
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_compute_bound(const float* __restrict__ input,
                                     float* __restrict__ output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        float result = x;

        /* ~100+ FLOPs per element loaded.
         * Arithmetic intensity ≈ 100/8 = 12.5 FLOP/byte → compute-bound. */
        #pragma unroll
        for (int i = 0; i < 20; i++) {
            result = sinf(result) * cosf(result) + sqrtf(fabsf(result));
            result = fmaf(result, 0.999f, 0.001f);
        }
        output[idx] = result;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL C: With __launch_bounds__ hint
 * ─────────────────────────────────────────────────────────────────────────
 * __launch_bounds__(256, 4) tells the compiler:
 *   - This kernel will never be launched with more than 256 threads/block
 *   - We want at least 4 blocks per SM to be resident
 *
 * This influences register allocation:
 *   - Without hint: compiler may use many registers, optimizing single-thread perf
 *   - With hint: compiler may spill some registers to keep within the budget
 *     so that 4 blocks × 256 threads = 1024 threads can fit per SM
 *
 * The register budget: 65536 / (4 × 256) = 64 regs/thread max
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void __launch_bounds__(256, 4)
kernel_launch_bounded(const float* __restrict__ input,
                      float* __restrict__ output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        float result = x;

        /* Same computation as kernel_compute_bound, but with __launch_bounds__.
         * The compiler may generate different code due to the register constraint. */
        #pragma unroll
        for (int i = 0; i < 20; i++) {
            result = sinf(result) * cosf(result) + sqrtf(fabsf(result));
            result = fmaf(result, 0.999f, 0.001f);
        }
        output[idx] = result;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL D: Shared memory reduction (block size significantly affects perf)
 * ─────────────────────────────────────────────────────────────────────────
 * For reduction kernels, block size directly affects:
 *   - How much work each block does
 *   - How many levels of reduction happen in shared memory
 *   - Total number of blocks (and thus kernel launch overhead)
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_reduction(const float* __restrict__ input,
                                 float* __restrict__ partial_sums, int N)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    /* Each thread loads TWO elements (loop unrolling for efficiency). */
    float sum = 0.0f;
    if (idx < N)                sum += input[idx];
    if (idx + blockDim.x < N)   sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    /* Tree reduction in shared memory.
     * At each step, half the threads drop out. */
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    /* Thread 0 writes the block's partial sum */
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK HELPER
 * ─────────────────────────────────────────────────────────────────────────
 * Times a kernel launch using CUDA events. Returns elapsed time in ms.
 *
 * We run the kernel multiple times:
 *   - warmupRuns: to warm caches, TLBs, etc.
 *   - timedRuns:  to get a stable measurement
 * ═══════════════════════════════════════════════════════════════════════════ */
template <typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, const float* d_in, float* d_out,
                       int N, int blockSize, int warmupRuns, int timedRuns)
{
    int gridSize = (N + blockSize - 1) / blockSize;

    /* Warmup: get past any first-launch overhead */
    for (int i = 0; i < warmupRuns; i++) {
        kernel<<<gridSize, blockSize>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed runs */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timedRuns; i++) {
        kernel<<<gridSize, blockSize>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / timedRuns;   /* average time per run */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK HELPER for the reduction kernel (needs shared memory size)
 * ═══════════════════════════════════════════════════════════════════════════ */
float benchmark_reduction(const float* d_in, float* d_partial,
                          int N, int blockSize, int warmupRuns, int timedRuns)
{
    /* Each block processes 2 × blockSize elements */
    int gridSize = (N + blockSize * 2 - 1) / (blockSize * 2);
    size_t sharedBytes = blockSize * sizeof(float);

    for (int i = 0; i < warmupRuns; i++) {
        kernel_reduction<<<gridSize, blockSize, sharedBytes>>>(d_in, d_partial, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timedRuns; i++) {
        kernel_reduction<<<gridSize, blockSize, sharedBytes>>>(d_in, d_partial, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / timedRuns;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main()
{
    /* ── Problem size ── */
    const int N = 1 << 24;   /* 16M elements = 64 MB */
    const int warmup = 5;
    const int runs   = 20;
    size_t bytes = N * sizeof(float);

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  LAUNCH CONFIGURATION BENCHMARK\n");
    printf("  N = %d elements (%.1f MB)\n", N, bytes / (1024.0 * 1024.0));
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");

    /* ── Allocate and initialize ── */
    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f + 0.001f * (i % 1000);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    /* ── Partial sums buffer for reduction (max grid size) ── */
    int maxGrid = (N + 31) / 32;    /* worst case: blockSize=32, ×2 elements each = 64 */
    float *d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, maxGrid * sizeof(float)));

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numSizes = 6;

    /* ══════════════════════════════════════════════════════════════════════
     * SECTION 1: cudaOccupancyMaxPotentialBlockSize — Auto-Tuner
     * ══════════════════════════════════════════════════════════════════════
     *
     * This API asks the CUDA runtime: "Given this kernel's register usage
     * and shared memory needs, what block size maximizes occupancy?"
     *
     * It considers all the hardware limits and returns:
     *   - minGridSize: minimum grid size to fully occupy the GPU
     *   - blockSize:   optimal threads per block
     */
    printf("─── CUDA Auto-Tuner: cudaOccupancyMaxPotentialBlockSize ────────────────────\n\n");

    {
        int minGridSize, optBlockSize;

        /* Memory-bound kernel */
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,         // output: suggested grid size
            &optBlockSize,        // output: optimal block size
            kernel_memory_bound,  // the kernel to analyze
            0,                    // dynamic shared memory (bytes)
            0                     // block size limit (0 = use device max)
        ));
        printf("  kernel_memory_bound:   optimal blockSize = %d, minGridSize = %d\n",
               optBlockSize, minGridSize);

        /* Compute-bound kernel */
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &optBlockSize,
            kernel_compute_bound, 0, 0
        ));
        printf("  kernel_compute_bound:  optimal blockSize = %d, minGridSize = %d\n",
               optBlockSize, minGridSize);

        /* Launch-bounded kernel */
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &optBlockSize,
            kernel_launch_bounded, 0, 0
        ));
        printf("  kernel_launch_bounded: optimal blockSize = %d, minGridSize = %d\n",
               optBlockSize, minGridSize);

        printf("\n  Note: the auto-tuner optimizes for OCCUPANCY, not necessarily for\n");
        printf("  the fastest execution time. Always benchmark to confirm!\n\n");
    }

    /* ══════════════════════════════════════════════════════════════════════
     * SECTION 2: Benchmark memory-bound kernel across block sizes
     * ══════════════════════════════════════════════════════════════════════ */
    printf("─── Benchmark: Memory-Bound Kernel ────────────────────────────────────────\n\n");
    printf("  %-12s │ %10s │ %10s │ %12s\n", "Block Size", "Grid Size", "Time (ms)", "BW (GB/s)");
    printf("  ─────────────┼────────────┼────────────┼─────────────\n");

    float bestTimeMemBound = 1e9;
    int   bestBSMemBound = 0;

    for (int i = 0; i < numSizes; i++) {
        int bs = blockSizes[i];
        int gs = (N + bs - 1) / bs;
        float ms = benchmark_kernel(kernel_memory_bound, d_in, d_out, N, bs, warmup, runs);

        /* Effective bandwidth: bytes read + bytes written, divided by time.
         * 2 * N * 4 bytes (read + write) / time_in_seconds / 1e9 = GB/s */
        float bw = (2.0f * N * sizeof(float)) / (ms * 1e-3f) / 1e9f;

        printf("  %-12d │ %10d │ %10.3f │ %9.1f\n", bs, gs, ms, bw);

        if (ms < bestTimeMemBound) {
            bestTimeMemBound = ms;
            bestBSMemBound = bs;
        }
    }
    printf("\n  Best block size for memory-bound kernel: %d (%.3f ms)\n\n",
           bestBSMemBound, bestTimeMemBound);

    /* ══════════════════════════════════════════════════════════════════════
     * SECTION 3: Benchmark compute-bound kernel across block sizes
     * ══════════════════════════════════════════════════════════════════════ */
    printf("─── Benchmark: Compute-Bound Kernel ───────────────────────────────────────\n\n");
    printf("  %-12s │ %10s │ %10s\n", "Block Size", "Grid Size", "Time (ms)");
    printf("  ─────────────┼────────────┼────────────\n");

    float bestTimeCompute = 1e9;
    int   bestBSCompute = 0;

    for (int i = 0; i < numSizes; i++) {
        int bs = blockSizes[i];
        int gs = (N + bs - 1) / bs;
        float ms = benchmark_kernel(kernel_compute_bound, d_in, d_out, N, bs, warmup, runs);

        printf("  %-12d │ %10d │ %10.3f\n", bs, gs, ms);

        if (ms < bestTimeCompute) {
            bestTimeCompute = ms;
            bestBSCompute = bs;
        }
    }
    printf("\n  Best block size for compute-bound kernel: %d (%.3f ms)\n\n",
           bestBSCompute, bestTimeCompute);

    /* ══════════════════════════════════════════════════════════════════════
     * SECTION 4: Compare launch-bounded vs unbounded
     * ══════════════════════════════════════════════════════════════════════
     * __launch_bounds__ may cause register spilling, which trades register
     * speed for higher occupancy. Is it worth it?
     */
    printf("─── Comparison: __launch_bounds__ Effect ──────────────────────────────────\n\n");

    {
        cudaFuncAttributes attr_unbounded, attr_bounded;
        CUDA_CHECK(cudaFuncGetAttributes(&attr_unbounded, kernel_compute_bound));
        CUDA_CHECK(cudaFuncGetAttributes(&attr_bounded,   kernel_launch_bounded));

        printf("  kernel_compute_bound  (no launch_bounds): %d regs/thread\n",
               attr_unbounded.numRegs);
        printf("  kernel_launch_bounded (__launch_bounds__): %d regs/thread\n\n",
               attr_bounded.numRegs);

        int bs = 256;   /* Use the same block size for fair comparison */

        float ms_unbounded = benchmark_kernel(kernel_compute_bound,  d_in, d_out, N, bs, warmup, runs);
        float ms_bounded   = benchmark_kernel(kernel_launch_bounded, d_in, d_out, N, bs, warmup, runs);

        /* Query occupancy for both */
        int blocks_unbounded, blocks_bounded;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_unbounded,
                   kernel_compute_bound, bs, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_bounded,
                   kernel_launch_bounded, bs, 0));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        int maxWarps = prop.maxThreadsPerMultiProcessor / 32;

        printf("  %-24s │ %10s │ %10s │ %8s\n", "Kernel", "Time (ms)", "Occupancy", "Blks/SM");
        printf("  ────────────────────────┼────────────┼────────────┼─────────\n");
        printf("  %-24s │ %10.3f │ %9.1f%% │ %7d\n", "Unbounded",
               ms_unbounded,
               100.0f * blocks_unbounded * (bs/32) / (float)maxWarps,
               blocks_unbounded);
        printf("  %-24s │ %10.3f │ %9.1f%% │ %7d\n", "launch_bounds(256,4)",
               ms_bounded,
               100.0f * blocks_bounded * (bs/32) / (float)maxWarps,
               blocks_bounded);

        if (ms_bounded < ms_unbounded) {
            printf("\n  __launch_bounds__ HELPED: higher occupancy outweighed register spilling.\n");
        } else {
            printf("\n  __launch_bounds__ HURT (or no difference): register spilling cost\n");
            printf("  outweighed the occupancy benefit. This is kernel-dependent!\n");
        }
        printf("\n");
    }

    /* ══════════════════════════════════════════════════════════════════════
     * SECTION 5: Reduction kernel — block size matters a LOT
     * ══════════════════════════════════════════════════════════════════════ */
    printf("─── Benchmark: Reduction Kernel ───────────────────────────────────────────\n\n");
    printf("  %-12s │ %10s │ %10s │ %12s\n", "Block Size", "Grid Size", "Time (ms)", "BW (GB/s)");
    printf("  ─────────────┼────────────┼────────────┼─────────────\n");

    float bestTimeReduction = 1e9;
    int   bestBSReduction = 0;

    for (int i = 0; i < numSizes; i++) {
        int bs = blockSizes[i];
        int gs = (N + bs * 2 - 1) / (bs * 2);
        float ms = benchmark_reduction(d_in, d_partial, N, bs, warmup, runs);

        /* For reduction, we read N elements → N × 4 bytes */
        float bw = (float)(N * sizeof(float)) / (ms * 1e-3f) / 1e9f;

        printf("  %-12d │ %10d │ %10.3f │ %9.1f\n", bs, gs, ms, bw);

        if (ms < bestTimeReduction) {
            bestTimeReduction = ms;
            bestBSReduction = bs;
        }
    }
    printf("\n  Best block size for reduction: %d (%.3f ms)\n\n",
           bestBSReduction, bestTimeReduction);

    /* ══════════════════════════════════════════════════════════════════════
     * SUMMARY
     * ══════════════════════════════════════════════════════════════════════ */
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  SUMMARY\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  Different kernels prefer different block sizes:\n");
    printf("    Memory-bound:  best at blockSize = %d\n", bestBSMemBound);
    printf("    Compute-bound: best at blockSize = %d\n", bestBSCompute);
    printf("    Reduction:     best at blockSize = %d\n\n", bestBSReduction);
    printf("  cudaOccupancyMaxPotentialBlockSize is a good starting point,\n");
    printf("  but always benchmark your specific kernel to find the true optimum.\n\n");
    printf("  The optimal config depends on:\n");
    printf("    - Whether the kernel is memory-bound or compute-bound\n");
    printf("    - Register usage (which affects occupancy)\n");
    printf("    - Shared memory usage\n");
    printf("    - Work distribution (does each block do enough work?)\n");
    printf("    - GPU architecture specifics\n\n");

    /* ── Cleanup ── */
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_partial));

    return 0;
}
