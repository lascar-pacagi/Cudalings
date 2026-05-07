/*
 * bandwidth_test.cu -- Measuring Actual GPU Memory Bandwidth
 * ==========================================================
 *
 * The single most important metric for GPU kernel performance is
 * MEMORY BANDWIDTH UTILIZATION. This program measures your GPU's
 * actual achievable bandwidth and teaches you to think about
 * performance in bandwidth terms.
 *
 * We run three simple memory-bound kernels:
 *   1. COPY:   out[i] = in[i]            (1 read + 1 write = 8 bytes/element)
 *   2. SCALE:  out[i] = alpha * in[i]    (1 read + 1 write = 8 bytes/element)
 *   3. ADD:    out[i] = a[i] + b[i]      (2 reads + 1 write = 12 bytes/element)
 *
 * These are the simplest possible kernels. Their performance is 100%
 * determined by memory bandwidth -- there is almost zero compute.
 * This makes them perfect for measuring raw bandwidth.
 *
 * Your Quadro P4000 theoretical peak: ~192 GB/s
 * Expected achievable: ~160-175 GB/s (80-90% of peak)
 *
 * Compile: nvcc -arch=sm_61 -O2 -o bandwidth_test bandwidth_test.cu
 * Run:     ./bandwidth_test
 *
 * Target: Quadro P4000 (CC 6.1, 14 SMs, 256-bit GDDR5)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// Array size: use a large array so the working set does NOT fit in cache.
// L2 cache on P4000 is ~2 MB. We use 64 MB arrays to ensure we measure
// DRAM bandwidth, not cache bandwidth.
#define N (16 * 1024 * 1024)   // 16M elements = 64 MB per array

#define BLOCK_SIZE 256         // Threads per block
#define ITERS 20               // Number of benchmark iterations

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


// ============================================================================
// KERNEL 1: COPY -- The Simplest Bandwidth Test
// ============================================================================
//
// out[i] = in[i]
//
// Bytes moved per element: 4 (read) + 4 (write) = 8 bytes
// FLOPs per element:       0
// Arithmetic intensity:    0 FLOP/byte (pure memory test)
//
// This is the GPU equivalent of memcpy. It measures the absolute maximum
// bandwidth your GPU can achieve for a simple streaming pattern.
//
__global__ void kernel_copy(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop for full coverage.
    // Access pattern is COALESCED: consecutive threads access consecutive
    // addresses. This is the best possible pattern for memory bandwidth.
    //
    //   Thread 0 -> in[0], out[0]
    //   Thread 1 -> in[1], out[1]
    //   Thread 2 -> in[2], out[2]
    //   ...
    //   Warp of 32 threads -> 128 consecutive bytes -> 1 memory transaction
    //
    for (; i < n; i += stride) {
        out[i] = in[i];
    }
}


// ============================================================================
// KERNEL 2: SCALE -- Bandwidth with Minimal Compute
// ============================================================================
//
// out[i] = alpha * in[i]
//
// Bytes moved per element: 4 (read) + 4 (write) = 8 bytes
// FLOPs per element:       1 (multiply)
// Arithmetic intensity:    1/8 = 0.125 FLOP/byte
//
// Even with the multiply, this kernel is 100% memory-bound.
// The GPU has thousands of ALUs that can easily do one multiply while
// waiting 400+ cycles for the next memory load.
//
__global__ void kernel_scale(const float *in, float *out, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        out[i] = alpha * in[i];
    }
}


// ============================================================================
// KERNEL 3: ADD -- Two Input Streams
// ============================================================================
//
// out[i] = a[i] + b[i]
//
// Bytes moved per element: 4 + 4 (reads) + 4 (write) = 12 bytes
// FLOPs per element:       1 (addition)
// Arithmetic intensity:    1/12 = 0.083 FLOP/byte
//
// This kernel reads from TWO input arrays, so it moves 50% more data
// per element than copy/scale. The achieved bandwidth should be similar
// (the memory controller handles multiple streams well), but the
// bytes-per-element is higher.
//
__global__ void kernel_add(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}


// ============================================================================
// TIMING HELPERS
// ============================================================================

typedef struct { cudaEvent_t start, stop; } GpuTimer;

void timer_create(GpuTimer *t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
}
void timer_destroy(GpuTimer *t) {
    CUDA_CHECK(cudaEventDestroy(t->start));
    CUDA_CHECK(cudaEventDestroy(t->stop));
}
void timer_start(GpuTimer *t) {
    CUDA_CHECK(cudaEventRecord(t->start, 0));
}
float timer_stop(GpuTimer *t) {
    float ms;
    CUDA_CHECK(cudaEventRecord(t->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    return ms;
}


// ============================================================================
// BANDWIDTH CALCULATION AND REPORTING
// ============================================================================
//
// Achieved bandwidth = total_bytes_moved / kernel_time
//
// HOW TO THINK ABOUT PERFORMANCE IN BANDWIDTH TERMS:
//
//   1. Count the bytes your kernel moves per element:
//      - Each float read  = 4 bytes
//      - Each float write = 4 bytes
//
//   2. Multiply by N to get total bytes.
//
//   3. Divide by kernel time to get GB/s.
//
//   4. Compare to theoretical peak (~192 GB/s for your GPU).
//      - >80% of peak = excellent (well-optimized memory-bound kernel)
//      - 50-80%       = decent (some inefficiency, investigate)
//      - <50%         = poor (uncoalesced access, bank conflicts, etc.)
//
//   5. If you are already at 80%+ of peak bandwidth, the ONLY ways to
//      go faster are:
//      a) Move less data (algorithmic change, better caching)
//      b) Use a GPU with more bandwidth
//      c) Overlap computation with memory access (harder)
//
void report_bandwidth(const char *name, float ms, double total_bytes,
                      double peak_bw_gb) {
    double seconds = ms * 1e-3;
    double gb_moved = total_bytes / 1e9;
    double achieved_bw = gb_moved / seconds;
    double utilization = 100.0 * achieved_bw / peak_bw_gb;

    printf("  %-10s %7.3f ms  |  %6.1f GB/s  |  %5.1f%% of peak",
           name, ms / ITERS, achieved_bw, utilization);

    // Visual bar showing bandwidth utilization
    printf("  [");
    int bar_len = 30;
    int filled = (int)(utilization / 100.0 * bar_len);
    for (int i = 0; i < bar_len; i++) {
        printf("%c", i < filled ? '#' : '.');
    }
    printf("]\n");
}


// ============================================================================
// MAIN
// ============================================================================

int main() {
    // --- Query device info to calculate theoretical peak bandwidth ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // Theoretical peak bandwidth calculation:
    //   BW = 2 (DDR) * memory_clock_rate (Hz) * bus_width (bytes)
    //
    //   memory_clock_rate is in kHz from the API
    //   bus_width is in bits from the API
    //
    double peak_bw_gb = 2.0 * prop.memoryClockRate * 1e3 *
                        (prop.memoryBusWidth / 8) / 1e9;

    printf("============================================================\n");
    printf("GPU MEMORY BANDWIDTH TEST\n");
    printf("============================================================\n");
    printf("Device:          %s\n", prop.name);
    printf("Memory clock:    %d MHz (effective)\n", prop.memoryClockRate / 1000);
    printf("Bus width:       %d bits\n", prop.memoryBusWidth);
    printf("Peak bandwidth:  %.1f GB/s (theoretical)\n", peak_bw_gb);
    printf("SMs:             %d\n", prop.multiProcessorCount);
    printf("\n");
    printf("Array size:      N = %d (%.0f MB per array)\n",
           N, (double)N * sizeof(float) / (1024 * 1024));
    printf("Block size:      %d threads\n", BLOCK_SIZE);
    printf("Iterations:      %d\n", ITERS);
    printf("============================================================\n\n");

    // --- Allocate memory ---
    size_t bytes = N * sizeof(float);
    float *h_in = (float *)malloc(bytes);

    float *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i * 0.001f;
    }
    CUDA_CHECK(cudaMemcpy(d_a, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_in, bytes, cudaMemcpyHostToDevice));

    // Grid configuration
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    GpuTimer timer;
    timer_create(&timer);

    // --- Warmup (first kernel launch has driver overhead) ---
    kernel_copy<<<num_blocks, BLOCK_SIZE>>>(d_a, d_out, N);
    kernel_scale<<<num_blocks, BLOCK_SIZE>>>(d_a, d_out, 2.0f, N);
    kernel_add<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Benchmark results (averaged over %d iterations):\n\n", ITERS);
    printf("  %-10s %10s  |  %10s  |  %15s\n",
           "Kernel", "Time/iter", "Bandwidth", "Utilization");
    printf("  %-10s %10s  |  %10s  |  %15s\n",
           "------", "---------", "---------", "-----------");

    // =========================================================================
    // TEST 1: COPY (8 bytes per element)
    // =========================================================================
    timer_start(&timer);
    for (int iter = 0; iter < ITERS; iter++) {
        kernel_copy<<<num_blocks, BLOCK_SIZE>>>(d_a, d_out, N);
    }
    float ms_copy = timer_stop(&timer);
    // Total bytes: N elements * (4 bytes read + 4 bytes write) * ITERS
    double copy_bytes = (double)N * 8.0 * ITERS;
    report_bandwidth("COPY", ms_copy, copy_bytes, peak_bw_gb);

    // =========================================================================
    // TEST 2: SCALE (8 bytes per element, 1 FLOP)
    // =========================================================================
    timer_start(&timer);
    for (int iter = 0; iter < ITERS; iter++) {
        kernel_scale<<<num_blocks, BLOCK_SIZE>>>(d_a, d_out, 2.0f, N);
    }
    float ms_scale = timer_stop(&timer);
    double scale_bytes = (double)N * 8.0 * ITERS;
    report_bandwidth("SCALE", ms_scale, scale_bytes, peak_bw_gb);

    // =========================================================================
    // TEST 3: ADD (12 bytes per element, 1 FLOP)
    // =========================================================================
    timer_start(&timer);
    for (int iter = 0; iter < ITERS; iter++) {
        kernel_add<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_out, N);
    }
    float ms_add = timer_stop(&timer);
    // 2 reads + 1 write = 12 bytes per element
    double add_bytes = (double)N * 12.0 * ITERS;
    report_bandwidth("ADD", ms_add, add_bytes, peak_bw_gb);

    // =========================================================================
    // Summary and Analysis
    // =========================================================================
    printf("\n============================================================\n");
    printf("HOW TO USE THESE NUMBERS\n");
    printf("============================================================\n");
    printf("\n");
    printf("Your GPU achieves roughly the bandwidth numbers shown above.\n");
    printf("Call this your PRACTICAL PEAK (let's say ~%.0f GB/s).\n\n",
           (double)N * 8.0 * ITERS / (ms_copy * 1e-3) / 1e9);

    printf("For any kernel, you can now estimate performance:\n\n");

    printf("  1. COUNT the bytes moved per element:\n");
    printf("     - vector_add (a+b->c):  12 bytes  (2 reads + 1 write)\n");
    printf("     - saxpy (a*x+y->y):     12 bytes  (2 reads + 1 write)\n");
    printf("     - dot product:           8 bytes  (2 reads, 1 write amortized)\n");
    printf("     - matrix multiply:       varies   (depends on tiling!)\n\n");

    double practical_peak = (double)N * 8.0 * ITERS / (ms_copy * 1e-3) / 1e9;
    printf("  2. ESTIMATE time = total_bytes / practical_peak_bw\n");
    printf("     Example: vector_add of 10M floats\n");
    printf("       bytes = 10M * 12 = 120 MB\n");
    printf("       time  = 0.120 / %.0f = %.3f ms\n\n",
           practical_peak,
           0.120 / practical_peak * 1000.0);

    printf("  3. COMPARE your kernel time against this estimate:\n");
    printf("     - Close to estimate: kernel is well-optimized\n");
    printf("     - Much slower: memory access pattern is bad\n");
    printf("       (uncoalesced, bank conflicts, etc.)\n");
    printf("     - Faster than estimate: your byte count is wrong,\n");
    printf("       or data fits in cache\n\n");

    printf("  4. IF your kernel is already at practical peak bandwidth:\n");
    printf("     - You CANNOT make it faster without:\n");
    printf("       a) Reducing bytes moved (algorithmic improvement)\n");
    printf("       b) Using a faster GPU\n");
    printf("     - This is the most important lesson in GPU optimization.\n");
    printf("       Don't waste time optimizing what is already optimal!\n\n");

    printf("============================================================\n");
    printf("ARITHMETIC INTENSITY CHEAT SHEET\n");
    printf("============================================================\n");
    printf("\n");
    printf("  Operation          AI (FLOP/byte)  Bound by\n");
    printf("  ---------          --------------  --------\n");
    printf("  copy               0.00            Memory\n");
    printf("  vector add         0.08            Memory\n");
    printf("  saxpy              0.17            Memory\n");
    printf("  dot product        0.25            Memory\n");
    printf("  SpMV               0.25            Memory\n");
    printf("  stencil (R=3)      ~1.0            Memory\n");
    printf("  matrix multiply    ~%.0f             Compute (at large N)\n",
           (float)N / 12.0);  // FLOPs = 2*N^3/3, bytes = 3*N^2*4
    printf("  (NxN, large N)\n");
    printf("\n");
    printf("  Ridge point for your GPU: %.1f FLOP/byte\n",
           5300.0 / peak_bw_gb);
    printf("  (Below this = memory-bound, above = compute-bound)\n");
    printf("\n");

    // Cleanup
    timer_destroy(&timer);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);

    printf("Done!\n");
    return 0;
}

/*
 * =============================================================================
 * KEY TAKEAWAYS:
 * =============================================================================
 *
 * 1. MEMORY BANDWIDTH is the #1 performance metric for most GPU kernels.
 *    If your kernel achieves >80% of peak bandwidth, it is well-optimized.
 *
 * 2. MEASURE, DON'T GUESS. Run this benchmark on your GPU to know your
 *    practical peak. Theoretical peak is rarely achievable.
 *
 * 3. BANDWIDTH UTILIZATION tells you if your memory access pattern is good.
 *    Low utilization means uncoalesced access, bank conflicts, or other issues.
 *
 * 4. ARITHMETIC INTENSITY determines whether your kernel is memory-bound
 *    or compute-bound. Most kernels are memory-bound (AI < ~28 FLOP/byte
 *    on this GPU).
 *
 * 5. Once you hit peak bandwidth, the ONLY way to go faster is to move
 *    less data. This is where shared memory, caching, and algorithmic
 *    improvements come in (see stencil_1d.cu for an example).
 *
 * 6. These simple copy/scale/add kernels are essentially the STREAM benchmark
 *    (https://www.cs.virginia.edu/stream/), adapted for GPU.
 *
 * =============================================================================
 */
