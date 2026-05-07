// =============================================================================
// Chapter 04: Reusable Benchmark Harness
// =============================================================================
//
// This file provides a templated benchmarking framework that you can use
// in all later chapters to accurately measure kernel performance.
//
// Features:
//   - Warm-up runs (to eliminate first-launch overhead)
//   - Multiple iterations with statistical analysis
//   - Mean, median, and standard deviation of timing
//   - Bandwidth (GB/s) and GFLOPS reporting
//   - Easy to use: just pass a lambda wrapping your kernel launch
//
// Example usage:
//
//   BenchmarkConfig config;
//   config.warmup_runs = 3;
//   config.benchmark_runs = 10;
//   config.bytes_moved = 2 * N * sizeof(float);
//   config.flops = N * 2.0;
//
//   BenchmarkResult result = benchmark("my_kernel", config, [&]() {
//       my_kernel<<<blocks, threads>>>(d_in, d_out, N);
//   });
//
// This file also demonstrates benchmarking vector_add with different
// block sizes to show how block size affects performance.
//
// Compile:  nvcc -arch=sm_61 -O2 -lineinfo -o benchmark_harness benchmark_harness.cu
// Run:      ./benchmark_harness
//
// Hardware: Quadro P4200 (CC 6.1)
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>   // std::sort
#include <functional>  // std::function

// =============================================================================
// Error checking macro
// =============================================================================
#define CUDA_CHECK(err) do {                                           \
    cudaError_t err_ = (err);                                          \
    if (err_ != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d -- %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err_));         \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)


// =============================================================================
// BENCHMARK CONFIGURATION
// =============================================================================
//
// This struct controls how the benchmark runs. Set these before calling
// benchmark(). The defaults are reasonable for most cases.
//
struct BenchmarkConfig {
    int warmup_runs;       // Number of warm-up iterations (not timed)
    int benchmark_runs;    // Number of timed iterations
    size_t bytes_moved;    // Total bytes read + written per kernel call
    double flops;          // Total FLOPs per kernel call

    // Constructor with sensible defaults
    BenchmarkConfig()
        : warmup_runs(3)       // 3 warm-up runs to stabilize clocks/caches
        , benchmark_runs(10)   // 10 timed runs for statistical significance
        , bytes_moved(0)       // Set to 0 if you don't want bandwidth reported
        , flops(0.0)           // Set to 0 if you don't want GFLOPS reported
    {}
};


// =============================================================================
// BENCHMARK RESULT
// =============================================================================
//
// Returned by benchmark(). Contains all the timing statistics and
// derived metrics (bandwidth, GFLOPS).
//
struct BenchmarkResult {
    float mean_ms;         // Mean time across all runs
    float median_ms;       // Median time (less affected by outliers)
    float min_ms;          // Fastest run
    float max_ms;          // Slowest run
    float stddev_ms;       // Standard deviation

    float bandwidth_gbps;  // GB/s based on median time
    float gflops;          // GFLOPS based on median time

    int num_runs;          // How many timed runs were performed
};


// =============================================================================
// THE BENCHMARK FUNCTION
// =============================================================================
//
// This is the core of the harness. It:
//   1. Runs the kernel a few times to warm up (first launch initializes
//      CUDA context, subsequent launches warm GPU caches and stabilize clocks)
//   2. Times each benchmark run with CUDA events (GPU-accurate timing)
//   3. Collects all times and computes statistics
//   4. Derives bandwidth and GFLOPS from the median time
//
// Parameters:
//   name   -- Display name for the kernel being benchmarked
//   config -- BenchmarkConfig with run counts and metric parameters
//   kernel -- A callable (lambda, function pointer) that launches the kernel
//             The callable should include the kernel launch AND nothing else.
//             It must NOT include cudaDeviceSynchronize -- the harness handles that.
//
// Returns:
//   BenchmarkResult with all timing statistics and derived metrics
//
// Why use the MEDIAN instead of the MEAN?
//   - GPU timing can have outliers (context switches, thermal throttling)
//   - The median is robust to these outliers
//   - The mean is still reported for reference
//   - The stddev tells you how stable the measurements are
//
BenchmarkResult benchmark(const char *name,
                          const BenchmarkConfig &config,
                          std::function<void()> kernel) {

    BenchmarkResult result = {};
    result.num_runs = config.benchmark_runs;

    // Allocate array for individual run times
    float *times = new float[config.benchmark_runs];

    // --- Step 1: Warm-up runs ---
    // These are not timed. They serve multiple purposes:
    //   - First CUDA call initializes the driver (~100ms overhead)
    //   - Subsequent calls warm L2 cache, TLBs, and GPU clocks
    //   - GPU may boost clock speed after seeing sustained workload
    for (int i = 0; i < config.warmup_runs; i++) {
        kernel();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // --- Step 2: Timed runs ---
    // Each run is individually timed with CUDA events.
    // We synchronize after each run to ensure we measure only this kernel.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < config.benchmark_runs; i++) {
        // Record start event -- places a timestamp in the GPU command stream
        CUDA_CHECK(cudaEventRecord(start));

        // Launch the kernel (user-provided callable)
        kernel();

        // Record stop event -- another timestamp after the kernel
        CUDA_CHECK(cudaEventRecord(stop));

        // Wait for the stop event -- CPU blocks until GPU reaches this point
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Compute elapsed time between start and stop (in milliseconds)
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // --- Step 3: Compute statistics ---

    // Sort times for median calculation
    // We make a copy so we can also compute mean from unsorted data
    float *sorted = new float[config.benchmark_runs];
    for (int i = 0; i < config.benchmark_runs; i++) {
        sorted[i] = times[i];
    }
    std::sort(sorted, sorted + config.benchmark_runs);

    // Median: middle element (or average of two middle elements)
    if (config.benchmark_runs % 2 == 0) {
        int mid = config.benchmark_runs / 2;
        result.median_ms = (sorted[mid - 1] + sorted[mid]) / 2.0f;
    } else {
        result.median_ms = sorted[config.benchmark_runs / 2];
    }

    // Min and max
    result.min_ms = sorted[0];
    result.max_ms = sorted[config.benchmark_runs - 1];

    // Mean
    float sum = 0.0f;
    for (int i = 0; i < config.benchmark_runs; i++) {
        sum += times[i];
    }
    result.mean_ms = sum / config.benchmark_runs;

    // Standard deviation
    //   stddev = sqrt( (1/N) * sum( (xi - mean)^2 ) )
    float var_sum = 0.0f;
    for (int i = 0; i < config.benchmark_runs; i++) {
        float diff = times[i] - result.mean_ms;
        var_sum += diff * diff;
    }
    result.stddev_ms = sqrtf(var_sum / config.benchmark_runs);

    // --- Step 4: Derived metrics ---

    // Bandwidth: bytes / time
    //   bytes_moved / (median_ms * 10^-3 s) / 10^9 = GB/s
    if (config.bytes_moved > 0) {
        result.bandwidth_gbps = (float)(config.bytes_moved
                                        / (result.median_ms * 1.0e6));
    }

    // GFLOPS: FLOPs / time / 10^9
    if (config.flops > 0) {
        result.gflops = (float)(config.flops / (result.median_ms * 1.0e6));
    }

    // --- Step 5: Print results ---
    printf("  %-28s", name);
    printf("  median: %7.3f ms", result.median_ms);
    printf("  mean: %7.3f ms", result.mean_ms);
    printf("  stddev: %6.3f ms", result.stddev_ms);
    printf("  [min: %7.3f, max: %7.3f]", result.min_ms, result.max_ms);

    if (config.bytes_moved > 0) {
        printf("  BW: %6.1f GB/s", result.bandwidth_gbps);
    }
    if (config.flops > 0) {
        printf("  %6.1f GFLOPS", result.gflops);
    }
    printf("\n");

    delete[] times;
    delete[] sorted;

    return result;
}


// =============================================================================
// EXAMPLE KERNEL: vector_add
// =============================================================================
//
// A simple vector addition kernel that we'll benchmark with different
// block sizes to show how configuration affects performance.
//
// C[i] = A[i] + B[i]
//
// This is memory-bound: 3 memory accesses (2 reads + 1 write) per 1 FLOP.
// Arithmetic intensity: 1 FLOP / 12 bytes = 0.083 FLOPs/byte
// Performance is dominated by memory bandwidth, not compute.
//
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}


// =============================================================================
// MAIN: Benchmark vector_add with different block sizes
// =============================================================================
int main() {
    printf("\n");
    printf("##########################################################\n");
    printf("#                                                        #\n");
    printf("#        Chapter 04: Benchmark Harness                   #\n");
    printf("#                                                        #\n");
    printf("##########################################################\n\n");

    // =========================================================================
    // Setup
    // =========================================================================
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);

    double peak_bw_gbps = 2.0 * prop.memoryClockRate * 1e3
                          * (prop.memoryBusWidth / 8.0) / 1.0e9;
    printf("  Theoretical Peak Bandwidth: %.1f GB/s\n", peak_bw_gbps);
    printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size: %d\n\n", prop.warpSize);

    // Allocate data -- 16M elements for meaningful benchmarks
    const int N = 1 << 24;   // 16,777,216 elements
    const size_t bytes = N * sizeof(float);

    printf("  Data size: %d elements (%.1f MB per array)\n\n",
           N, bytes / (1024.0 * 1024.0));

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *d_a, *d_b, *d_c;

    // Initialize with some data
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // =========================================================================
    // Benchmark: vector_add with different block sizes
    // =========================================================================
    //
    // Block size affects performance in several ways:
    //
    //   - Too small (e.g., 32): Low occupancy, SM underutilized.
    //     Each block has only 1 warp, so the SM can't hide memory
    //     latency by switching between warps.
    //
    //   - Good range (128-512): High occupancy, good latency hiding.
    //     Multiple warps per block, and enough blocks to fill all SMs.
    //
    //   - Too large (1024): Fewer blocks per SM. If the kernel uses
    //     many registers, large blocks may reduce occupancy.
    //
    //   - Must be a multiple of warp size (32) for efficiency.
    //     Non-multiples waste lanes in the last warp of each block.
    //
    printf("  ==========================================================\n");
    printf("  Benchmarking vector_add with different block sizes\n");
    printf("  ==========================================================\n\n");

    // Configuration: same for all runs
    BenchmarkConfig config;
    config.warmup_runs = 5;        // Extra warm-up for stable measurements
    config.benchmark_runs = 20;    // More runs for better statistics

    // Bytes moved per kernel call:
    //   - Read A: N * 4 bytes
    //   - Read B: N * 4 bytes
    //   - Write C: N * 4 bytes
    //   Total: 3 * N * 4 bytes
    config.bytes_moved = 3 * bytes;

    // FLOPs per kernel call:
    //   - 1 addition per element
    //   Total: N FLOPs
    config.flops = (double)N;

    // Block sizes to test: powers of 2 from 32 to 1024
    //
    // Why these values?
    //   32   = 1 warp (minimum for coalesced access)
    //   64   = 2 warps
    //   128  = 4 warps (common good choice)
    //   256  = 8 warps (very common default)
    //   512  = 16 warps
    //   1024 = 32 warps (maximum for CC 6.1)
    //
    int block_sizes[] = { 32, 64, 128, 256, 512, 1024 };
    int num_sizes = sizeof(block_sizes) / sizeof(block_sizes[0]);

    BenchmarkResult results[6];

    for (int i = 0; i < num_sizes; i++) {
        int bs = block_sizes[i];
        int grid = (N + bs - 1) / bs;

        char name[64];
        snprintf(name, sizeof(name), "vector_add (block=%d)", bs);

        // The lambda captures the kernel launch parameters.
        // Note: we do NOT call cudaDeviceSynchronize() inside the lambda --
        // the benchmark() function handles synchronization.
        results[i] = benchmark(name, config, [&]() {
            vector_add<<<grid, bs>>>(d_a, d_b, d_c, N);
        });
    }

    // =========================================================================
    // Analysis: Which block size was best?
    // =========================================================================
    printf("\n");
    printf("  ==========================================================\n");
    printf("  ANALYSIS\n");
    printf("  ==========================================================\n\n");

    // Find the best block size (lowest median time)
    int best_idx = 0;
    for (int i = 1; i < num_sizes; i++) {
        if (results[i].median_ms < results[best_idx].median_ms) {
            best_idx = i;
        }
    }

    printf("  Best block size: %d threads (%.3f ms, %.1f GB/s)\n\n",
           block_sizes[best_idx],
           results[best_idx].median_ms,
           results[best_idx].bandwidth_gbps);

    // Show bandwidth utilization for each
    printf("  Bandwidth utilization (peak = %.1f GB/s):\n\n", peak_bw_gbps);
    for (int i = 0; i < num_sizes; i++) {
        float pct = results[i].bandwidth_gbps / peak_bw_gbps * 100.0f;
        int bar_len = (int)(pct / 2.0f);  // 50 chars = 100%
        if (bar_len > 50) bar_len = 50;
        if (bar_len < 1) bar_len = 1;

        printf("    block=%4d |", block_sizes[i]);
        for (int j = 0; j < bar_len; j++) printf("#");
        for (int j = bar_len; j < 50; j++) printf(" ");
        printf("| %5.1f%% (%.1f GB/s)\n", pct, results[i].bandwidth_gbps);
    }
    printf("\n");

    // Key observations
    printf("  KEY OBSERVATIONS:\n\n");
    printf("  - vector_add is memory-bound (0.083 FLOPs/byte)\n");
    printf("  - Performance is mostly about memory bandwidth, not compute\n");
    printf("  - Block sizes 128-512 tend to perform similarly\n");
    printf("  - Very small blocks (32) may show lower performance due to\n");
    printf("    reduced occupancy and less latency hiding\n");
    printf("  - 256 is a safe default for most kernels\n\n");

    // =========================================================================
    // Demonstrate measurement quality: stddev analysis
    // =========================================================================
    printf("  MEASUREMENT QUALITY (coefficient of variation):\n\n");
    for (int i = 0; i < num_sizes; i++) {
        // CV = stddev / mean * 100%
        // CV < 5%  = very stable measurements
        // CV < 10% = acceptable
        // CV > 10% = noisy, consider more runs or larger data
        float cv = results[i].stddev_ms / results[i].mean_ms * 100.0f;
        const char *quality;
        if (cv < 2.0f) quality = "excellent";
        else if (cv < 5.0f) quality = "good";
        else if (cv < 10.0f) quality = "acceptable";
        else quality = "noisy - increase runs or data size";

        printf("    block=%4d  CV=%.1f%%  (%s)\n",
               block_sizes[i], cv, quality);
    }
    printf("\n");

    // =========================================================================
    // Usage instructions
    // =========================================================================
    printf("  ==========================================================\n");
    printf("  REUSING THIS HARNESS IN LATER CHAPTERS\n");
    printf("  ==========================================================\n\n");
    printf("  Copy the BenchmarkConfig, BenchmarkResult, and benchmark()\n");
    printf("  function to your code. Usage pattern:\n\n");
    printf("    BenchmarkConfig config;\n");
    printf("    config.bytes_moved = <total bytes read + written>;\n");
    printf("    config.flops = <total FLOPs>;\n\n");
    printf("    BenchmarkResult r = benchmark(\"name\", config, [&]() {\n");
    printf("        my_kernel<<<grid, block>>>(args...);\n");
    printf("    });\n\n");
    printf("    // Access results: r.median_ms, r.bandwidth_gbps, r.gflops\n\n");

    // =========================================================================
    // Cleanup
    // =========================================================================
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);

    return 0;
}
