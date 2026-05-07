// =============================================================================
// Chapter 04: Profiling Basics
// =============================================================================
//
// This program creates four kernels with INTENTIONALLY different performance
// characteristics, so you can see how they appear in profiling tools:
//
//   1. Compute-bound kernel  -- heavy math, light memory
//   2. Memory-bound kernel   -- heavy memory, light math
//   3. Uncoalesced kernel    -- BAD memory access pattern
//   4. Coalesced kernel      -- GOOD memory access pattern
//
// Each kernel is timed with CUDA events and analyzed for bandwidth/compute
// utilization. Then you can run the whole program under nvprof to see
// how profiling tools report these differences.
//
// Compile:  nvcc -arch=sm_61 -O2 -lineinfo -o profiling_basics profiling_basics.cu
// Run:      ./profiling_basics
// Profile:  nvprof ./profiling_basics
//           nvprof --print-gpu-trace ./profiling_basics
//           nvprof --metrics achieved_occupancy,gld_throughput,gst_throughput ./profiling_basics
//
// Hardware: Quadro P4200 (CC 6.1)
//           Theoretical peak memory bandwidth: ~192 GB/s
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>

// =============================================================================
// Error checking macro (same as error_handling.cu)
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
// KERNEL 1: Compute-Bound
// =============================================================================
//
// This kernel does a LOT of math per element but reads/writes very little data.
// It reads one float, performs ~100 transcendental operations (sin, cos, exp),
// and writes one float.
//
// Profile characteristics:
//   - HIGH compute utilization (SM units busy)
//   - LOW memory throughput (barely touching memory)
//   - Arithmetic intensity: ~100 FLOPs / 8 bytes = ~12.5 FLOPs/byte
//
// In nvprof, you'll see:
//   - Relatively long execution time
//   - Low gld_throughput and gst_throughput
//   - High sm_efficiency
//
__global__ void compute_bound_kernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];

        // Do a LOT of math -- each iteration does ~10 FLOPs
        // We do 50 iterations = ~500 floating-point operations per element
        for (int i = 0; i < 50; i++) {
            val = sinf(val) * cosf(val) + expf(-val * val);
            // sinf ~= 8 FLOPs, cosf ~= 8 FLOPs, expf ~= 8 FLOPs
            // multiply, add = 2 FLOPs
            // Total per iteration: ~26 FLOPs (approximation)
        }

        output[idx] = val;
        // Total: ~1300 FLOPs for 8 bytes transferred (1 read + 1 write)
        // Arithmetic intensity: ~1300 / 8 = 162.5 FLOPs/byte
        // This is FAR above the "ridge point" of the roofline model,
        // meaning compute is the bottleneck, not memory.
    }
}


// =============================================================================
// KERNEL 2: Memory-Bound
// =============================================================================
//
// This kernel reads data, does almost no math, and writes data.
// It's essentially a scaled copy: output[i] = input[i] * 2.0f + 1.0f
//
// Profile characteristics:
//   - LOW compute utilization (barely any math)
//   - HIGH memory throughput (limited by bandwidth)
//   - Arithmetic intensity: 2 FLOPs / 8 bytes = 0.25 FLOPs/byte
//
// In nvprof, you'll see:
//   - Short execution time (limited by bandwidth)
//   - High gld_throughput and gst_throughput (should approach ~192 GB/s)
//   - gld_efficiency close to 100% (coalesced access)
//
// A well-written memory-bound kernel like this should approach the
// theoretical peak bandwidth of the GPU.
//
__global__ void memory_bound_kernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Minimal computation: just a multiply-add
        // This takes nanoseconds, but the memory read/write takes microseconds
        output[idx] = input[idx] * 2.0f + 1.0f;

        // 2 FLOPs (multiply + add) for 8 bytes (4-byte read + 4-byte write)
        // Arithmetic intensity: 0.25 FLOPs/byte
        // This is FAR below the ridge point -- memory is the bottleneck
    }
}


// =============================================================================
// KERNEL 3: Uncoalesced (Bad) Memory Access
// =============================================================================
//
// This kernel deliberately uses a STRIDED access pattern. Instead of
// consecutive threads reading consecutive addresses (coalesced), each
// thread reads with a stride, causing multiple memory transactions
// where one would suffice.
//
// Coalesced (good):  Thread 0 reads addr 0, Thread 1 reads addr 4, ...
//                    All 32 threads in a warp read a contiguous 128-byte block
//                    --> 1 memory transaction
//
// Strided (bad):     Thread 0 reads addr 0, Thread 1 reads addr 128, ...
//                    32 threads read scattered addresses
//                    --> Up to 32 separate memory transactions!
//
// Profile characteristics:
//   - VERY low gld_efficiency (wasted bandwidth)
//   - More execution time than the coalesced version for the same work
//   - gld_throughput may look high (GPU is doing lots of transactions)
//     but effective throughput is low
//
__global__ void uncoalesced_kernel(const float *input, float *output, int n,
                                    int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Strided access: thread i accesses element (i * stride) % n
    // With stride=32, consecutive threads in a warp access elements
    // that are 32*4 = 128 bytes apart -- terrible for coalescing!
    int strided_idx = (idx * stride) % n;

    if (idx < n) {
        output[strided_idx] = input[strided_idx] * 2.0f + 1.0f;
    }

    // Why this is bad:
    //
    //   Warp of 32 threads, stride=32:
    //   Thread 0 reads input[0]     --> cache line A
    //   Thread 1 reads input[32]    --> cache line B (128 bytes later!)
    //   Thread 2 reads input[64]    --> cache line C
    //   Thread 3 reads input[96]    --> cache line D
    //   ...
    //   Thread 31 reads input[992]  --> cache line AF
    //
    //   Each thread hits a DIFFERENT cache line (128 bytes each).
    //   The GPU loads 32 * 128 = 4096 bytes but only uses 32 * 4 = 128 bytes.
    //   That's 3.125% efficiency!
    //
    // Contrast with coalesced access (stride=1):
    //   Thread 0 reads input[0]     --> cache line A
    //   Thread 1 reads input[1]     --> cache line A (same line!)
    //   Thread 2 reads input[2]     --> cache line A
    //   ...
    //   Thread 31 reads input[31]   --> cache line A
    //
    //   All 32 threads fit in ONE 128-byte cache line.
    //   The GPU loads 128 bytes and uses 128 bytes. 100% efficiency!
}


// =============================================================================
// KERNEL 4: Coalesced (Good) Memory Access
// =============================================================================
//
// This kernel does the same work as the uncoalesced kernel but with
// perfect coalescing: consecutive threads access consecutive addresses.
//
// Profile characteristics:
//   - HIGH gld_efficiency (~100%)
//   - MUCH faster than uncoalesced for the same data size
//   - Memory throughput approaches theoretical peak
//
__global__ void coalesced_kernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Perfect coalescing: thread i accesses element i
        // Consecutive threads access consecutive memory addresses
        output[idx] = input[idx] * 2.0f + 1.0f;
    }

    // This is identical math to the uncoalesced kernel,
    // but the access pattern is sequential, which the memory
    // controller can serve with minimal transactions.
}


// =============================================================================
// CUDA Event Timing Utility
// =============================================================================
//
// CUDA events measure GPU time accurately. The workflow:
//   1. Create two events (start, stop)
//   2. Record start event (places marker in GPU command stream)
//   3. Launch kernel
//   4. Record stop event
//   5. Synchronize on stop event (CPU waits for GPU to reach the marker)
//   6. Compute elapsed time between the two markers
//   7. Destroy events
//
// This measures ONLY GPU time, excluding CPU overhead, scheduling, etc.
// It's the gold standard for kernel timing.
//
struct TimingResult {
    float elapsed_ms;       // Time in milliseconds
    float bandwidth_gbps;   // Effective bandwidth in GB/s
    float gflops;           // Estimated GFLOPS
};


// =============================================================================
// Helper: time a kernel launch and compute metrics
// =============================================================================
//
// Parameters:
//   kernel_name   -- for display
//   bytes_moved   -- total bytes read + written (for bandwidth calculation)
//   flops         -- total floating-point operations (for GFLOPS calculation)
//
// The bandwidth and GFLOPS numbers let you compare against theoretical peaks:
//   P4200 peak bandwidth: ~192 GB/s
//   P4200 peak GFLOPS:    ~5.5 TFLOPS FP32 (single precision)
//
TimingResult time_kernel(const char *kernel_name,
                         size_t bytes_moved,
                         double flops,
                         float elapsed_ms) {
    TimingResult result;
    result.elapsed_ms = elapsed_ms;

    // Bandwidth = bytes moved / time
    // Convert: bytes / ms = bytes / (ms * 10^-3 s) = bytes * 10^3 / s
    // Then convert to GB/s: divide by 10^9
    // Combined: (bytes / ms) * (10^3 / 10^9) = bytes / (ms * 10^6)
    result.bandwidth_gbps = (float)(bytes_moved / (elapsed_ms * 1.0e6));

    // GFLOPS = floating-point operations / time / 10^9
    result.gflops = (float)(flops / (elapsed_ms * 1.0e6));

    printf("  %-25s  %8.3f ms  |  %7.1f GB/s  |  %7.1f GFLOPS\n",
           kernel_name, result.elapsed_ms,
           result.bandwidth_gbps, result.gflops);

    return result;
}


// =============================================================================
// MAIN
// =============================================================================
int main() {
    printf("\n");
    printf("##########################################################\n");
    printf("#                                                        #\n");
    printf("#        Chapter 04: Profiling Basics                    #\n");
    printf("#                                                        #\n");
    printf("##########################################################\n\n");

    // =========================================================================
    // Setup
    // =========================================================================
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  SMs: %d, Clock: %d MHz, Memory Clock: %d MHz\n",
           prop.multiProcessorCount,
           prop.clockRate / 1000,
           prop.memoryClockRate / 1000);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);

    // Calculate theoretical peak memory bandwidth:
    //   BW = memory_clock (Hz) * bus_width (bytes) * 2 (DDR)
    //   P4200: 1502 MHz * 256 bits / 8 * 2 = ~192 GB/s
    double peak_bw_gbps = 2.0 * prop.memoryClockRate * 1e3
                          * (prop.memoryBusWidth / 8.0) / 1.0e9;
    printf("  Theoretical Peak Bandwidth: %.1f GB/s\n\n", peak_bw_gbps);

    // Allocate data -- large enough to keep the GPU busy
    // and amortize launch overhead
    const int N = 1 << 24;  // 16M elements = 64 MB
    const size_t bytes = N * sizeof(float);
    printf("  Data size: %d elements (%.1f MB)\n\n", N, bytes / (1024.0 * 1024.0));

    float *h_input  = (float *)malloc(bytes);
    float *h_output = (float *)malloc(bytes);
    float *d_input  = nullptr;
    float *d_output = nullptr;

    // Initialize host data with some values
    for (int i = 0; i < N; i++) {
        h_input[i] = 0.5f + (i % 100) * 0.01f;  // Values in [0.5, 1.5)
    }

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed_ms;

    printf("  %-25s  %8s     |  %9s     |  %9s\n",
           "Kernel", "Time", "Bandwidth", "Compute");
    printf("  %-25s  %8s     |  %9s     |  %9s\n",
           "-------------------------", "--------",
           "---------", "---------");

    // =========================================================================
    // Kernel 1: Compute-Bound
    // =========================================================================
    //
    // Warm up the GPU (first kernel launch initializes the CUDA context,
    // which adds ~100ms overhead that we don't want to measure)
    compute_bound_kernel<<<blocks, threads>>>(d_input, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Now time it for real
    CUDA_CHECK(cudaEventRecord(start));
    compute_bound_kernel<<<blocks, threads>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Metrics:
    //   Bytes moved: N reads + N writes = 2N * 4 bytes
    //   FLOPs: ~1300 per element (50 iterations * ~26 FLOPs each)
    size_t compute_bytes = 2 * bytes;           // read + write
    double compute_flops = (double)N * 1300.0;  // ~1300 FLOPs/element
    time_kernel("Compute-bound", compute_bytes, compute_flops, elapsed_ms);

    // =========================================================================
    // Kernel 2: Memory-Bound
    // =========================================================================
    CUDA_CHECK(cudaEventRecord(start));
    memory_bound_kernel<<<blocks, threads>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Metrics:
    //   Bytes moved: N reads + N writes = 2N * 4 bytes
    //   FLOPs: 2 per element (1 multiply + 1 add)
    size_t memory_bytes = 2 * bytes;
    double memory_flops = (double)N * 2.0;
    TimingResult mem_result = time_kernel("Memory-bound", memory_bytes, memory_flops, elapsed_ms);

    // =========================================================================
    // Kernel 3: Uncoalesced Access
    // =========================================================================
    //
    // Stride of 32 means consecutive threads in a warp hit different cache lines
    int stride = 32;

    CUDA_CHECK(cudaEventRecord(start));
    uncoalesced_kernel<<<blocks, threads>>>(d_input, d_output, N, stride);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Same bytes and FLOPs as memory-bound, but much slower due to
    // wasted memory transactions from uncoalesced access
    size_t uncoal_bytes = 2 * bytes;
    double uncoal_flops = (double)N * 2.0;
    TimingResult uncoal_result = time_kernel("Uncoalesced (stride=32)", uncoal_bytes, uncoal_flops, elapsed_ms);

    // =========================================================================
    // Kernel 4: Coalesced Access
    // =========================================================================
    CUDA_CHECK(cudaEventRecord(start));
    coalesced_kernel<<<blocks, threads>>>(d_input, d_output, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    size_t coal_bytes = 2 * bytes;
    double coal_flops = (double)N * 2.0;
    TimingResult coal_result = time_kernel("Coalesced", coal_bytes, coal_flops, elapsed_ms);

    // =========================================================================
    // Analysis
    // =========================================================================
    printf("\n");
    printf("  ==========================================================\n");
    printf("  ANALYSIS\n");
    printf("  ==========================================================\n\n");

    printf("  Theoretical peak bandwidth: %.1f GB/s\n\n", peak_bw_gbps);

    // Memory-bound bandwidth utilization
    float mem_util = mem_result.bandwidth_gbps / peak_bw_gbps * 100.0f;
    printf("  Memory-bound kernel:\n");
    printf("    Achieved bandwidth: %.1f GB/s (%.1f%% of peak)\n",
           mem_result.bandwidth_gbps, mem_util);
    printf("    This is the BEST CASE for simple read-write kernels.\n");
    printf("    If this is much below peak, check for other bottlenecks.\n\n");

    // Coalesced vs uncoalesced comparison
    float speedup = uncoal_result.elapsed_ms / coal_result.elapsed_ms;
    printf("  Coalesced vs Uncoalesced:\n");
    printf("    Coalesced:    %.3f ms (%.1f GB/s)\n",
           coal_result.elapsed_ms, coal_result.bandwidth_gbps);
    printf("    Uncoalesced:  %.3f ms (%.1f GB/s)\n",
           uncoal_result.elapsed_ms, uncoal_result.bandwidth_gbps);
    printf("    Speedup from coalescing: %.1fx\n", speedup);
    printf("    --> This is why memory access patterns matter!\n\n");

    // Bandwidth utilization bar chart (ASCII)
    printf("  Bandwidth utilization (relative to peak):\n\n");

    float bars[] = { mem_result.bandwidth_gbps, coal_result.bandwidth_gbps,
                     uncoal_result.bandwidth_gbps };
    const char *names[] = { "Memory-bound ", "Coalesced    ", "Uncoalesced  " };
    for (int i = 0; i < 3; i++) {
        int bar_len = (int)(bars[i] / peak_bw_gbps * 50.0f);
        if (bar_len > 50) bar_len = 50;
        if (bar_len < 1) bar_len = 1;
        printf("    %s |", names[i]);
        for (int j = 0; j < bar_len; j++) printf("#");
        for (int j = bar_len; j < 50; j++) printf(" ");
        printf("| %.1f GB/s\n", bars[i]);
    }
    printf("    Peak         |");
    for (int j = 0; j < 50; j++) printf("=");
    printf("| %.1f GB/s\n\n", peak_bw_gbps);

    // =========================================================================
    // Profiling instructions
    // =========================================================================
    printf("  ==========================================================\n");
    printf("  HOW TO PROFILE THIS PROGRAM\n");
    printf("  ==========================================================\n\n");
    printf("  1. Quick summary (see where time is spent):\n");
    printf("     $ nvprof ./profiling_basics\n\n");
    printf("  2. Detailed GPU trace (every kernel and memcpy):\n");
    printf("     $ nvprof --print-gpu-trace ./profiling_basics\n\n");
    printf("  3. Memory efficiency metrics:\n");
    printf("     $ nvprof --metrics gld_efficiency,gst_efficiency ./profiling_basics\n");
    printf("     --> Compare uncoalesced (low) vs coalesced (high)\n\n");
    printf("  4. Throughput metrics:\n");
    printf("     $ nvprof --metrics gld_throughput,gst_throughput ./profiling_basics\n");
    printf("     --> Shows actual bytes/sec for loads and stores\n\n");
    printf("  5. Occupancy:\n");
    printf("     $ nvprof --metrics achieved_occupancy ./profiling_basics\n");
    printf("     --> Ratio of active warps to max possible warps\n\n");
    printf("  6. SM utilization:\n");
    printf("     $ nvprof --metrics sm_efficiency ./profiling_basics\n");
    printf("     --> Percentage of time at least one warp is active\n\n");

    // =========================================================================
    // Cleanup
    // =========================================================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    printf("  Done! Try the nvprof commands above to explore further.\n\n");

    return 0;
}
