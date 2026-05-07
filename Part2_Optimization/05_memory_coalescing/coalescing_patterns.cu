/*
 * coalescing_patterns.cu -- Demonstrating Memory Coalescing Impact
 * =================================================================
 *
 * This program benchmarks five different memory access patterns and shows
 * the dramatic bandwidth differences caused by coalescing (or lack thereof).
 *
 * The five patterns:
 *   1. Coalesced:     thread i reads data[i]           (consecutive)
 *   2. Stride-2:      thread i reads data[i*2]         (skip every other)
 *   3. Stride-32:     thread i reads data[i*32]        (worst case per warp)
 *   4. Random:        thread i reads data[random[i]]   (scattered)
 *   5. Misaligned:    thread i reads data[i + 1]       (off by one element)
 *
 * All kernels do the same amount of "work" (one read + one write per thread),
 * so any performance difference is PURELY due to memory access patterns.
 *
 * Expected results on Quadro P4200 (~192 GB/s theoretical):
 *   Coalesced:    ~160-170 GB/s  (near peak)
 *   Stride-2:     ~60-90 GB/s   (roughly half)
 *   Stride-32:    ~5-10 GB/s    (catastrophic)
 *   Random:        ~5-15 GB/s   (catastrophic)
 *   Misaligned:   ~100-140 GB/s (moderate penalty)
 *
 * Compile: nvcc -arch=sm_61 -O2 -lineinfo -o coalescing_patterns coalescing_patterns.cu
 * Run:     ./coalescing_patterns
 *
 * Target: Quadro P4200 (CC 6.1, 18 SMs)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

/*
 * We need a large array so:
 *   1. It does NOT fit in L2 cache (~2 MB) -- we measure DRAM bandwidth.
 *   2. The strided kernels (stride=32) still have enough elements.
 *
 * N = 16M elements = 64 MB per array.
 * For stride-32, each thread reads one of N/32 = 512K elements spread
 * across the full 64 MB, so we still get meaningful throughput numbers.
 */
#define N (16 * 1024 * 1024)   // 16M elements

#define BLOCK_SIZE 256         // Threads per block (8 warps)
#define WARMUP_ITERS 5         // Warmup runs (not timed)
#define BENCH_ITERS 20         // Timed benchmark iterations

// ----------------------------------------------------------------------------
// Error checking macro (from Chapter 04)
// ----------------------------------------------------------------------------
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
// KERNEL 1: PERFECTLY COALESCED ACCESS
// ============================================================================
/*
 * Each thread reads and writes at its global index. Consecutive threads
 * access consecutive memory locations.
 *
 * Memory access pattern for one warp (32 threads):
 *
 *   Thread:  T0   T1   T2   T3   T4  ...  T31
 *            |    |    |    |    |         |
 *            v    v    v    v    v         v
 *   Memory: [f0 | f1 | f2 | f3 | f4 |...| f31]
 *           |<------- 128 bytes = 1 cache line -------->|
 *
 *   Transactions: 1 read + 1 write = 2 total per warp
 *   Bytes fetched: 128 (read) + 128 (write) = 256
 *   Bytes needed:  128 (read) + 128 (write) = 256
 *   Efficiency: 100%
 */
__global__ void kernel_coalesced(const float *in, float *out, int n) {
    // Standard global index calculation
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop: handles arrays larger than the grid
    int stride = gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        // in[i] and out[i] -- consecutive threads, consecutive addresses
        // This is the IDEAL access pattern.
        out[i] = in[i] * 2.0f;
    }
}


// ============================================================================
// KERNEL 2: STRIDE-2 ACCESS (Half Bandwidth)
// ============================================================================
/*
 * Each thread accesses every OTHER element. This means a warp's 32 addresses
 * span 256 bytes instead of 128, requiring 2 cache line fetches.
 *
 * Memory access pattern for one warp:
 *
 *   Thread:  T0        T1        T2        T3        ...
 *            |         |         |         |
 *            v         v         v         v
 *   Memory: [f0 | -- | f2 | -- | f4 | -- | f6 | -- | ...]
 *           |<--- cache line 1 --->|<--- cache line 2 --->|
 *
 *   32 threads span 32 * 2 * 4 = 256 bytes = 2 cache lines
 *   Transactions: 2 reads + 2 writes = 4 per warp
 *   Bytes fetched: 256 + 256 = 512
 *   Bytes needed:  128 + 128 = 256
 *   Efficiency: 50%
 *
 * NOTE: We only process N/2 elements to ensure we don't go out of bounds.
 */
__global__ void kernel_stride2(const float *in, float *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread processes one element, but at position tid*2
    for (int i = tid; i < n / 2; i += stride) {
        int idx = i * 2;    // Stride of 2: skip every other element
        out[idx] = in[idx] * 2.0f;
    }
}


// ============================================================================
// KERNEL 3: STRIDE-32 ACCESS (Worst Case for a Warp)
// ============================================================================
/*
 * Each thread accesses elements 32 positions apart. This is the WORST case
 * because each thread in a warp hits a DIFFERENT cache line.
 *
 * Memory access pattern for one warp:
 *
 *   T0             T1             T2             T3
 *   |              |              |              |
 *   v              v              v              v
 *   [f0 |..........|f32|..........|f64|..........|f96|...]
 *   |  128 bytes  |   128 bytes  |   128 bytes  |
 *   cache line 0   cache line 1   cache line 2   cache line 3
 *
 *   Each of the 32 threads in a warp hits a SEPARATE cache line!
 *   Transactions: 32 reads + 32 writes = 64 per warp
 *   Bytes fetched: 32 * 128 + 32 * 128 = 8192
 *   Bytes needed:  128 + 128 = 256
 *   Efficiency: 256 / 8192 = 3.1%
 *
 *   This is why stride-32 access is catastrophically slow.
 */
__global__ void kernel_stride32(const float *in, float *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n / 32; i += stride) {
        int idx = i * 32;   // Stride of 32: each warp thread hits different cache line
        out[idx] = in[idx] * 2.0f;
    }
}


// ============================================================================
// KERNEL 4: RANDOM ACCESS (Pathological Case)
// ============================================================================
/*
 * Each thread reads from a random location. This is typical of hash table
 * lookups, graph traversals, and sparse matrix operations.
 *
 * Memory access pattern for one warp:
 *
 *   T0       T1    T2         T3      ...
 *   |        |     |          |
 *   v        v     v          v
 *   [.......f917.....f3......f42081.......f7.......]
 *
 *   Each thread likely hits a different cache line.
 *   Transactions: up to 32 reads + 32 writes per warp
 *   Efficiency: ~3% (similar to stride-32, but with added TLB and cache misses)
 *
 * We use a pre-computed random index array to avoid overhead from
 * random number generation inside the kernel.
 */
__global__ void kernel_random(const float *in, float *out,
                              const int *indices, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        int idx = indices[i];   // Random index -- scattered access
        out[i] = in[idx] * 2.0f;
        // Note: out[i] IS coalesced (write), but in[idx] is NOT (read).
        // The read is the bottleneck here.
    }
}


// ============================================================================
// KERNEL 5: MISALIGNED ACCESS (Off By One)
// ============================================================================
/*
 * Threads access consecutive elements, but starting at an offset of 1.
 * The access IS consecutive (stride-1), but crosses a cache line boundary.
 *
 * Memory access pattern for one warp:
 *
 *   Thread:  T0   T1   T2   T3   ...  T31
 *            |    |    |    |         |
 *            v    v    v    v         v
 *   Memory: [--| f1 | f2 | f3 | ... | f31 | f32 ]
 *           |<--- cache line 1 ----------->|<-CL2->|
 *                                            ^^
 *                                            T31 lands in the next cache line!
 *
 *   Without the misalignment: 1 cache line (128 bytes)
 *   With misalignment of 1 float: warp spans 132 bytes -> 2 cache lines
 *
 *   Transactions: 2 reads instead of 1 (per warp iteration)
 *   Efficiency: ~50-75% (less severe than strides, because it's only the
 *               boundary warps that straddle two cache lines)
 *
 *   NOTE: Modern GPUs (CC 3.5+) handle misalignment better thanks to
 *   L1 caching. On your P4200, the penalty is moderate (~10-30%).
 */
__global__ void kernel_misaligned(const float *in, float *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n - 1; i += stride) {
        // Access at i+1: consecutive but shifted by 4 bytes from alignment
        out[i + 1] = in[i + 1] * 2.0f;
    }
}


// ============================================================================
// BENCHMARKING INFRASTRUCTURE
// ============================================================================

/*
 * Struct to hold benchmark results for each pattern.
 * We compute "effective bandwidth" = data_moved / time.
 *
 * "data_moved" is the MINIMUM bytes the kernel needs (not what the hardware
 * actually fetches). This lets us compare kernels fairly -- they all need
 * the same amount of data, but some cause more actual memory traffic.
 */
typedef struct {
    const char *name;
    float time_ms;
    float bandwidth_gbps;
} BenchResult;

/*
 * Print an ASCII bar chart comparing the bandwidth of all patterns.
 * This makes the coalescing impact immediately visible.
 */
void print_bar_chart(BenchResult *results, int count, float peak_bw) {
    printf("\n");
    printf("==========================================================\n");
    printf("  MEMORY COALESCING IMPACT -- Effective Bandwidth (GB/s)\n");
    printf("==========================================================\n");
    printf("  Peak theoretical: %.0f GB/s\n", peak_bw);
    printf("----------------------------------------------------------\n");

    // Find max bandwidth for scaling the bars
    float max_bw = 0.0f;
    for (int i = 0; i < count; i++) {
        if (results[i].bandwidth_gbps > max_bw) {
            max_bw = results[i].bandwidth_gbps;
        }
    }

    // Bar width in characters
    const int bar_max = 40;

    for (int i = 0; i < count; i++) {
        // Compute bar length proportional to bandwidth
        int bar_len = (int)(results[i].bandwidth_gbps / max_bw * bar_max);
        if (bar_len < 1) bar_len = 1;

        // Print label (fixed width)
        printf("  %-14s |", results[i].name);

        // Print bar
        for (int j = 0; j < bar_len; j++) {
            printf("#");
        }

        // Print value after bar
        printf(" %.1f GB/s", results[i].bandwidth_gbps);

        // Print efficiency vs peak
        float efficiency = results[i].bandwidth_gbps / peak_bw * 100.0f;
        printf(" (%.0f%%)", efficiency);
        printf("\n");
    }

    printf("----------------------------------------------------------\n");
    printf("  Coalesced vs Stride-32 speedup: %.1fx\n",
           results[0].bandwidth_gbps / results[2].bandwidth_gbps);
    printf("  Coalesced vs Random speedup:    %.1fx\n",
           results[0].bandwidth_gbps / results[3].bandwidth_gbps);
    printf("==========================================================\n\n");
}


// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("Memory Coalescing Patterns Benchmark\n");
    printf("=====================================\n");
    printf("Array size: %d elements (%.1f MB)\n", N, (float)N * 4 / (1024 * 1024));
    printf("Block size: %d threads\n\n", BLOCK_SIZE);

    // -----------------------------------------------------------------------
    // Allocate device memory
    // -----------------------------------------------------------------------
    float *d_in, *d_out;
    int   *d_indices;

    CUDA_CHECK(cudaMalloc(&d_in,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));

    // -----------------------------------------------------------------------
    // Initialize data on host, then copy to device
    // -----------------------------------------------------------------------
    float *h_in     = (float *)malloc(N * sizeof(float));
    int   *h_indices = (int *)malloc(N * sizeof(int));

    // Fill input with sequential values
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i;
    }

    // Generate random indices for the random-access kernel
    // Use a simple LCG (linear congruential generator) for reproducibility
    srand(42);
    for (int i = 0; i < N; i++) {
        h_indices[i] = rand() % N;
    }

    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

    // -----------------------------------------------------------------------
    // Determine launch configuration
    // -----------------------------------------------------------------------
    /*
     * We use enough blocks to keep all 18 SMs busy but not so many that
     * the grid-stride loop only runs once. A good rule:
     *   blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE
     * But we cap it to avoid overly large grids for strided kernels.
     */
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // -----------------------------------------------------------------------
    // CUDA events for timing
    // -----------------------------------------------------------------------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Storage for results
    BenchResult results[5];

    // -----------------------------------------------------------------------
    // Benchmark helper: times a kernel over BENCH_ITERS iterations
    // -----------------------------------------------------------------------
    // We use macros here because kernel launches have different signatures.
    // A function pointer approach is possible but more complex.

    float elapsed_ms;

    // ===== KERNEL 1: COALESCED =====
    printf("Benchmarking coalesced access...\n");

    // Warmup (populates caches, avoids first-launch overhead)
    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_coalesced<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        kernel_coalesced<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Bandwidth = bytes_moved / time
    // Each element: 4 bytes read + 4 bytes written = 8 bytes
    // Total: N elements * 8 bytes * BENCH_ITERS iterations
    results[0].name = "Coalesced";
    results[0].time_ms = elapsed_ms / BENCH_ITERS;
    results[0].bandwidth_gbps = ((float)N * 8 * BENCH_ITERS) /
                                 (elapsed_ms / 1000.0f) / 1e9;

    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s\n",
           results[0].time_ms, results[0].bandwidth_gbps);

    // ===== KERNEL 2: STRIDE-2 =====
    printf("Benchmarking stride-2 access...\n");

    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_stride2<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        kernel_stride2<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // N/2 elements processed (each at stride 2), 8 bytes per element
    results[1].name = "Stride-2";
    results[1].time_ms = elapsed_ms / BENCH_ITERS;
    results[1].bandwidth_gbps = ((float)(N / 2) * 8 * BENCH_ITERS) /
                                 (elapsed_ms / 1000.0f) / 1e9;

    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s\n",
           results[1].time_ms, results[1].bandwidth_gbps);

    // ===== KERNEL 3: STRIDE-32 =====
    printf("Benchmarking stride-32 access...\n");

    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_stride32<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        kernel_stride32<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // N/32 elements processed, 8 bytes per element
    results[2].name = "Stride-32";
    results[2].time_ms = elapsed_ms / BENCH_ITERS;
    results[2].bandwidth_gbps = ((float)(N / 32) * 8 * BENCH_ITERS) /
                                 (elapsed_ms / 1000.0f) / 1e9;

    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s\n",
           results[2].time_ms, results[2].bandwidth_gbps);

    // ===== KERNEL 4: RANDOM =====
    printf("Benchmarking random access...\n");

    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_random<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, d_indices, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        kernel_random<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, d_indices, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // N elements, but read is random (index array read is coalesced, data read is random)
    // We count 4 bytes for the random read + 4 bytes for coalesced write = 8 bytes
    // Plus 4 bytes for the index read = 12 bytes total
    results[3].name = "Random";
    results[3].time_ms = elapsed_ms / BENCH_ITERS;
    results[3].bandwidth_gbps = ((float)N * 12 * BENCH_ITERS) /
                                 (elapsed_ms / 1000.0f) / 1e9;

    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s\n",
           results[3].time_ms, results[3].bandwidth_gbps);

    // ===== KERNEL 5: MISALIGNED =====
    printf("Benchmarking misaligned access...\n");

    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_misaligned<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        kernel_misaligned<<<grid_size, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // N-1 elements, 8 bytes each
    results[4].name = "Misaligned";
    results[4].time_ms = elapsed_ms / BENCH_ITERS;
    results[4].bandwidth_gbps = ((float)(N - 1) * 8 * BENCH_ITERS) /
                                 (elapsed_ms / 1000.0f) / 1e9;

    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s\n",
           results[4].time_ms, results[4].bandwidth_gbps);

    // -----------------------------------------------------------------------
    // Print ASCII bar chart
    // -----------------------------------------------------------------------
    float peak_bw = 192.0f;  // Quadro P4200 theoretical peak (GB/s)
    print_bar_chart(results, 5, peak_bw);

    // -----------------------------------------------------------------------
    // Key takeaways
    // -----------------------------------------------------------------------
    printf("KEY TAKEAWAYS:\n");
    printf("  - Coalesced access achieves near-peak bandwidth.\n");
    printf("  - Stride-2 roughly halves bandwidth (2x cache lines fetched).\n");
    printf("  - Stride-32 is catastrophic: each warp thread hits a different\n");
    printf("    cache line, wasting 97%% of fetched data.\n");
    printf("  - Random access is similarly bad, with added TLB pressure.\n");
    printf("  - Misalignment has a moderate penalty on modern GPUs (CC 3.5+)\n");
    printf("    thanks to L1 caching, but still measurable.\n\n");

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_indices));
    free(h_in);
    free(h_indices);

    return 0;
}
