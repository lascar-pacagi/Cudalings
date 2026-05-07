/*
 * stencil_1d.cu -- 1D Stencil with Shared Memory Optimization
 * ============================================================
 *
 * A 1D stencil computes each output element as a weighted sum of
 * neighboring input elements. This is the core of convolution, blurring,
 * finite differences, and many other operations.
 *
 * For a radius-R stencil:
 *   out[i] = sum(weight[j] * in[i - R + j])  for j = 0 to 2*R
 *
 * Example (radius = 3, 7-point stencil):
 *   out[i] = w0*in[i-3] + w1*in[i-2] + w2*in[i-1] + w3*in[i]
 *          + w4*in[i+1] + w5*in[i+2] + w6*in[i+3]
 *
 * This program implements two versions:
 *   1. NAIVE:     Each thread reads all neighbors from global memory
 *   2. OPTIMIZED: Load a tile + halos into shared memory, then compute
 *
 * The shared memory version is faster because of DATA REUSE:
 * each input element is used by (2*R + 1) output elements.
 * Without shared memory, it gets read from DRAM that many times.
 * With shared memory, it is read from DRAM once and from shared memory
 * (2*R + 1) times.
 *
 * Compile: nvcc -arch=sm_61 -O2 -o stencil_1d stencil_1d.cu
 * Run:     ./stencil_1d
 *
 * Target: Quadro P4000 (CC 6.1, 14 SMs)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

#define N (1 << 24)      // 4M elements (16 MB)
#define RADIUS 15         // Stencil radius: each output depends on 2*R+1 = 7 inputs
#define BLOCK_SIZE 256   // Threads per block
#define ITERS 100        // Benchmark iterations

// The stencil width = 2 * RADIUS + 1 = 7 for RADIUS=3
// Each output[i] depends on input[i-3..i+3]

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
// STENCIL WEIGHTS IN CONSTANT MEMORY
// ============================================================================
//
// The weights are the same for all threads, so constant memory is perfect.
// All threads in a warp read the same weight at the same time -> broadcast.
//
__constant__ float d_weights[2 * RADIUS + 1];


// ============================================================================
// KERNEL 1: NAIVE -- Global Memory Only
// ============================================================================
//
// Each thread computes one output element by reading (2*RADIUS+1) input
// elements directly from global memory (DRAM).
//
// Problem: massive redundant global memory reads!
//
//   Consider threads computing output[100] through output[102] with RADIUS=3:
//
//   output[100] reads: input[97] input[98] input[99] input[100] input[101] input[102] input[103]
//   output[101] reads: input[98] input[99] input[100] input[101] input[102] input[103] input[104]
//   output[102] reads: input[99] input[100] input[101] input[102] input[103] input[104] input[105]
//                       ------   --------  ---------  ---------  ---------  ---------
//                       overlap! overlap!  overlap!   overlap!   overlap!   overlap!
//
//   input[100] is read by threads computing output[97] through output[103]
//   That is 7 redundant reads from DRAM for every single element!
//
//   Total global memory reads = N * (2*RADIUS + 1) = N * 7
//   But we only NEED N reads if we cache properly.
//   Wasted bandwidth ratio: 7x
//
__global__ void stencil_naive(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float result = 0.0f;

        // Sum over the stencil window: in[i-RADIUS .. i+RADIUS]
        for (int j = -RADIUS; j <= RADIUS; j++) {
            // Clamp to array bounds (handle edges)
            int idx = i + j;
            if (idx < 0) idx = 0;
            if (idx >= n) idx = n - 1;

            // Each of these reads goes to global memory (~500 cycle latency).
            // The L1/L2 cache helps somewhat, but for large arrays the
            // working set exceeds cache capacity and we get DRAM latency.
            result += d_weights[j + RADIUS] * in[idx];
        }

        out[i] = result;
    }
}


// ============================================================================
// KERNEL 2: OPTIMIZED -- Shared Memory with Halo Cells
// ============================================================================
//
// The key insight: if we load a TILE of input data into shared memory,
// all threads in the block can read from it at ~5 cycle latency instead
// of ~500 cycle latency from DRAM.
//
// But there is a subtlety: threads at the EDGES of the tile need to read
// input elements that belong to NEIGHBORING tiles. These extra elements
// are called HALO cells (or ghost cells).
//
// ┌──────────────────────────────────────────────────────────────────┐
// │                                                                  │
// │  ASCII DIAGRAM: Shared Memory Tiling with Halo Cells             │
// │                                                                  │
// │  Global input array:                                             │
// │  ╔═══════╤═══════════════════════════════════════════╤═══════╗   │
// │  ║ ...   │  Block k's region (BLOCK_SIZE elements)   │  ...  ║   │
// │  ╚═══════╧═══════════════════════════════════════════╧═══════╝   │
// │                                                                  │
// │  What we need in shared memory for Block k:                      │
// │                                                                  │
// │  ←RADIUS→←────────── BLOCK_SIZE elements ──────────→←RADIUS→    │
// │  ┌───────┬──────────────────────────────────────────┬───────┐    │
// │  │ HALO  │            INTERIOR                      │ HALO  │    │
// │  │ LEFT  │     (main tile -- 1 element per thread)  │ RIGHT │    │
// │  │       │                                          │       │    │
// │  │ in[s] │  in[s+R]  in[s+R+1] ... in[s+R+BS-1]   │in[e-R]│    │
// │  │ ...   │                                          │ ...   │    │
// │  │in[s+R-1]                                        │ in[e] │    │
// │  └───────┴──────────────────────────────────────────┴───────┘    │
// │                                                                  │
// │  s = blockIdx.x * BLOCK_SIZE - RADIUS  (start of halo region)   │
// │  e = s + BLOCK_SIZE + 2*RADIUS - 1     (end of halo region)     │
// │                                                                  │
// │  Shared memory indices:                                          │
// │  [0]  [1]  [2] ... [R-1] [R] [R+1] ... [R+BS-1] [R+BS] ... [R+BS+R-1]
// │  ←──── left halo ──→  ←──── interior ────→  ←── right halo ──→ │
// │                                                                  │
// │  Total shared memory size: BLOCK_SIZE + 2 * RADIUS               │
// │  For BLOCK_SIZE=256, RADIUS=3: 256 + 6 = 262 floats = 1048 bytes│
// │                                                                  │
// │  Loading strategy:                                               │
// │  - Each thread loads 1 interior element (threadIdx.x + RADIUS)   │
// │  - First RADIUS threads also load the left halo                  │
// │  - Last RADIUS threads also load the right halo                  │
// │  - Then __syncthreads() to ensure all data is loaded             │
// │  - Then compute from shared memory -- ALL reads are on-chip!     │
// │                                                                  │
// │  BANDWIDTH SAVINGS:                                              │
// │  Naive:     N * (2R+1) = N * 7 global reads                     │
// │  Optimized: N + 2*R*num_blocks ≈ N global reads (+ tiny halo)   │
// │  Speedup:   up to (2R+1)x = 7x fewer global reads               │
// │                                                                  │
// └──────────────────────────────────────────────────────────────────┘
//
__global__ void stencil_shared(const float *in, float *out, int n) {
    // Shared memory: BLOCK_SIZE interior elements + RADIUS on each side
    // This array is in on-chip SRAM (~5 cycle latency).
    __shared__ float s_tile[BLOCK_SIZE + 2 * RADIUS];

    // Global index of this thread's output element
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    // Local index into shared memory (offset by RADIUS for the left halo)
    int lidx = threadIdx.x + RADIUS;

    // =========================================================================
    // STEP 1: Load interior elements into shared memory
    // =========================================================================
    // Each thread loads its own element into the interior of the tile.
    // Bounds check: if gidx >= n, load 0 (or clamp).
    if (gidx < n) {
        s_tile[lidx] = in[gidx];
    } else {
        s_tile[lidx] = 0.0f;
    }

    // =========================================================================
    // STEP 2: Load HALO elements (the tricky part!)
    // =========================================================================
    //
    // The first RADIUS threads in the block load the LEFT halo.
    // The last RADIUS threads in the block load the RIGHT halo.
    //
    // Why? Because the stencil computation for thread 0 in this block
    // needs input[blockStart - RADIUS .. blockStart - 1], which belong
    // to the previous block's region. Similarly, the last thread needs
    // input elements from the next block's region.
    //
    // LEFT HALO: threads 0..RADIUS-1 load elements to the left
    if (threadIdx.x < RADIUS) {
        int halo_idx = gidx - RADIUS;
        if (halo_idx < 0) {
            s_tile[threadIdx.x] = in[0];   // clamp at boundary
        } else {
            s_tile[threadIdx.x] = in[halo_idx];
        }
    }

    // RIGHT HALO: threads (BLOCK_SIZE - RADIUS)..BLOCK_SIZE-1 load right
    if (threadIdx.x >= BLOCK_SIZE - RADIUS) {
        int halo_idx = gidx + RADIUS;
        if (halo_idx >= n) {
            s_tile[lidx + RADIUS] = in[n - 1];   // clamp at boundary
        } else {
            s_tile[lidx + RADIUS] = in[halo_idx];
        }
    }

    // =========================================================================
    // STEP 3: SYNCHRONIZE
    // =========================================================================
    //
    // This is CRITICAL. We must wait for ALL threads in the block to finish
    // loading their elements (interior + halos) before any thread starts
    // reading from shared memory.
    //
    // Without this barrier:
    //   Thread 5 might try to read s_tile[4] (loaded by thread 1)
    //   before thread 1 has finished writing it. RACE CONDITION!
    //
    __syncthreads();

    // =========================================================================
    // STEP 4: Compute stencil from shared memory
    // =========================================================================
    //
    // Now ALL reads are from shared memory (on-chip, ~5 cycles).
    // No global memory accesses needed for the stencil computation!
    //
    if (gidx < n) {
        float result = 0.0f;

        // The stencil window in shared memory: s_tile[lidx-R .. lidx+R]
        // This is the SAME computation as the naive kernel, but all reads
        // hit shared memory instead of global memory.
        for (int j = -RADIUS; j <= RADIUS; j++) {
            result += d_weights[j + RADIUS] * s_tile[lidx + j];
        }

        out[gidx] = result;
    }
}


// ============================================================================
// CPU REFERENCE IMPLEMENTATION
// ============================================================================
//
// Simple sequential stencil for correctness verification.
//
void stencil_cpu(const float *in, float *out, const float *weights, int n, int radius) {
    for (int i = 0; i < n; i++) {
        float result = 0.0f;
        for (int j = -radius; j <= radius; j++) {
            int idx = i + j;
            if (idx < 0) idx = 0;
            if (idx >= n) idx = n - 1;
            result += weights[j + radius] * in[idx];
        }
        out[i] = result;
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
// MAIN
// ============================================================================

int main() {
    printf("============================================================\n");
    printf("1D STENCIL: Naive (Global) vs Optimized (Shared Memory)\n");
    printf("============================================================\n");
    printf("N = %d elements (%.1f MB)\n", N, (float)N * sizeof(float) / (1024*1024));
    printf("Radius = %d (stencil width = %d)\n", RADIUS, 2 * RADIUS + 1);
    printf("Block size = %d\n", BLOCK_SIZE);
    printf("Iterations = %d\n\n", ITERS);

    // --- Setup stencil weights ---
    // Use a simple averaging (box) filter: each weight = 1/(2R+1)
    // This computes a moving average -- a simple 1D blur.
    int stencil_width = 2 * RADIUS + 1;
    float h_weights[2 * RADIUS + 1];
    for (int i = 0; i < stencil_width; i++) {
        h_weights[i] = 1.0f / stencil_width;
    }

    printf("Weights (box filter / moving average):\n  ");
    for (int i = 0; i < stencil_width; i++) {
        printf("%.4f ", h_weights[i]);
    }
    printf("\n\n");

    // Copy weights to constant memory on the GPU
    CUDA_CHECK(cudaMemcpyToSymbol(d_weights, h_weights,
                                   stencil_width * sizeof(float)));

    // --- Allocate memory ---
    size_t bytes = N * sizeof(float);
    float *h_in     = (float *)malloc(bytes);
    float *h_out_cpu = (float *)malloc(bytes);
    float *h_out_naive = (float *)malloc(bytes);
    float *h_out_shared = (float *)malloc(bytes);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // --- Initialize input ---
    // Use a sine wave so the stencil (blur) has a visible effect.
    for (int i = 0; i < N; i++) {
        h_in[i] = sinf(2.0f * 3.14159f * i / 1000.0f);
    }
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // --- Grid config ---
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    GpuTimer timer;
    timer_create(&timer);

    // --- Warmup ---
    stencil_naive<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    stencil_shared<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================================================================
    // Benchmark: Naive Kernel (Global Memory Only)
    // =========================================================================
    timer_start(&timer);
    for (int iter = 0; iter < ITERS; iter++) {
        stencil_naive<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    float ms_naive = timer_stop(&timer);

    // Copy results for verification
    CUDA_CHECK(cudaMemcpy(h_out_naive, d_out, bytes, cudaMemcpyDeviceToHost));

    // Calculate effective bandwidth for naive kernel:
    // Each thread reads (2R+1) floats from global memory + writes 1 float
    // Total bytes = N * ((2R+1) + 1) * 4
    // NOTE: The L1/L2 cache reduces actual DRAM traffic, so the "effective
    // bandwidth" here is based on the REQUESTED bytes, not actual DRAM bytes.
    double naive_bytes_per_iter = (double)N * ((2*RADIUS+1) + 1) * sizeof(float);
    double naive_bw = naive_bytes_per_iter * ITERS / (ms_naive * 1e-3) / 1e9;

    printf("NAIVE  (global only):  %7.2f ms total  (%6.3f ms/iter)\n",
           ms_naive, ms_naive / ITERS);
    printf("  Requested bandwidth: %.1f GB/s (includes redundant reads)\n\n", naive_bw);

    // =========================================================================
    // Benchmark: Optimized Kernel (Shared Memory)
    // =========================================================================
    timer_start(&timer);
    for (int iter = 0; iter < ITERS; iter++) {
        stencil_shared<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    float ms_shared = timer_stop(&timer);

    // Copy results for verification
    CUDA_CHECK(cudaMemcpy(h_out_shared, d_out, bytes, cudaMemcpyDeviceToHost));

    // Effective bandwidth for shared memory version:
    // Each element loaded from global memory ONCE (+ small halo overhead)
    // Total DRAM reads ≈ N + 2*RADIUS*num_blocks ≈ N (halo is tiny)
    // Plus N writes
    double shared_bytes_per_iter = (double)N * 2 * sizeof(float);  // 1 read + 1 write
    double shared_bw = shared_bytes_per_iter * ITERS / (ms_shared * 1e-3) / 1e9;

    printf("SHARED (optimized):    %7.2f ms total  (%6.3f ms/iter)\n",
           ms_shared, ms_shared / ITERS);
    printf("  Effective bandwidth: %.1f GB/s (minimal redundant reads)\n\n", shared_bw);

    // =========================================================================
    // Speedup
    // =========================================================================
    float speedup = ms_naive / ms_shared;
    printf("SPEEDUP: %.2fx faster with shared memory!\n\n", speedup);

    printf("------------------------------------------------------------\n");
    printf("WHY SHARED MEMORY HELPS:\n");
    printf("------------------------------------------------------------\n");
    printf("Naive kernel:  each element read from global memory %d times\n",
           2 * RADIUS + 1);
    printf("               (once for each output element that uses it)\n");
    printf("               Total global reads = N * %d = %d\n",
           2*RADIUS+1, N * (2*RADIUS+1));
    printf("\n");
    printf("Shared kernel: each element read from global memory ~1 time\n");
    printf("               (loaded into shared memory, then reused)\n");
    printf("               Total global reads ≈ N = %d\n", N);
    printf("\n");
    printf("Theoretical max speedup from reduced global traffic: %dx\n",
           2*RADIUS+1);
    printf("Actual speedup: %.2fx (limited by overhead, cache effects)\n\n",
           speedup);

    // =========================================================================
    // Verify correctness against CPU
    // =========================================================================
    printf("--- Correctness Verification ---\n");

    // Run CPU reference
    stencil_cpu(h_in, h_out_cpu, h_weights, N, RADIUS);

    // Check naive kernel
    int errors_naive = 0;
    float max_err_naive = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_out_naive[i] - h_out_cpu[i]);
        if (diff > max_err_naive) max_err_naive = diff;
        if (diff > 1e-4f) errors_naive++;
    }
    printf("Naive  vs CPU: %s (max error = %.2e)\n",
           errors_naive == 0 ? "PASSED" : "FAILED", max_err_naive);

    // Check shared memory kernel
    int errors_shared = 0;
    float max_err_shared = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = fabsf(h_out_shared[i] - h_out_cpu[i]);
        if (diff > max_err_shared) max_err_shared = diff;
        if (diff > 1e-4f) errors_shared++;
    }
    printf("Shared vs CPU: %s (max error = %.2e)\n",
           errors_shared == 0 ? "PASSED" : "FAILED", max_err_shared);

    // =========================================================================
    // Cleanup
    // =========================================================================
    timer_destroy(&timer);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out_cpu);
    free(h_out_naive);
    free(h_out_shared);

    printf("\nDone!\n");
    return 0;
}

/*
 * =============================================================================
 * KEY TAKEAWAYS:
 * =============================================================================
 *
 * 1. SHARED MEMORY eliminates redundant global memory reads.
 *    For a radius-R stencil, each input element is reused (2R+1) times.
 *    Loading it into shared memory once saves (2R) global memory reads.
 *
 * 2. HALO CELLS (ghost cells) are the extra elements at the edges of each
 *    tile that belong to neighboring tiles. They must be loaded into shared
 *    memory so that threads at tile boundaries can compute correctly.
 *
 * 3. __syncthreads() is MANDATORY between loading shared memory and reading
 *    from it. Forgetting this is one of the most common CUDA bugs.
 *
 * 4. The shared memory tile size is BLOCK_SIZE + 2*RADIUS.
 *    For BLOCK_SIZE=256 and RADIUS=3: 262 floats = 1048 bytes.
 *    This is tiny compared to the 48 KB available per block.
 *
 * 5. The theoretical maximum speedup from shared memory is (2R+1)x.
 *    In practice, you get less due to:
 *    - L1/L2 cache already helps the naive version somewhat
 *    - Shared memory loading overhead (extra instructions, sync)
 *    - The computation is still memory-bound (just less so)
 *
 * 6. Larger RADIUS = more data reuse = more benefit from shared memory.
 *    Try changing RADIUS to 7 or 15 and observe the speedup increase!
 *
 * =============================================================================
 */
