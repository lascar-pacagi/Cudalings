/*
 * Chapter 06: Bank Conflicts in Shared Memory
 * =============================================
 *
 * GPU: Quadro P4200 (CC 6.1, 18 SMs, 48 KB shared mem per SM)
 * CUDA: 11.7
 *
 * Shared memory is organized into 32 BANKS, each 4 bytes wide.
 * When threads in a warp access different addresses that map to the
 * SAME bank, those accesses are serialized -- this is a BANK CONFLICT.
 *
 * Bank number = (byte_address / 4) % 32
 *
 * For a float array smem[]:
 *   smem[i] is in bank (i % 32)
 *
 * This program demonstrates:
 *   1. NO bank conflicts (stride-1 access)
 *   2. 2-way bank conflicts (stride-2 access)
 *   3. 32-way bank conflicts (stride-32, all same bank)
 *   4. Broadcast (all threads same address -- no conflict)
 *   5. Padding trick to fix conflicts in 2D arrays
 *
 * Compile: nvcc -arch=sm_61 -O2 -lineinfo -o bank_conflicts bank_conflicts.cu
 * Profile: nvprof --metrics shared_load_transactions_per_request ./bank_conflicts
 */

#include <stdio.h>
#include <cuda_runtime.h>

// ============================================================================
// Error checking macro
// ============================================================================
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ============================================================================
// Constants
// ============================================================================

// We use a fixed block size of 256 threads (8 warps).
// Each thread does ITERATIONS rounds of read-modify-write to make
// the timing measurable. Without many iterations, the kernel launch
// overhead would dominate.
#define BLOCK_SIZE  256
#define ITERATIONS  10000

// Number of elements in shared memory. Must be large enough to
// accommodate stride-32 access: thread 31 accesses index 31*32 = 992.
// Round up to 1024.
#define SMEM_SIZE   1024

// ============================================================================
// Kernel 1: NO bank conflicts (stride-1)
// ============================================================================
/*
 * Access pattern:  smem[threadIdx.x]
 *
 * Thread 0 -> smem[0]  -> bank 0
 * Thread 1 -> smem[1]  -> bank 1
 * Thread 2 -> smem[2]  -> bank 2
 *   ...
 * Thread 31 -> smem[31] -> bank 31
 *
 * Every thread hits a DIFFERENT bank. Perfect parallel access.
 * Expected: 1 transaction per request.
 */
__global__ void stride1_no_conflict(float *output) {
    // Static shared memory allocation -- size known at compile time
    __shared__ float smem[SMEM_SIZE];

    int tid = threadIdx.x;

    // Initialize shared memory
    smem[tid] = (float)tid;
    __syncthreads();   // Ensure all writes complete before reads begin

    float val = 0.0f;
    for (int i = 0; i < ITERATIONS; i++) {
        // Stride-1 access: each thread reads its own index
        // Thread tid reads smem[tid], which is in bank (tid % 32)
        // Since tid 0..31 map to banks 0..31: NO CONFLICT
        val += smem[tid];
    }

    // Write result to prevent compiler from optimizing away the loop
    if (tid == 0) output[0] = val;
}

// ============================================================================
// Kernel 2: 2-WAY bank conflicts (stride-2)
// ============================================================================
/*
 * Access pattern:  smem[threadIdx.x * 2]
 *
 * Thread 0  -> smem[0]   -> bank 0
 * Thread 1  -> smem[2]   -> bank 2
 * Thread 2  -> smem[4]   -> bank 4
 *   ...
 * Thread 15 -> smem[30]  -> bank 30
 * Thread 16 -> smem[32]  -> bank 0   <-- CONFLICT with thread 0!
 * Thread 17 -> smem[34]  -> bank 2   <-- CONFLICT with thread 1!
 *   ...
 * Thread 31 -> smem[62]  -> bank 30  <-- CONFLICT with thread 15!
 *
 * Two threads hit each even-numbered bank. Odd banks unused.
 * Expected: 2 transactions per request (2x slowdown).
 */
__global__ void stride2_two_way_conflict(float *output) {
    __shared__ float smem[SMEM_SIZE];

    int tid = threadIdx.x;

    // Initialize (need to fill enough elements for stride-2 access)
    if (tid * 2 < SMEM_SIZE) smem[tid * 2] = (float)tid;
    __syncthreads();

    float val = 0.0f;
    for (int i = 0; i < ITERATIONS; i++) {
        // Stride-2: threads 0 and 16 both hit bank 0
        // This creates a 2-way conflict on every even bank
        int index = (tid % 32) * 2;   // Keep within a warp's pattern
        val += smem[index];
    }

    if (tid == 0) output[0] = val;
}

// ============================================================================
// Kernel 3: 32-WAY bank conflicts (stride-32) -- WORST CASE
// ============================================================================
/*
 * Access pattern:  smem[threadIdx.x * 32]
 *
 * Thread 0  -> smem[0]   -> bank 0
 * Thread 1  -> smem[32]  -> bank 0   <-- SAME BANK!
 * Thread 2  -> smem[64]  -> bank 0   <-- SAME BANK!
 *   ...
 * Thread 31 -> smem[992] -> bank 0   <-- ALL BANK 0!
 *
 * ALL 32 threads hit bank 0. Complete serialization.
 * Expected: 32 transactions per request (32x slowdown!).
 *
 * NOTE: This is NOT a broadcast because each thread accesses a
 * DIFFERENT address in bank 0. Broadcast only works when all
 * threads read the SAME address.
 */
__global__ void stride32_full_conflict(float *output) {
    __shared__ float smem[SMEM_SIZE];

    int tid = threadIdx.x;

    // Initialize
    if (tid * 32 < SMEM_SIZE) smem[tid * 32] = (float)tid;
    __syncthreads();

    float val = 0.0f;
    for (int i = 0; i < ITERATIONS; i++) {
        // Stride-32: ALL threads in the warp hit bank 0
        // This is the absolute worst case for bank conflicts
        int index = (tid % 32) * 32;
        if (index < SMEM_SIZE) val += smem[index];
    }

    if (tid == 0) output[0] = val;
}

// ============================================================================
// Kernel 4: BROADCAST -- all threads read the SAME address (no conflict!)
// ============================================================================
/*
 * Access pattern:  smem[0] for ALL threads
 *
 * All 32 threads read the exact same address (smem[0]).
 * The hardware detects this and BROADCASTS the value to all threads.
 * This is a special case: same address = no conflict.
 *
 * Expected: 1 transaction per request.
 */
__global__ void broadcast_no_conflict(float *output) {
    __shared__ float smem[SMEM_SIZE];

    int tid = threadIdx.x;

    if (tid == 0) smem[0] = 42.0f;
    __syncthreads();

    float val = 0.0f;
    for (int i = 0; i < ITERATIONS; i++) {
        // ALL threads read the exact SAME address
        // Hardware broadcasts the value -- no serialization
        val += smem[0];
    }

    if (tid == 0) output[0] = val;
}

// ============================================================================
// Kernel 5: 2D array column access WITHOUT padding -- bank conflicts
// ============================================================================
/*
 * This is the REAL-WORLD scenario where bank conflicts bite you.
 * Consider a 32x32 shared memory tile used in matrix transpose:
 *
 *   __shared__ float tile[32][32];
 *
 * Reading ROW-wise:    tile[row][threadIdx.x]  -- stride-1, no conflict
 * Reading COLUMN-wise: tile[threadIdx.x][col]  -- stride-32, 32-way conflict!
 *
 * Why? tile[0][col] and tile[1][col] are 32 floats apart = 128 bytes apart.
 * 128 / 4 = 32, so they differ by exactly 32 in the bank formula:
 *   bank of tile[i][col] = (i * 32 + col) % 32 = col  (for all i!)
 * Every row maps column `col` to the SAME bank.
 */
#define TILE_DIM 32

__global__ void column_access_no_padding(float *output) {
    // 32x32 shared memory tile -- NO padding
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int tid = threadIdx.x;
    if (tid >= TILE_DIM) return;  // Block is BLOCK_SIZE=256, tile only has 32 rows
    int row = tid;

    // Initialize the tile (row-wise, no conflicts)
    for (int col = 0; col < TILE_DIM; col++) {
        tile[row][col] = (float)(row * TILE_DIM + col);
    }
    __syncthreads();

    float val = 0.0f;
    for (int i = 0; i < ITERATIONS; i++) {
        // COLUMN access: tile[threadIdx.x][fixed_col]
        // Thread 0 reads tile[0][0] -> bank 0
        // Thread 1 reads tile[1][0] -> bank (32 % 32) = 0  CONFLICT!
        // Thread 2 reads tile[2][0] -> bank (64 % 32) = 0  CONFLICT!
        //   ...
        // ALL threads hit bank 0 -> 32-way conflict!
        int col = i % TILE_DIM;
        val += tile[row][col];   // This is actually row access (no conflict)
    }

    // Now do the problematic column access
    for (int i = 0; i < ITERATIONS; i++) {
        int col = 0;
        val += tile[tid][col];   // Column access -> 32-way conflict!
    }

    if (tid == 0) output[0] = val;
}

// ============================================================================
// Kernel 6: 2D array column access WITH padding -- no bank conflicts!
// ============================================================================
/*
 * THE PADDING TRICK:
 *
 *   __shared__ float tile[32][32 + 1];   // <-- +1 padding per row!
 *
 * Now each row is 33 floats wide (132 bytes). The bank mapping changes:
 *   tile[0][0] -> bank (0 * 33 + 0) % 32 = 0
 *   tile[1][0] -> bank (1 * 33 + 0) % 32 = 1   <-- different bank!
 *   tile[2][0] -> bank (2 * 33 + 0) % 32 = 2   <-- different bank!
 *   ...
 *   tile[31][0] -> bank (31 * 33 + 0) % 32 = 31  <-- different bank!
 *
 * Each row's column 0 is in a different bank. NO CONFLICTS!
 * The padding element wastes 32 * 4 = 128 bytes per tile.
 * Totally worth it for eliminating 32-way conflicts.
 */
__global__ void column_access_with_padding(float *output) {
    // 32x(32+1) shared memory tile -- WITH padding
    // The +1 shifts each row by one bank relative to the previous row
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // <-- THE FIX!

    int tid = threadIdx.x;
    if (tid >= TILE_DIM) return;  // Block is BLOCK_SIZE=256, tile only has 32 rows
    int row = tid;

    // Initialize the tile
    for (int col = 0; col < TILE_DIM; col++) {
        tile[row][col] = (float)(row * TILE_DIM + col);
    }
    __syncthreads();

    float val = 0.0f;
    for (int i = 0; i < ITERATIONS; i++) {
        int col = i % TILE_DIM;
        val += tile[row][col];
    }

    // Column access -- now conflict-free thanks to padding!
    for (int i = 0; i < ITERATIONS; i++) {
        int col = 0;
        val += tile[tid][col];   // Column access -> NO conflict with padding!
    }

    if (tid == 0) output[0] = val;
}

// ============================================================================
// Timing utility
// ============================================================================
typedef void (*kernel_fn)(float *);

float benchmark_kernel(kernel_fn kernel, const char *name, float *d_output) {
    // Warmup run
    kernel<<<1, BLOCK_SIZE>>>(d_output);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed run using CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Run multiple launches for more stable timing
    const int LAUNCHES = 20;

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < LAUNCHES; i++) {
        kernel<<<1, BLOCK_SIZE>>>(d_output);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= LAUNCHES;  // Average per launch

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("  %-35s %8.3f ms\n", name, ms);
    return ms;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("Chapter 06: Bank Conflicts in Shared Memory\n");
    printf("============================================\n\n");

    // Print device info
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shared memory per SM: %zu bytes (%zu KB)\n",
           prop.sharedMemPerMultiprocessor,
           prop.sharedMemPerMultiprocessor / 1024);
    printf("Shared memory per block: %zu bytes (%zu KB)\n",
           prop.sharedMemPerBlock,
           prop.sharedMemPerBlock / 1024);
    printf("Warp size: %d\n\n", prop.warpSize);

    // Allocate output buffer on device (just to prevent compiler optimization)
    float *d_output;
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float)));

    // ========================================================================
    // Part 1: Stride-based access patterns
    // ========================================================================
    printf("Part 1: Stride-Based Access Patterns (%d iterations each)\n", ITERATIONS);
    printf("-----------------------------------------------------------\n");
    printf("  Each warp has 32 threads accessing shared memory.\n");
    printf("  Bank number = (element_index) %% 32\n\n");

    float t_stride1  = benchmark_kernel(stride1_no_conflict,       "Stride-1 (no conflict)",   d_output);
    float t_stride2  = benchmark_kernel(stride2_two_way_conflict,  "Stride-2 (2-way conflict)", d_output);
    float t_stride32 = benchmark_kernel(stride32_full_conflict,    "Stride-32 (32-way conflict)", d_output);
    float t_bcast    = benchmark_kernel(broadcast_no_conflict,     "Broadcast (same address)",  d_output);

    printf("\n  Relative to stride-1 (no conflict):\n");
    printf("    Stride-1:   %.2fx (baseline)\n", t_stride1 / t_stride1);
    printf("    Stride-2:   %.2fx (expected ~2x)\n", t_stride2 / t_stride1);
    printf("    Stride-32:  %.2fx (expected ~32x)\n", t_stride32 / t_stride1);
    printf("    Broadcast:  %.2fx (expected ~1x)\n", t_bcast / t_stride1);

    // ========================================================================
    // Part 2: The padding trick for 2D arrays
    // ========================================================================
    printf("\n\nPart 2: The Padding Trick for 2D Shared Memory Arrays\n");
    printf("------------------------------------------------------\n");
    printf("  32x32 tile: column access causes 32-way bank conflicts.\n");
    printf("  32x33 tile (+1 padding): shifts each row, no conflicts.\n\n");

    float t_nopad  = benchmark_kernel(column_access_no_padding,    "Column access (no padding)",   d_output);
    float t_padded = benchmark_kernel(column_access_with_padding,  "Column access (with padding)", d_output);

    printf("\n  Speedup from padding: %.2fx\n", t_nopad / t_padded);

    // ========================================================================
    // Summary
    // ========================================================================
    printf("\n\nSummary\n");
    printf("-------\n");
    printf("  Bank conflicts serialize shared memory accesses within a warp.\n");
    printf("  - Stride-1: each thread hits a different bank -> ideal.\n");
    printf("  - Stride-2: two threads per bank -> 2x penalty.\n");
    printf("  - Stride-32: all threads same bank -> 32x penalty.\n");
    printf("  - Broadcast: same address -> hardware optimization, no penalty.\n");
    printf("  - Padding (+1): shifts bank mapping per row, eliminates conflicts.\n");
    printf("\n  Use nvprof to verify:\n");
    printf("    nvprof --metrics shared_load_transactions_per_request ./bank_conflicts\n\n");

    CHECK_CUDA(cudaFree(d_output));
    return 0;
}
