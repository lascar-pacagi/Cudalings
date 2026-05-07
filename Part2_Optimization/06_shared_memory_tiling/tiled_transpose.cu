/*
 * Chapter 06: Tiled Matrix Transpose
 * ====================================
 *
 * GPU: Quadro P4200 (CC 6.1, 18 SMs, 48 KB shared mem per SM)
 * CUDA: 11.7
 *
 * Matrix transpose is THE canonical example of shared memory optimization.
 * The problem: transposing a matrix requires reading rows and writing columns
 * (or vice versa). One of these access patterns is non-coalesced.
 *
 * This program implements three versions:
 *   1. NAIVE transpose:         coalesced reads, non-coalesced writes
 *   2. TILED transpose:         coalesced reads AND writes (shared memory)
 *   3. TILED + PADDED transpose: same as #2, plus bank conflict elimination
 *
 * The tiled approach uses shared memory as a staging area:
 *   - Load a tile from the input (coalesced row reads)
 *   - Store it in shared memory
 *   - Read from shared memory in transposed order
 *   - Write the tile to the output (coalesced row writes)
 *
 * Both the global reads AND writes are coalesced, giving a large speedup.
 *
 * Compile: nvcc -arch=sm_61 -O2 -lineinfo -o tiled_transpose tiled_transpose.cu
 * Profile: nvprof --metrics gld_efficiency,gst_efficiency ./tiled_transpose
 */

#include <stdio.h>
#include <stdlib.h>
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

// Tile dimension. 32x32 = 1024 threads per block, 4 KB shared memory.
// This is the sweet spot: matches warp size, fits many blocks per SM.
#define TILE_DIM    32

// Block rows: we use a 32x8 thread block. Each thread transposes
// TILE_DIM/BLOCK_ROWS = 4 elements. This means fewer threads per block,
// allowing more blocks per SM (better occupancy), while each thread
// does more work (better instruction-level parallelism).
#define BLOCK_ROWS  8

// Matrix size (must be multiple of TILE_DIM for simplicity)
#define N           4096

// ============================================================================
// Kernel 1: NAIVE transpose
// ============================================================================
/*
 * The simplest transpose: read from input[y][x], write to output[x][y].
 *
 * MEMORY ACCESS PATTERN:
 *
 *   Input matrix (row-major):          Output matrix (row-major):
 *   +--+--+--+--+--+--+               +--+--+--+--+--+--+
 *   | 0| 1| 2| 3| 4| 5|               | 0| 6|12|18|24|30|
 *   +--+--+--+--+--+--+               +--+--+--+--+--+--+
 *   | 6| 7| 8| 9|10|11|               | 1| 7|13|19|25|31|
 *   +--+--+--+--+--+--+               +--+--+--+--+--+--+
 *   |12|13|14|15|16|17|    -------->   | 2| 8|14|20|26|32|
 *   +--+--+--+--+--+--+               +--+--+--+--+--+--+
 *   |18|19|20|21|22|23|               | 3| 9|15|21|27|33|
 *   +--+--+--+--+--+--+               +--+--+--+--+--+--+
 *   |24|25|26|27|28|29|               | 4|10|16|22|28|34|
 *   +--+--+--+--+--+--+               +--+--+--+--+--+--+
 *   |30|31|32|33|34|35|               | 5|11|17|23|29|35|
 *   +--+--+--+--+--+--+               +--+--+--+--+--+--+
 *
 * Reading input[y][x]:  threads in a warp have consecutive x values
 *   -> consecutive addresses -> COALESCED READS (good!)
 *
 * Writing output[x][y]: threads in a warp have consecutive x values
 *   -> output[x][y] with consecutive x means stride-N in memory
 *   -> NON-COALESCED WRITES (bad! each thread hits a different cache line)
 */
__global__ void naive_transpose(const float *input, float *output, int n) {
    // Each thread handles one element per iteration,
    // but loops over TILE_DIM/BLOCK_ROWS iterations to cover the full tile
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Loop because we have 32x8 threads covering a 32x32 tile
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n && (y + j) < n) {
            // Read:  input[(y+j) * n + x]     -- coalesced (consecutive x)
            // Write: output[x * n + (y+j)]    -- NON-coalesced (stride n)
            output[x * n + (y + j)] = input[(y + j) * n + x];
        }
    }
}

// ============================================================================
// Kernel 2: TILED transpose using shared memory
// ============================================================================
/*
 * THE BIG IDEA: Use shared memory as a staging area to make BOTH
 * the global read AND the global write coalesced.
 *
 * HOW IT WORKS (for one 32x32 tile):
 *
 * Step 1: Load tile from input into shared memory (coalesced reads)
 * ================================================================
 *
 *   Input Matrix                    Shared Memory tile[32][32]
 *   (global memory)
 *   +----+----+----+----+           +----+----+----+----+
 *   | a0 | a1 | a2 | a3 |   --->   | a0 | a1 | a2 | a3 |   Row 0
 *   +----+----+----+----+   load   +----+----+----+----+
 *   | b0 | b1 | b2 | b3 |   --->   | b0 | b1 | b2 | b3 |   Row 1
 *   +----+----+----+----+          +----+----+----+----+
 *   | c0 | c1 | c2 | c3 |   --->   | c0 | c1 | c2 | c3 |   Row 2
 *   +----+----+----+----+          +----+----+----+----+
 *   | d0 | d1 | d2 | d3 |   --->   | d0 | d1 | d2 | d3 |   Row 3
 *   +----+----+----+----+          +----+----+----+----+
 *
 *   Threads read input[y][x] with consecutive x -> COALESCED
 *   Threads write tile[threadIdx.y][threadIdx.x] -> row-wise, no bank conflict
 *
 *
 * Step 2: __syncthreads()
 * =======================
 *   Wait for ALL threads to finish loading. Without this, some threads
 *   might read from shared memory before other threads have written to it!
 *
 *
 * Step 3: Write from shared memory to output (coalesced writes)
 * =============================================================
 *
 *   Shared Memory tile[32][32]      Output Matrix
 *                                   (global memory, TRANSPOSED position)
 *   +----+----+----+----+
 *   | a0 | a1 | a2 | a3 |          +----+----+----+----+
 *   +----+----+----+----+          | a0 | b0 | c0 | d0 |   Row 0
 *   | b0 | b1 | b2 | b3 |          +----+----+----+----+
 *   +----+----+----+----+          | a1 | b1 | c1 | d1 |   Row 1
 *   | c0 | c1 | c2 | c3 |          +----+----+----+----+
 *   +----+----+----+----+          | a2 | b2 | c2 | d2 |   Row 2
 *   | d0 | d1 | d2 | d3 |          +----+----+----+----+
 *   +----+----+----+----+          | a3 | b3 | c3 | d3 |   Row 3
 *                                   +----+----+----+----+
 *
 *   KEY TRICK: We swap the block indices!
 *   - Read from tile[threadIdx.x][threadIdx.y]  (column-wise from shared mem)
 *   - Write to output at the TRANSPOSED block position (row-wise to global)
 *
 *   Reading shared memory COLUMN-wise: tile[threadIdx.x][col]
 *   -> bank (threadIdx.x * 32 + col) % 32 = threadIdx.x % 32
 *   -> wait, threadIdx.x varies across threads in a warp
 *   -> actually, this DOES cause 32-way bank conflicts!
 *   -> See Kernel 3 for the fix.
 *
 *   Writing to output: consecutive threads write consecutive addresses -> COALESCED
 */
__global__ void tiled_transpose(const float *input, float *output, int n) {
    // Shared memory tile -- holds one 32x32 block of the input
    __shared__ float tile[TILE_DIM][TILE_DIM];

    // Input coordinates: which element to read from global memory
    int x_in = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_in = blockIdx.y * TILE_DIM + threadIdx.y;

    // Output coordinates: SWAP block indices for transposition!
    // The output tile position is the transpose of the input tile position.
    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    // Step 1: Load tile from global memory into shared memory
    // Each thread loads TILE_DIM/BLOCK_ROWS = 4 elements
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x_in < n && (y_in + j) < n) {
            // Read from input: consecutive threadIdx.x -> COALESCED
            // Write to shared memory: tile[row][col] -> row-wise, no bank conflict
            tile[threadIdx.y + j][threadIdx.x] = input[(y_in + j) * n + x_in];
        }
    }

    // Step 2: CRITICAL BARRIER
    // All threads must finish loading before any thread reads from shared memory.
    // Without this, thread A might read tile[r][c] before thread B writes it.
    __syncthreads();

    // Step 3: Write from shared memory to output (transposed)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x_out < n && (y_out + j) < n) {
            // Read from shared memory: tile[threadIdx.x][threadIdx.y + j]
            //   -> COLUMN access! threadIdx.x varies across warp threads,
            //      all accessing column (threadIdx.y + j)
            //   -> bank = (threadIdx.x * 32 + col) % 32 = threadIdx.x
            //   -> Actually each thread hits a DIFFERENT bank here because
            //      threadIdx.x IS the lane index within the warp. BUT WAIT:
            //      tile[tx][ty] where tx = 0..31 and ty is fixed means
            //      addresses stride-32 apart -> all land in the same bank!
            //   -> 32-way bank conflict!  (Fixed in kernel 3)
            //
            // Write to output: consecutive threadIdx.x -> COALESCED (good!)
            output[(y_out + j) * n + x_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================================
// Kernel 3: TILED + PADDED transpose (bank-conflict-free)
// ============================================================================
/*
 * Same as Kernel 2, but with +1 PADDING to eliminate bank conflicts.
 *
 * THE PADDING TRICK:
 *   __shared__ float tile[32][32];     // Column access = 32-way conflict
 *   __shared__ float tile[32][32+1];   // Column access = NO conflict!
 *
 * Why it works:
 *   Without padding: tile[i][col] is at offset (i * 32 + col)
 *     bank = (i * 32 + col) % 32 = col    (for all i!)
 *     All rows, same column -> same bank -> 32-way conflict.
 *
 *   With padding: tile[i][col] is at offset (i * 33 + col)
 *     bank = (i * 33 + col) % 32 = (i + col) % 32
 *     Different rows -> different banks -> NO conflict!
 *
 * Cost: 32 * 4 = 128 extra bytes per tile. Negligible.
 * Benefit: eliminates all bank conflicts on column access.
 */
__global__ void tiled_transpose_padded(const float *input, float *output, int n) {
    // THE FIX: +1 padding eliminates bank conflicts on column access
    //
    //  Without padding (TILE_DIM = 32):
    //    tile[0][0] -> offset 0   -> bank 0
    //    tile[1][0] -> offset 32  -> bank 0   CONFLICT!
    //    tile[2][0] -> offset 64  -> bank 0   CONFLICT!
    //
    //  With padding (TILE_DIM + 1 = 33 stride):
    //    tile[0][0] -> offset 0   -> bank 0
    //    tile[1][0] -> offset 33  -> bank 1   NO conflict!
    //    tile[2][0] -> offset 66  -> bank 2   NO conflict!
    //
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];   // <-- +1 PADDING

    int x_in = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_in = blockIdx.y * TILE_DIM + threadIdx.y;
    int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    // Load tile (same as before -- padding doesn't affect row-wise writes)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x_in < n && (y_in + j) < n) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y_in + j) * n + x_in];
        }
    }

    __syncthreads();

    // Write transposed (column access is now conflict-free thanks to padding!)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x_out < n && (y_out + j) < n) {
            // tile[threadIdx.x][threadIdx.y + j] with padding:
            //   offset = threadIdx.x * 33 + (threadIdx.y + j)
            //   bank = (threadIdx.x * 33 + col) % 32 = (threadIdx.x + col) % 32
            //   Each thread (different threadIdx.x) hits a DIFFERENT bank!
            output[(y_out + j) * n + x_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// ============================================================================
// CPU reference transpose (for verification)
// ============================================================================
void cpu_transpose(const float *input, float *output, int n) {
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            output[x * n + y] = input[y * n + x];
        }
    }
}

// ============================================================================
// Verification
// ============================================================================
int verify(const float *gpu_result, const float *cpu_result, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > 1e-5f) {
            int row = i / n, col = i % n;
            fprintf(stderr, "MISMATCH at [%d][%d]: GPU=%.6f, CPU=%.6f\n",
                    row, col, gpu_result[i], cpu_result[i]);
            return 0;
        }
    }
    return 1;
}

// ============================================================================
// Benchmark utility
// ============================================================================
typedef void (*transpose_fn)(const float *, float *, int);

float benchmark_transpose(transpose_fn kernel, const char *name,
                           const float *d_input, float *d_output,
                           const float *h_cpu_ref, int n) {
    // Grid and block dimensions
    dim3 block(TILE_DIM, BLOCK_ROWS);  // 32 x 8 = 256 threads
    dim3 grid((n + TILE_DIM - 1) / TILE_DIM,
              (n + TILE_DIM - 1) / TILE_DIM);

    // Warmup
    kernel<<<grid, block>>>(d_input, d_output, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify correctness
    float *h_output = (float *)malloc(n * n * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, n * n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    int correct = verify(h_output, h_cpu_ref, n);

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int RUNS = 20;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < RUNS; i++) {
        kernel<<<grid, block>>>(d_input, d_output, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= RUNS;

    // Calculate effective bandwidth
    // Transpose reads and writes N*N floats each = 2 * N*N * 4 bytes
    float bytes = 2.0f * n * n * sizeof(float);
    float gb_s = bytes / (ms * 1e6f);  // GB/s

    printf("  %-35s %8.3f ms  %7.2f GB/s  %s\n",
           name, ms, gb_s, correct ? "PASS" : "FAIL");

    free(h_output);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("Chapter 06: Tiled Matrix Transpose\n");
    printf("===================================\n\n");

    printf("Matrix size: %d x %d (%zu MB)\n", N, N,
           (size_t)N * N * sizeof(float) / (1024 * 1024));
    printf("Tile size:   %d x %d\n", TILE_DIM, TILE_DIM);
    printf("Block size:  %d x %d = %d threads\n\n",
           TILE_DIM, BLOCK_ROWS, TILE_DIM * BLOCK_ROWS);

    /*
     * WHY 32x8 THREAD BLOCKS?
     *
     * We could use 32x32 = 1024 threads, each handling one element.
     * But 32x8 = 256 threads is better because:
     *   1. Each thread does 4 loads/stores -> better instruction-level parallelism
     *   2. Fewer threads per block -> more blocks per SM -> better occupancy
     *   3. 256 threads still fills warps nicely (8 warps per block)
     *   4. Shared memory usage is the same (32x32 tile regardless)
     */

    // Allocate host memory
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_input   = (float *)malloc(bytes);
    float *h_cpu_ref = (float *)malloc(bytes);

    // Initialize input with recognizable values
    for (int i = 0; i < N * N; i++) {
        h_input[i] = (float)(i % 1000) * 0.001f;
    }

    // CPU reference
    cpu_transpose(h_input, h_cpu_ref, N);

    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // ========================================================================
    // Benchmark all three versions
    // ========================================================================
    printf("Benchmarking (average of 20 runs)\n");
    printf("----------------------------------\n");

    float t_naive  = benchmark_transpose(naive_transpose,
                                          "Naive (non-coalesced writes)",
                                          d_input, d_output, h_cpu_ref, N);

    float t_tiled  = benchmark_transpose(tiled_transpose,
                                          "Tiled (shared memory)",
                                          d_input, d_output, h_cpu_ref, N);

    float t_padded = benchmark_transpose(tiled_transpose_padded,
                                          "Tiled + Padded (no bank conflicts)",
                                          d_input, d_output, h_cpu_ref, N);

    // ========================================================================
    // Summary
    // ========================================================================
    printf("\nSpeedups:\n");
    printf("  Tiled vs Naive:           %.2fx\n", t_naive / t_tiled);
    printf("  Tiled+Padded vs Naive:    %.2fx\n", t_naive / t_padded);
    printf("  Tiled+Padded vs Tiled:    %.2fx\n", t_tiled / t_padded);

    printf("\nAnalysis:\n");
    printf("  +---------------------------------------------------------+\n");
    printf("  | Version   | Global Reads | Global Writes | Bank Conflicts |\n");
    printf("  +-----------+--------------+---------------+----------------+\n");
    printf("  | Naive     | Coalesced    | NON-coalesced | N/A            |\n");
    printf("  | Tiled     | Coalesced    | Coalesced     | Yes (32-way)   |\n");
    printf("  | Tiled+Pad | Coalesced    | Coalesced     | None           |\n");
    printf("  +-----------+--------------+---------------+----------------+\n");

    printf("\n  The naive version suffers from non-coalesced writes.\n");
    printf("  Tiling fixes coalescing by using shared memory as a staging area.\n");
    printf("  Padding fixes the remaining bank conflicts in shared memory.\n");

    printf("\n  Profile with:\n");
    printf("    nvprof --metrics gld_efficiency,gst_efficiency ./tiled_transpose\n");
    printf("    nvprof --metrics shared_load_transactions_per_request ./tiled_transpose\n\n");

    // Cleanup
    free(h_input);
    free(h_cpu_ref);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
