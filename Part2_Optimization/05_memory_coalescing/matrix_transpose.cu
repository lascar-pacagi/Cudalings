/*
 * matrix_transpose.cu -- The Classic Memory Coalescing Case Study
 * ================================================================
 *
 * Matrix transpose is the textbook example of the coalescing problem.
 * No matter how you do a naive transpose, either the reads OR the writes
 * must be non-coalesced. You can't have both coalesced simultaneously
 * with a naive approach.
 *
 * WHY TRANSPOSE IS HARD FOR GPUs:
 *
 *   Input matrix (row-major, 4x4 example):
 *   ┌─────────────────────┐
 *   │  0   1   2   3      │  Row 0: addresses 0, 4, 8, 12
 *   │  4   5   6   7      │  Row 1: addresses 16, 20, 24, 28
 *   │  8   9  10  11      │  Row 2: addresses 32, 36, 40, 44
 *   │ 12  13  14  15      │  Row 3: addresses 48, 52, 56, 60
 *   └─────────────────────┘
 *
 *   Output (transposed):
 *   ┌─────────────────────┐
 *   │  0   4   8  12      │  Column 0 of input -> Row 0 of output
 *   │  1   5   9  13      │  Column 1 of input -> Row 1 of output
 *   │  2   6  10  14      │
 *   │  3   7  11  15      │
 *   └─────────────────────┘
 *
 *   out[j * N + i] = in[i * N + j]
 *
 *   For a warp reading row 0 of input (coalesced read):
 *     T0 reads in[0][0], T1 reads in[0][1], T2 reads in[0][2], ...
 *     Addresses: 0, 4, 8, 12 -> consecutive -> COALESCED READ
 *
 *   But writing the transposed output (column of output):
 *     T0 writes out[0][0], T1 writes out[1][0], T2 writes out[2][0], ...
 *     Addresses: 0, N*4, 2*N*4, 3*N*4 -> stride N -> NON-COALESCED WRITE!
 *
 *   Alternatively, you can do coalesced writes by reading columns:
 *     Read column:  in[0][j], in[1][j], in[2][j], ... -> stride N -> non-coalesced
 *     Write row:    out[j][0], out[j][1], out[j][2], ... -> coalesced
 *
 *   Either way, you have ONE coalesced and ONE non-coalesced access.
 *   The solution (covered in Chapter 06) uses shared memory as an
 *   intermediate buffer to "re-organize" the data within a tile.
 *
 * We implement two naive approaches and benchmark them:
 *   1. Read rows (coalesced), write columns (strided)
 *   2. Read columns (strided), write rows (coalesced)
 *
 * Expected: Both are slow (~30-50 GB/s), well below peak.
 * With shared memory tiling (Chapter 06): ~150+ GB/s.
 *
 * Compile: nvcc -arch=sm_61 -O2 -lineinfo -o matrix_transpose matrix_transpose.cu
 * Run:     ./matrix_transpose
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
 * Matrix dimensions: N x N.
 * We use a large power-of-2 size so:
 *   1. It doesn't fit in L2 cache (N=4096 -> 4096*4096*4 = 64 MB)
 *   2. It's divisible by common tile sizes (16, 32)
 */
#define N 4096

#define BLOCK_DIM_X 32    // Tile width  (= warp size for best coalescing)
#define BLOCK_DIM_Y 8     // Tile height (8 rows per block, each thread does 4 rows)

#define WARMUP_ITERS 5
#define BENCH_ITERS  20

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
// KERNEL 1: COPY (BASELINE -- No transpose, both reads and writes coalesced)
// ============================================================================
/*
 * Simple copy: out[i][j] = in[i][j].
 * Both reads and writes are coalesced because consecutive threads
 * access consecutive columns in the same row.
 *
 * Access pattern for a warp (threads map to columns):
 *   T0 -> in[row][0],  T1 -> in[row][1],  ... T31 -> in[row][31]
 *         out[row][0],       out[row][1],       ... out[row][31]
 *   All consecutive -> COALESCED reads AND writes.
 *
 * This gives us an upper bound on achievable bandwidth.
 */
__global__ void kernel_copy(const float *in, float *out, int n) {
    int col = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    int row = blockIdx.y * BLOCK_DIM_X + threadIdx.y;  // stride = tile height (BLOCK_DIM_X)

    /*
     * Each thread block covers a BLOCK_DIM_X x BLOCK_DIM_Y tile.
     * But we set BLOCK_DIM_Y = 8, while the tile height is 32.
     * So each thread processes 4 rows (32 / 8 = 4 iterations).
     * This reduces the number of threads per block while still
     * covering the full tile.
     */
    if (col < n) {
        for (int j = 0; j < BLOCK_DIM_X; j += BLOCK_DIM_Y) {
            int r = row + j;
            if (r < n) {
                out[r * n + col] = in[r * n + col];
            }
        }
    }
}


// ============================================================================
// KERNEL 2: NAIVE TRANSPOSE -- Read Rows, Write Columns
// ============================================================================
/*
 * out[j][i] = in[i][j]
 *
 * For a warp with threads mapping to columns (j):
 *
 *   READ:  in[i][j]  where j = T0..T31 (consecutive j) -> COALESCED
 *   WRITE: out[j][i] where j = T0..T31 (each thread writes to a different row)
 *          -> addresses: out[0*n+i], out[1*n+i], out[2*n+i], ...
 *          -> stride of n elements (n * 4 bytes) between adjacent threads
 *          -> NON-COALESCED (stride = n)
 *
 *   VISUAL (simplified 4x4):
 *
 *   Input (row-major):              Output (row-major):
 *   ┌──────────────────┐            ┌──────────────────┐
 *   │ a  b  c  d       │            │ a  e  i  m       │
 *   │ e  f  g  h       │   ---->    │ b  f  j  n       │
 *   │ i  j  k  l       │            │ c  g  k  o       │
 *   │ m  n  o  p       │            │ d  h  l  p       │
 *   └──────────────────┘            └──────────────────┘
 *
 *   Warp reads row 0: [a, b, c, d] -- addresses 0,4,8,12 -> COALESCED
 *   Warp writes col 0 of output:
 *     T0 writes out[0][0]=a  (addr 0)
 *     T1 writes out[1][0]=b  (addr 16)    <- stride of N=4
 *     T2 writes out[2][0]=c  (addr 32)
 *     T3 writes out[3][0]=d  (addr 48)
 *   -> NON-COALESCED (stride = N)
 */
__global__ void kernel_transpose_naive_read_coalesced(const float *in,
                                                       float *out, int n) {
    int col = blockIdx.x * BLOCK_DIM_X + threadIdx.x;   // input column (j)
    int row = blockIdx.y * BLOCK_DIM_X + threadIdx.y;   // input row (i), stride = tile height

    if (col < n) {
        for (int j = 0; j < BLOCK_DIM_X; j += BLOCK_DIM_Y) {
            int r = row + j;
            if (r < n) {
                // Read: in[r * n + col] -> consecutive col across threads -> COALESCED
                // Write: out[col * n + r] -> col varies across threads, r is same
                //        -> stride n between adjacent threads -> NON-COALESCED
                out[col * n + r] = in[r * n + col];
            }
        }
    }
}


// ============================================================================
// KERNEL 3: NAIVE TRANSPOSE -- Read Columns, Write Rows
// ============================================================================
/*
 * Same transpose but threads are organized differently:
 * Each thread reads from a column and writes to a row.
 *
 *   READ:  in[j][i]  where i = T0..T31 (each thread reads a different row)
 *          -> stride n between adjacent threads -> NON-COALESCED
 *   WRITE: out[i][j] where i = T0..T31 (consecutive i) -> COALESCED
 *
 * This is the "mirror" of kernel 2: now writes are coalesced but reads aren't.
 * Performance should be similar to kernel 2.
 */
__global__ void kernel_transpose_naive_write_coalesced(const float *in,
                                                        float *out, int n) {
    // Note: we swap the meaning of threadIdx.x mapping
    int col = blockIdx.x * BLOCK_DIM_X + threadIdx.x;   // output column (i)
    int row = blockIdx.y * BLOCK_DIM_X + threadIdx.y;   // output row (j), stride = tile height

    if (col < n) {
        for (int j = 0; j < BLOCK_DIM_X; j += BLOCK_DIM_Y) {
            int r = row + j;
            if (r < n) {
                // Read: in[col * n + r] -> col varies across threads, r is same
                //       -> stride n between adjacent threads -> NON-COALESCED
                // Write: out[r * n + col] -> consecutive col across threads -> COALESCED
                out[r * n + col] = in[col * n + r];
            }
        }
    }
}


// ============================================================================
// CORRECTNESS VERIFICATION
// ============================================================================

/*
 * CPU reference transpose for verification.
 */
void transpose_cpu(const float *in, float *out, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            out[j * n + i] = in[i * n + j];
        }
    }
}

int verify(const float *gpu_result, const float *cpu_result, int n) {
    for (int i = 0; i < n * n; i++) {
        if (fabsf(gpu_result[i] - cpu_result[i]) > 1e-5f) {
            printf("  MISMATCH at index %d: GPU=%.6f, CPU=%.6f\n",
                   i, gpu_result[i], cpu_result[i]);
            return 0;
        }
    }
    return 1;
}


// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("Matrix Transpose: A Coalescing Case Study\n");
    printf("==========================================\n");
    printf("Matrix size: %d x %d (%.1f MB)\n", N, N,
           (float)N * N * sizeof(float) / (1024 * 1024));
    printf("Block: %d x %d threads\n\n", BLOCK_DIM_X, BLOCK_DIM_Y);

    size_t bytes = N * N * sizeof(float);

    // -----------------------------------------------------------------------
    // Allocate host memory
    // -----------------------------------------------------------------------
    float *h_in       = (float *)malloc(bytes);
    float *h_out      = (float *)malloc(bytes);
    float *h_ref      = (float *)malloc(bytes);

    // Initialize input matrix
    for (int i = 0; i < N * N; i++) {
        h_in[i] = (float)i;
    }

    // CPU reference
    transpose_cpu(h_in, h_ref, N);

    // -----------------------------------------------------------------------
    // Allocate device memory
    // -----------------------------------------------------------------------
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------------
    // Launch configuration
    // -----------------------------------------------------------------------
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((N + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (N + BLOCK_DIM_X - 1) / BLOCK_DIM_X);  // Y stride = full tile height

    printf("Grid: %d x %d blocks, Block: %d x %d threads\n\n",
           grid.x, grid.y, block.x, block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float elapsed_ms;

    // ===================================================================
    // BENCHMARK 1: COPY (Upper bound on bandwidth)
    // ===================================================================
    printf("--- Kernel: Copy (baseline, no transpose) ---\n");

    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_copy<<<grid, block>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        kernel_copy<<<grid, block>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float copy_time = elapsed_ms / BENCH_ITERS;
    // Bytes moved: N*N reads + N*N writes = 2 * N*N * 4 bytes
    float copy_bw = (2.0f * N * N * sizeof(float) * BENCH_ITERS) /
                     (elapsed_ms / 1000.0f) / 1e9;

    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s (both r/w coalesced)\n",
           copy_time, copy_bw);

    // ===================================================================
    // BENCHMARK 2: Naive transpose -- read coalesced, write strided
    // ===================================================================
    printf("\n--- Kernel: Naive transpose (read coalesced, write strided) ---\n");
    /*
     *   ACCESS PATTERN:
     *   ┌──────────────────────────────────────────────────────────────┐
     *   │ READ:  threads read consecutive columns -> COALESCED        │
     *   │        T0->col0, T1->col1, T2->col2, ... all same row      │
     *   │        Addresses: base + 0, base + 4, base + 8, ...        │
     *   │                                                             │
     *   │ WRITE: threads write to consecutive ROWS -> STRIDED         │
     *   │        T0->row0, T1->row1, T2->row2, ... all same column   │
     *   │        Addresses: base + 0, base + N*4, base + 2*N*4, ...  │
     *   │        Stride = N elements = N*4 bytes between threads      │
     *   └──────────────────────────────────────────────────────────────┘
     */

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_transpose_naive_read_coalesced<<<grid, block>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        kernel_transpose_naive_read_coalesced<<<grid, block>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float naive_rc_time = elapsed_ms / BENCH_ITERS;
    float naive_rc_bw = (2.0f * N * N * sizeof(float) * BENCH_ITERS) /
                         (elapsed_ms / 1000.0f) / 1e9;

    // Verify correctness
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    int rc_correct = verify(h_out, h_ref, N);

    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s %s\n",
           naive_rc_time, naive_rc_bw, rc_correct ? "(CORRECT)" : "(WRONG)");
    printf("  Slowdown vs copy: %.2fx\n", naive_rc_time / copy_time);

    // ===================================================================
    // BENCHMARK 3: Naive transpose -- read strided, write coalesced
    // ===================================================================
    printf("\n--- Kernel: Naive transpose (read strided, write coalesced) ---\n");
    /*
     *   ACCESS PATTERN:
     *   ┌──────────────────────────────────────────────────────────────┐
     *   │ READ:  threads read consecutive ROWS -> STRIDED             │
     *   │        Stride = N elements between adjacent threads         │
     *   │                                                             │
     *   │ WRITE: threads write consecutive columns -> COALESCED       │
     *   │        Addresses: base + 0, base + 4, base + 8, ...        │
     *   └──────────────────────────────────────────────────────────────┘
     */

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_transpose_naive_write_coalesced<<<grid, block>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        kernel_transpose_naive_write_coalesced<<<grid, block>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float naive_wc_time = elapsed_ms / BENCH_ITERS;
    float naive_wc_bw = (2.0f * N * N * sizeof(float) * BENCH_ITERS) /
                         (elapsed_ms / 1000.0f) / 1e9;

    // Verify correctness
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    int wc_correct = verify(h_out, h_ref, N);

    printf("  Time: %.3f ms, Bandwidth: %.1f GB/s %s\n",
           naive_wc_time, naive_wc_bw, wc_correct ? "(CORRECT)" : "(WRONG)");
    printf("  Slowdown vs copy: %.2fx\n", naive_wc_time / copy_time);

    // ===================================================================
    // SUMMARY
    // ===================================================================
    printf("\n");
    printf("==========================================================\n");
    printf("  MATRIX TRANSPOSE SUMMARY (%d x %d)\n", N, N);
    printf("==========================================================\n");
    printf("  Kernel                         Time(ms)  BW(GB/s)  Ratio\n");
    printf("  ----------------------------  --------  --------  -----\n");
    printf("  Copy (both coalesced)          %7.3f   %6.1f    1.00x\n",
           copy_time, copy_bw);
    printf("  Transpose (read coal.)         %7.3f   %6.1f    %.2fx\n",
           naive_rc_time, naive_rc_bw, copy_time / naive_rc_time);
    printf("  Transpose (write coal.)        %7.3f   %6.1f    %.2fx\n",
           naive_wc_time, naive_wc_bw, copy_time / naive_wc_time);
    printf("  Transpose (shared mem tile)    ??????    ~150+    ~1.0x\n");
    printf("==========================================================\n\n");

    printf("THE COALESCING DILEMMA:\n");
    printf("  A naive transpose CANNOT coalesce both reads and writes.\n");
    printf("  If you read row-by-row (coalesced), you write column-by-column (strided).\n");
    printf("  If you read column-by-column (strided), you write row-by-row (coalesced).\n\n");

    printf("THE SOLUTION (Chapter 06 -- Shared Memory Tiling):\n");
    printf("  1. Read a tile from input into shared memory (coalesced global read)\n");
    printf("  2. __syncthreads()\n");
    printf("  3. Read the transposed data from shared memory (no bank conflicts)\n");
    printf("  4. Write to output (coalesced global write)\n\n");
    printf("  Shared memory acts as a 'scratchpad' that lets us reorganize data\n");
    printf("  so BOTH the global read and global write can be coalesced.\n");
    printf("  This typically achieves ~90-95%% of copy bandwidth.\n\n");

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    free(h_ref);

    return 0;
}
