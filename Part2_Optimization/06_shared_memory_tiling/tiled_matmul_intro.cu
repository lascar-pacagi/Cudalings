/*
 * Chapter 06: Tiled Matrix Multiplication (Introduction)
 * =======================================================
 *
 * GPU: Quadro P4200 (CC 6.1, 18 SMs, 48 KB shared mem per SM)
 * CUDA: 11.7
 *
 * Matrix multiplication is THE workhorse of deep learning. Every forward pass,
 * every backward pass, every attention head -- it all comes down to matmul.
 * Understanding how to make matmul fast on GPUs is probably the single most
 * important optimization skill in ML systems.
 *
 * This file implements two versions:
 *   1. NAIVE matmul:  each thread computes one element of C by reading an
 *                     entire row of A and column of B from global memory.
 *                     This is bandwidth-bound and wastes most memory traffic.
 *
 *   2. TILED matmul:  we load tiles of A and B into shared memory, compute
 *                     partial dot products, and accumulate across tiles.
 *                     This reuses data massively and approaches compute-bound.
 *
 * THE CORE INSIGHT:
 *   In naive matmul, to compute C[i][j] = sum(A[i][k] * B[k][j] for k=0..N-1),
 *   every thread reads N floats from A and N floats from B. But threads
 *   computing different columns in the same row of C all read the SAME row
 *   of A! And threads computing different rows in the same column of C all
 *   read the SAME column of B! We are reading the same data over and over.
 *
 *   Tiling fixes this: a block of threads cooperatively loads a tile of A
 *   and a tile of B into shared memory, then ALL threads in the block can
 *   reuse that data. Each float loaded from global memory is used TILE_SIZE
 *   times instead of once. This reduces global memory traffic by ~TILE_SIZE x.
 *
 * Compile: nvcc -arch=sm_61 -O2 -lineinfo -o tiled_matmul_intro tiled_matmul_intro.cu
 * Profile: nvprof --metrics gld_efficiency,gst_efficiency,flop_count_sp ./tiled_matmul_intro
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ============================================================================
// Error checking macro
// ============================================================================
// Every CUDA call can fail silently. This macro catches errors immediately.
// Without this, you get mysterious wrong results or segfaults later.
// ALWAYS wrap every CUDA API call with this in real code.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ============================================================================
// Constants
// ============================================================================

// Tile size = block dimension. Each block is TILE_SIZE x TILE_SIZE threads.
// 16x16 = 256 threads per block. We use 16 instead of 32 here because:
//   - 16x16 tile = 16*16*4 = 1024 bytes per tile
//   - We load TWO tiles (one from A, one from B) = 2048 bytes shared mem
//   - With 16x16, we can fit many blocks per SM -> good occupancy
//   - 32x32 would use 32*32*4*2 = 8192 bytes -> fewer blocks per SM
//   - For matmul specifically, 16 is often the sweet spot on older GPUs
//
// You can experiment: change to 32 and observe the effect on performance.
// On modern GPUs with more shared memory, 32 can be better.
#define TILE_SIZE 16

// Matrix dimensions: we compute C = A * B where:
//   A is M x K
//   B is K x N
//   C is M x N
// For simplicity in this intro, we use square matrices.
#define M 1024
#define K 1024
#define N 1024

// ============================================================================
// Kernel 1: NAIVE matrix multiplication
// ============================================================================
/*
 * Each thread computes exactly ONE element of C.
 *
 *                          B (K x N)
 *                     j
 *                     |
 *                     v
 *              +------+------+------+
 *              |      |      |      |
 *              |      | B[k] |      |     thread (i,j) reads the entire
 *              |      | [j]  |      |     column j of B: B[0][j], B[1][j], ...
 *              |      |      |      |
 *              +------+------+------+
 *
 *    A (M x K)
 *  i --> +------+------+------+         C (M x N)
 *        | A[i] | A[i] | A[i] |         +------+------+------+
 *        | [0]  | [1]  | [2]  |   *     |      | C[i] |      |
 *        +------+------+------+    =    |      | [j]  |      |
 *        |      |      |      |         +------+------+------+
 *        +------+------+------+
 *
 *   C[i][j] = sum_{k=0}^{K-1} A[i][k] * B[k][j]
 *
 * THE PROBLEM:
 *   Thread (i, j) reads K floats from row i of A and K floats from column j of B.
 *   But thread (i, j+1) also reads the SAME K floats from row i of A!
 *   In a 16x16 block, row i of A is read 16 times by different threads.
 *   Total global memory reads per block = 16*16 * 2*K = 512*K reads.
 *   That is a LOT of redundant memory traffic.
 *
 *   Arithmetic intensity = (2*K FLOPs) / (2*K * 4 bytes) = 0.25 FLOP/byte
 *   This is extremely low. The GPU can do ~5 TFLOPS but memory bandwidth is
 *   ~200 GB/s, so the ideal arithmetic intensity is 5000/200 = 25 FLOP/byte.
 *   We are 100x below the compute roof. Completely bandwidth-bound.
 */
__global__ void matmul_naive(const float *A, const float *B, float *C,
                              int m, int k, int n) {
    // Which element of C does this thread compute?
    // row = which row of C (and A), col = which column of C (and B)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: the grid might be larger than the matrix if dimensions
    // are not exact multiples of the block size. Without this check, we
    // would read/write out of bounds -> undefined behavior, crashes, or
    // silent corruption. This is the "grid-stride style" guard.
    if (row < m && col < n) {
        float sum = 0.0f;

        // The innermost loop: dot product of row i of A with column j of B.
        // This loop is where all the time is spent.
        // Each iteration does:
        //   1. Load A[row][kk] from global memory (likely L2 cache hit if lucky)
        //   2. Load B[kk][col] from global memory (column access = strided = BAD)
        //   3. One multiply-add (FMA)
        //
        // The ratio is: 2 global loads per 2 FLOPs = 1 load per FLOP
        // Memory is the bottleneck, not compute. The ALUs sit idle waiting.
        for (int kk = 0; kk < k; kk++) {
            sum += A[row * k + kk] * B[kk * n + col];
        }

        C[row * n + col] = sum;
    }
}

// ============================================================================
// Kernel 2: TILED matrix multiplication using shared memory
// ============================================================================
/*
 *
 * ============================================================================
 *                    THE BIG PICTURE: HOW TILING WORKS
 * ============================================================================
 *
 * Instead of each thread independently reading entire rows/columns from
 * global memory, we break the dot product into chunks of size TILE_SIZE.
 * A block of threads cooperatively loads one tile of A and one tile of B
 * into shared memory, computes partial dot products, then moves to the
 * next pair of tiles.
 *
 *
 *       Matrix A (M x K)                    Matrix B (K x N)
 *       +---------+---------+-----+         +---------+---------+-----+
 *       |         |         |     |         |  tile 0 |         |     |
 *       |         |         |     |         |  of B   |         |     |
 *       |  tile 0 |  tile 1 | ... |         +---------+---------+-----+
 *  row->|  of A   |  of A   |     |         |  tile 1 |         |     |
 *       |         |         |     |         |  of B   |         |     |
 *       |         |         |     |         +---------+---------+-----+
 *       +---------+---------+-----+         |   ...   |         |     |
 *       |         |         |     |         +---------+---------+-----+
 *       |         |         |     |                      ^
 *       +---------+---------+-----+                      col
 *
 *
 * For a block computing a TILE_SIZE x TILE_SIZE sub-matrix of C at
 * position (blockRow, blockCol):
 *
 *   C_sub[ty][tx] = sum over t of:
 *       ( tile_t of A's row ) dot ( tile_t of B's column )
 *
 * where t goes from 0 to ceil(K / TILE_SIZE) - 1.
 *
 *
 * ============================================================================
 *            DETAILED WALK-THROUGH OF ONE ITERATION (tile t)
 * ============================================================================
 *
 *  Step 1: COOPERATIVE LOAD -- All threads in the block load one tile each
 *  =====================================================================
 *
 *   From A: load the tile at row=blockRow, horizontal position=t
 *   From B: load the tile at vertical position=t, col=blockCol
 *
 *     Shared Memory A tile           Shared Memory B tile
 *     (TILE_SIZE x TILE_SIZE)        (TILE_SIZE x TILE_SIZE)
 *     +----+----+----+----+          +----+----+----+----+
 *     |a0,0|a0,1|a0,2|a0,3|          |b0,0|b0,1|b0,2|b0,3|
 *     +----+----+----+----+          +----+----+----+----+
 *     |a1,0|a1,1|a1,2|a1,3|          |b1,0|b1,1|b1,2|b1,3|
 *     +----+----+----+----+          +----+----+----+----+
 *     |a2,0|a2,1|a2,2|a2,3|          |b2,0|b2,1|b2,2|b2,3|
 *     +----+----+----+----+          +----+----+----+----+
 *     |a3,0|a3,1|a3,2|a3,3|          |b3,0|b3,1|b3,2|b3,3|
 *     +----+----+----+----+          +----+----+----+----+
 *
 *   Each thread loads exactly ONE element into tile_A and ONE into tile_B.
 *   Thread (ty, tx) loads:
 *     tile_A[ty][tx] = A[ (blockRow*TILE + ty) ][ (t*TILE + tx) ]
 *     tile_B[ty][tx] = B[ (t*TILE + ty)        ][ (blockCol*TILE + tx) ]
 *
 *   Note: the LOAD from A is coalesced because consecutive tx values
 *   (threads in the same warp) read consecutive memory addresses. Same for B.
 *
 *
 *  Step 2: __syncthreads()  <--- CRITICAL BARRIER
 *  =================================================
 *
 *   We MUST wait here. Without this barrier:
 *     - Thread 0 might start computing using tile_B[5][3]
 *     - But thread 42 hasn't written tile_B[5][3] yet!
 *     - Result: race condition, wrong answers, nondeterministic bugs
 *
 *   __syncthreads() guarantees that ALL threads in the block have finished
 *   their loads before ANY thread proceeds to the computation step.
 *
 *
 *  Step 3: COMPUTE -- Each thread accumulates a partial dot product
 *  ================================================================
 *
 *   Thread (ty, tx) computes:
 *
 *     for kk = 0 to TILE_SIZE-1:
 *         sum += tile_A[ty][kk] * tile_B[kk][tx]
 *
 *   Visually for thread (ty=1, tx=2):
 *
 *     tile_A row 1:    [a1,0] [a1,1] [a1,2] [a1,3]
 *                        *      *      *      *
 *     tile_B col 2:    [b0,2] [b1,2] [b2,2] [b3,2]
 *                        =      =      =      =
 *     products:        [p0]   [p1]   [p2]   [p3]
 *                        \      |      |      /
 *                         +-----+------+-----+
 *                                |
 *                          sum += p0+p1+p2+p3
 *
 *   KEY INSIGHT: tile_A[ty][kk] is the SAME value for all threads with
 *   the same ty (same row). So this one load from shared memory serves
 *   TILE_SIZE threads! Similarly, tile_B[kk][tx] is shared by all threads
 *   with the same tx. Data reuse = TILE_SIZE x.
 *
 *   This is why tiling works: each float loaded from global memory is
 *   used TILE_SIZE times from shared memory instead of once.
 *
 *
 *  Step 4: __syncthreads()  <--- SECOND BARRIER
 *  ===============================================
 *
 *   Before we load the NEXT tile, we must ensure all threads are done
 *   reading the CURRENT tile. Otherwise:
 *     - Thread 0 finishes the loop fast and starts loading tile t+1
 *     - Thread 0 overwrites tile_A[0][0] with new data
 *     - Thread 200 is still reading tile_A[0][0] from tile t!
 *     - Result: wrong answer.
 *
 *   This second barrier protects the shared memory from being overwritten
 *   before all threads have finished reading it.
 *
 *
 *  Step 5: REPEAT for the next tile (t+1), accumulating into sum
 *  ==============================================================
 *
 *
 * ============================================================================
 *           FULL PICTURE: ALL TILES FOR ONE OUTPUT SUB-MATRIX
 * ============================================================================
 *
 *  To compute one TILE_SIZE x TILE_SIZE block of C, we slide along K:
 *
 *          A                              B
 *    +---+---+---+---+             +---+---+---+---+
 *    |   | t0| t1| t2|             |   |   |   |   |
 *    +---+---+---+---+             | t0|   |   |   |
 *    |   | t0| t1| t2| <-- row    |   |   |   |   |
 *    +---+---+---+---+             +---+---+---+---+
 *    |   | t0| t1| t2|             | t1|   |   |   |
 *    +---+---+---+---+             |   |   |   |   |
 *    |   |   |   |   |             +---+---+---+---+
 *    +---+---+---+---+             | t2|   |   |   |
 *                                  |   |   |   |   |
 *                                  +---+---+---+---+
 *                                    ^
 *                                   col
 *
 *    C[row_block][col_block] = tile_A_t0 * tile_B_t0    (partial)
 *                            + tile_A_t1 * tile_B_t1    (partial)
 *                            + tile_A_t2 * tile_B_t2    (partial)
 *                            = full result!
 *
 *  Number of global memory loads per block:
 *    Naive:  TILE_SIZE^2 * 2K  (each thread reads 2K values independently)
 *    Tiled:  2 * TILE_SIZE^2 * (K/TILE_SIZE) = 2 * TILE_SIZE * K
 *
 *  Reduction factor = (TILE_SIZE^2 * 2K) / (2 * TILE_SIZE * K) = TILE_SIZE
 *  With TILE_SIZE = 16, we get 16x less global memory traffic!
 *
 *  Arithmetic intensity goes from 0.25 FLOP/byte to 0.25*16 = 4 FLOP/byte.
 *  Still below the roof of ~25 but a MASSIVE improvement.
 *
 * ============================================================================
 */
__global__ void matmul_tiled(const float *A, const float *B, float *C,
                              int m, int k, int n) {
    // Shared memory tiles: each block loads a TILE_SIZE x TILE_SIZE chunk
    // of A and B into these fast on-chip buffers.
    // Cost: 2 * 16 * 16 * 4 = 2048 bytes per block.
    // P4200 has 48 KB shared mem per SM, so we can fit 48*1024/2048 = 24 blocks
    // from shared memory alone (though register pressure and occupancy limits
    // will cap it lower in practice).
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // Which thread am I within this block?
    int tx = threadIdx.x;  // column within the tile
    int ty = threadIdx.y;  // row within the tile

    // Which element of C does this thread ultimately compute?
    // row = global row in C (and A)
    // col = global column in C (and B)
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator for the dot product. This lives in a register -- the fastest
    // memory on the GPU. Each thread maintains its own partial sum.
    float sum = 0.0f;

    // How many tiles do we need to cover the K dimension?
    // If K=1024 and TILE_SIZE=16, we need 64 iterations.
    int num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles along the K dimension
    // Each iteration loads one tile of A and one tile of B, computes
    // partial dot products, and moves on.
    for (int t = 0; t < num_tiles; t++) {

        // ====================================================================
        // STEP 1: Cooperative load from global memory into shared memory
        // ====================================================================
        // Each thread loads exactly one element into tile_A and one into tile_B.
        //
        // For tile_A: we need row "row" of A, columns t*TILE_SIZE .. (t+1)*TILE_SIZE-1
        //   Global column index = t * TILE_SIZE + tx
        //
        // For tile_B: we need column "col" of B, rows t*TILE_SIZE .. (t+1)*TILE_SIZE-1
        //   Global row index = t * TILE_SIZE + ty

        int a_col = t * TILE_SIZE + tx;  // which column of A to read
        int b_row = t * TILE_SIZE + ty;  // which row of B to read

        // Bounds checking: if the matrix dimensions are not multiples of TILE_SIZE,
        // some threads will try to read beyond the matrix edge. We load 0.0 for
        // out-of-bounds elements -- this is correct because 0 * anything = 0,
        // so it doesn't affect the dot product sum.
        if (row < m && a_col < k) {
            tile_A[ty][tx] = A[row * k + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;
        }

        if (b_row < k && col < n) {
            tile_B[ty][tx] = B[b_row * n + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        // ====================================================================
        // STEP 2: Barrier -- wait for all threads to finish loading
        // ====================================================================
        // This is NOT optional. This is NOT a "nice to have." This is
        // REQUIRED for correctness. Without this, thread 0 might start
        // reading tile_B[5][3] before thread 42 has written it.
        // Shared memory is shared by ALL threads in the block, and they
        // execute asynchronously in warps. The barrier synchronizes them.
        __syncthreads();

        // ====================================================================
        // STEP 3: Compute partial dot product using the tiles
        // ====================================================================
        // Each thread multiplies its row of tile_A by its column of tile_B.
        // This is TILE_SIZE multiply-add operations -- all from shared memory!
        //
        // Shared memory latency: ~5 ns (28-32 cycles on Pascal)
        // Global memory latency: ~400 ns (~200-800 cycles)
        // That is ~80x faster per access.
        //
        // Plus, each value in tile_A[ty][kk] is read by TILE_SIZE threads
        // (all threads with the same ty), and tile_B[kk][tx] is read by
        // TILE_SIZE threads (all threads with the same tx).
        // Data reuse = TILE_SIZE = 16x for each loaded value.
        //
        // The compiler will likely unroll this loop since TILE_SIZE is
        // a compile-time constant. Unrolling exposes more ILP (instruction-
        // level parallelism) to the GPU's instruction scheduler.
        #pragma unroll
        for (int kk = 0; kk < TILE_SIZE; kk++) {
            sum += tile_A[ty][kk] * tile_B[kk][tx];
        }

        // ====================================================================
        // STEP 4: Second barrier -- wait before loading the NEXT tile
        // ====================================================================
        // Why do we need ANOTHER barrier? Because we are about to loop back
        // to Step 1 and overwrite tile_A and tile_B with new data. If some
        // thread is still in Step 3 reading the old tile, and another thread
        // has already started Step 1 writing the new tile, we get a race
        // condition. The second __syncthreads() prevents this.
        //
        // Rule of thumb: you need __syncthreads() BOTH after writing to
        // shared memory (before reads) AND before overwriting shared memory
        // (after all reads are done).
        __syncthreads();
    }

    // ========================================================================
    // STEP 5: Write the result to global memory
    // ========================================================================
    // After all tiles have been processed, "sum" contains the complete
    // dot product C[row][col]. Write it out.
    // Bounds check again for non-multiple-of-TILE_SIZE matrices.
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

// ============================================================================
// CPU reference implementation (for correctness checking)
// ============================================================================
// This is the textbook O(M*N*K) triple loop. It is slow but correct.
// We use it to verify our GPU results. In production you would use
// cuBLAS sgemm, but for learning, writing your own is essential.
void cpu_matmul(const float *A, const float *B, float *C,
                int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                sum += A[i * k + kk] * B[kk * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// ============================================================================
// Verification: compare GPU result against CPU reference
// ============================================================================
// Floating point arithmetic is not associative! The GPU computes the sum
// in a different order than the CPU, so results won't be bit-identical.
// We allow a small tolerance. For matmul with K=1024, errors up to ~1e-3
// are normal for float32 due to accumulated rounding.
int verify_result(const float *gpu_result, const float *cpu_result,
                  int m, int n) {
    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < m * n; i++) {
        float err = fabsf(gpu_result[i] - cpu_result[i]);
        // Relative tolerance: allow error proportional to magnitude
        float rel_tol = fmaxf(fabsf(cpu_result[i]) * 1e-4f, 1e-5f);
        if (err > rel_tol) {
            if (errors < 5) {
                int row = i / n, col = i % n;
                fprintf(stderr, "  MISMATCH at [%d][%d]: GPU=%.6f CPU=%.6f diff=%.6f\n",
                        row, col, gpu_result[i], cpu_result[i], err);
            }
            errors++;
        }
        if (err > max_err) max_err = err;
    }
    if (errors > 0) {
        fprintf(stderr, "  Total mismatches: %d / %d (max error: %.6e)\n",
                errors, m * n, max_err);
    }
    return errors == 0;
}

// ============================================================================
// Benchmark harness
// ============================================================================
// We use CUDA events for GPU timing. These are the gold standard for
// measuring GPU kernel execution time because:
//   1. They measure time on the GPU clock, not the CPU clock
//   2. They account for kernel launch overhead
//   3. They don't require cudaDeviceSynchronize() between iterations
//      (just record events and sync at the end)
//
// The function pointer typedef lets us benchmark both kernels with the
// same harness code -- no copy-paste.
typedef void (*matmul_fn)(const float *, const float *, float *, int, int, int);

void benchmark_kernel(matmul_fn kernel, const char *name,
                      const float *d_A, const float *d_B, float *d_C,
                      const float *h_C_ref,
                      int m, int k, int n) {
    // Grid and block dimensions
    // Block: TILE_SIZE x TILE_SIZE threads (16x16 = 256)
    // Grid: enough blocks to cover the entire C matrix
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
              (m + TILE_SIZE - 1) / TILE_SIZE);

    // ---- Warmup ----
    // The first kernel launch has extra overhead: lazy context init,
    // JIT compilation, memory page mapping. Always do a warmup run.
    kernel<<<grid, block>>>(d_A, d_B, d_C, m, k, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Verify correctness ----
    size_t c_bytes = (size_t)m * n * sizeof(float);
    float *h_C_gpu = (float *)malloc(c_bytes);
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, c_bytes, cudaMemcpyDeviceToHost));
    int correct = verify_result(h_C_gpu, h_C_ref, m, n);

    // ---- Timed runs ----
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int RUNS = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < RUNS; i++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, m, k, n);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_total;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    float ms_avg = ms_total / RUNS;

    // ---- Compute performance metrics ----
    //
    // GFLOPS: how many billions of floating point operations per second.
    // For matmul C = A*B with A(MxK), B(KxN), C(MxN):
    //   Each element of C requires K multiplies and K-1 adds ~= 2K FLOPs
    //   Total FLOPs = 2 * M * N * K
    //   (the factor of 2 counts both multiply and add)
    double flops = 2.0 * m * n * k;
    double gflops = (flops / (ms_avg * 1e-3)) / 1e9;

    // Effective bandwidth: how much global memory data we moved.
    // Minimum data movement: read A (M*K), read B (K*N), write C (M*N)
    // This is the theoretical minimum -- naive does much more.
    double bytes_moved = (double)(m * k + k * n + m * n) * sizeof(float);
    double bandwidth_gb_s = (bytes_moved / (ms_avg * 1e-3)) / 1e9;

    // Peak theoretical values for Quadro P4200:
    //   - FP32 compute: ~5.3 TFLOPS
    //   - Memory bandwidth: ~192 GB/s (GDDR5, 256-bit bus)
    //
    // Note: for matmul, the relevant ceiling depends on arithmetic intensity.
    // Naive matmul is bandwidth-bound; tiled matmul moves toward compute-bound.

    printf("  %-40s\n", name);
    printf("    Time:       %8.3f ms\n", ms_avg);
    printf("    GFLOPS:     %8.2f  (peak ~5300)\n", gflops);
    printf("    Bandwidth:  %8.2f GB/s (of ~192 GB/s peak)\n", bandwidth_gb_s);
    printf("    Status:     %s\n\n", correct ? "PASS" : "*** FAIL ***");

    // Cleanup
    free(h_C_gpu);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("=============================================================\n");
    printf("  Chapter 06: Tiled Matrix Multiplication (Introduction)\n");
    printf("=============================================================\n\n");

    // Print GPU info so we know what we are working with
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU:            %s\n", prop.name);
    printf("SMs:            %d\n", prop.multiProcessorCount);
    printf("Shared mem/SM:  %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("Shared mem/blk: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Warp size:      %d\n\n", prop.warpSize);

    printf("Matrix:   C(%d x %d) = A(%d x %d) * B(%d x %d)\n",
           M, N, M, K, K, N);
    printf("Tile:     %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Block:    %d x %d = %d threads\n",
           TILE_SIZE, TILE_SIZE, TILE_SIZE * TILE_SIZE);
    printf("Grid:     %d x %d = %d blocks\n",
           (N + TILE_SIZE - 1) / TILE_SIZE,
           (M + TILE_SIZE - 1) / TILE_SIZE,
           ((N + TILE_SIZE - 1) / TILE_SIZE) * ((M + TILE_SIZE - 1) / TILE_SIZE));

    // Shared memory per block for tiled kernel:
    // 2 tiles * TILE_SIZE * TILE_SIZE * sizeof(float)
    size_t smem_per_block = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    printf("Shared mem/blk: %zu bytes (2 tiles of %dx%d floats)\n",
           smem_per_block, TILE_SIZE, TILE_SIZE);

    // How many blocks can fit per SM based on shared memory alone?
    int max_blocks_smem = (int)(prop.sharedMemPerMultiprocessor / smem_per_block);
    printf("Max blocks/SM (shared mem limit): %d\n\n", max_blocks_smem);

    // ========================================================================
    // Allocate host memory
    // ========================================================================
    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = (size_t)K * N * sizeof(float);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    float *h_A     = (float *)malloc(a_bytes);
    float *h_B     = (float *)malloc(b_bytes);
    float *h_C_ref = (float *)malloc(c_bytes);

    // Initialize matrices with small random values.
    // We keep values small (0 to 1) to reduce floating point error accumulation.
    // With K=1024 terms in the sum, using values in [0,1] keeps the result
    // in [0, K/4] on average, which is well within float32 precision.
    srand(42);  // fixed seed for reproducibility
    for (int i = 0; i < M * K; i++) {
        h_A[i] = (float)(rand() % 1000) * 0.001f;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 1000) * 0.001f;
    }

    // ========================================================================
    // CPU reference computation
    // ========================================================================
    printf("Computing CPU reference (this may take a moment)...\n");
    cpu_matmul(h_A, h_B, h_C_ref, M, K, N);
    printf("CPU reference done.\n\n");

    // ========================================================================
    // Allocate device memory and copy inputs
    // ========================================================================
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice));

    // ========================================================================
    // Benchmark both kernels
    // ========================================================================
    printf("Benchmarking (average of 20 runs each)\n");
    printf("---------------------------------------\n\n");

    benchmark_kernel(matmul_naive, "Naive matmul (global memory only)",
                     d_A, d_B, d_C, h_C_ref, M, K, N);

    benchmark_kernel(matmul_tiled, "Tiled matmul (shared memory, TILE=16)",
                     d_A, d_B, d_C, h_C_ref, M, K, N);

    // ========================================================================
    // Summary and analysis
    // ========================================================================
    printf("=============================================================\n");
    printf("  Analysis\n");
    printf("=============================================================\n\n");

    printf("  +-----------------------------------------------------------+\n");
    printf("  | Metric          | Naive             | Tiled               |\n");
    printf("  +-----------------+-------------------+---------------------+\n");
    printf("  | Global loads    | 2*K per thread    | 2*K/TILE per thread |\n");
    printf("  | Data reuse      | None              | TILE_SIZE = %dx     |\n", TILE_SIZE);
    printf("  | Arith intensity | ~0.25 FLOP/byte   | ~%.1f FLOP/byte     |\n",
           0.25f * TILE_SIZE);
    printf("  | Shared mem      | Not used          | 2 x %dx%d tiles     |\n",
           TILE_SIZE, TILE_SIZE);
    printf("  | Bottleneck      | Memory bandwidth  | Closer to compute   |\n");
    printf("  +-----------------------------------------------------------+\n");

    printf("\n  WHY the tiled version is faster:\n");
    printf("    In naive matmul, each thread reads 2*K = %d floats from global\n", 2 * K);
    printf("    memory (400+ ns latency each). Most of these reads are redundant --\n");
    printf("    neighboring threads read the same row of A or same column of B.\n\n");
    printf("    In tiled matmul, threads cooperatively load small tiles into shared\n");
    printf("    memory (~5 ns latency), then reuse each loaded value %d times.\n", TILE_SIZE);
    printf("    Global memory traffic drops by ~%dx.\n\n", TILE_SIZE);

    printf("  NEXT STEPS (upcoming lessons):\n");
    printf("    1. Increase TILE_SIZE to 32 for more data reuse\n");
    printf("    2. Have each thread compute multiple output elements\n");
    printf("    3. Use register blocking for even more reuse\n");
    printf("    4. Compare against cuBLAS (the speed-of-light reference)\n");
    printf("    5. Add padding to eliminate bank conflicts in tile_A\n\n");

    printf("  Profile with:\n");
    printf("    nvprof --metrics gld_efficiency,flop_count_sp ./tiled_matmul_intro\n");
    printf("    nvprof --metrics shared_load_transactions_per_request ./tiled_matmul_intro\n\n");

    // ========================================================================
    // Cleanup
    // ========================================================================
    free(h_A);
    free(h_B);
    free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
