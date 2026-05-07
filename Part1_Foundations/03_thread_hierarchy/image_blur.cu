/**
 * =============================================================================
 * Chapter 03: Image Box Blur -- Practical 2D Grid Example
 * =============================================================================
 *
 * This program applies a box blur (average filter) to a synthetic grayscale
 * image. We implement two versions:
 *
 *   1. NAIVE:   each thread reads the full neighborhood from global memory
 *   2. SHARED:  load a tile (with halo) into shared memory, then compute
 *
 * The shared memory version demonstrates:
 *   - 2D shared memory tiles with halo regions
 *   - Cooperative loading (threads load the tile together)
 *   - __syncthreads() for synchronization
 *   - Boundary handling (clamping at image edges)
 *
 * Hardware: Quadro P4200 (CC 6.1, 48 KB shared memory per SM)
 *
 * =============================================================================
 *
 * BOX BLUR: for each pixel, replace it with the average of itself and its
 * neighbors in a (2*RADIUS+1) x (2*RADIUS+1) window.
 *
 * Example with RADIUS=1 (3x3 window):
 *
 *   Input:              For pixel P, average the 3x3 neighborhood:
 *   +---+---+---+---+
 *   | a | b | c | d |       +---+---+---+
 *   +---+---+---+---+       | a | b | c |
 *   | e | f | g | h |       +---+---+---+
 *   +---+---+---+---+       | e |[P]| g |   P_out = (a+b+c+e+f+g+i+j+k) / 9
 *   | i | j | k | l |       +---+---+---+
 *   +---+---+---+---+       | i | j | k |
 *   | m | n | o | p |       +---+---+---+
 *   +---+---+---+---+
 *
 * =============================================================================
 *
 * 2D SHARED MEMORY TILE WITH HALO
 * ================================
 *
 * For a block of TILE_W x TILE_H output pixels with blur RADIUS, we need
 * to load (TILE_W + 2*RADIUS) x (TILE_H + 2*RADIUS) input pixels into
 * shared memory.
 *
 * Example: TILE_W=16, TILE_H=16, RADIUS=2
 * Shared memory size: (16+4) x (16+4) = 20 x 20 = 400 floats
 *
 *   <------- TILE_W + 2*RADIUS = 20 ------->
 *   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+  ^
 *   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
 *   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+  |
 *   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | RADIUS=2
 *   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+  v
 *   |  |  |##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|  |  |  ^
 *   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+  |
 *   |  |  |##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|  |  |  |
 *   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+  |
 *   |  |  |##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|  |  |  |
 *   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+  |
 *   |  |  |##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|  |  |  | TILE_H
 *   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+  | = 16
 *   |  |  |##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|  |  |  |
 *   |  ...|..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|..|...|  |  |
 *   |  |  |##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|##|  |  |  |
 *   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+  v
 *   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  ^
 *   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+  | RADIUS=2
 *   |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  v
 *   +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
 *
 *   Legend:
 *     ## = output pixel region (TILE_W x TILE_H) -- threads compute these
 *     (blank) = halo/apron region -- needed for blur neighbors, not output
 *
 *   The halo extends RADIUS pixels beyond the output tile in every direction.
 *   Without the halo, threads at the edge of the tile couldn't read their
 *   full blur neighborhood from shared memory.
 *
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ---------------------------------------------------------------------------
 * Configuration
 * ---------------------------------------------------------------------------*/
#define IMG_WIDTH  1024    /* Image width in pixels */
#define IMG_HEIGHT 1024    /* Image height in pixels */
#define BLUR_RADIUS 3      /* Box blur radius: window is (2*R+1) x (2*R+1) = 7x7 */

/*
 * TILE dimensions: how many OUTPUT pixels each block produces.
 * Block dimensions = TILE dimensions (one thread per output pixel).
 * 16x16 = 256 threads per block -- a good default.
 */
#define TILE_W 16
#define TILE_H 16

/*
 * Shared memory tile includes the halo/apron around the output region.
 * Each thread may need to load more than one pixel into shared memory
 * to cover the halo.
 */
#define SMEM_W (TILE_W + 2 * BLUR_RADIUS)   /* 16 + 6 = 22 */
#define SMEM_H (TILE_H + 2 * BLUR_RADIUS)   /* 16 + 6 = 22 */

/* Number of timing iterations */
#define ITERATIONS 20


/* ---------------------------------------------------------------------------
 * Helper: clamp a coordinate to [0, max-1]
 * ---------------------------------------------------------------------------*/
__device__ __host__ int clamp_coord(int val, int max_val) {
    return val < 0 ? 0 : (val >= max_val ? max_val - 1 : val);
}


/* ---------------------------------------------------------------------------
 * Kernel 1: NAIVE box blur (global memory only)
 * ---------------------------------------------------------------------------
 *
 * Each thread:
 *   1. Computes its output pixel position (row, col)
 *   2. Reads ALL (2R+1)^2 neighbors from GLOBAL memory
 *   3. Averages them and writes the output
 *
 * This is simple but slow because:
 *   - Many redundant global memory reads (neighboring threads read
 *     overlapping neighborhoods)
 *   - Global memory has ~400-600 cycle latency
 * ---------------------------------------------------------------------------*/
__global__ void blur_naive(const float *input, float *output,
                           int width, int height, int radius) {
    /* Compute global pixel coordinates */
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    /*
     * Sum all pixels in the (2R+1) x (2R+1) neighborhood.
     *
     * For RADIUS=3, each thread reads 7*7 = 49 pixels from global memory.
     * With 1024*1024 = 1M pixels, that's 49M global memory reads total!
     *
     * Neighboring threads share most of their reads:
     *   Thread at (row, col) reads rows [row-3, row+3], cols [col-3, col+3]
     *   Thread at (row, col+1) reads rows [row-3, row+3], cols [col-2, col+4]
     *   -> 42 out of 49 reads overlap! (6/7 = 86% redundancy)
     */
    float sum = 0.0f;
    int count = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            /* Clamp to image boundaries (replicate edge pixels) */
            int ny = clamp_coord(row + dy, height);
            int nx = clamp_coord(col + dx, width);
            sum += input[ny * width + nx];
            count++;
        }
    }

    output[row * width + col] = sum / (float)count;
}


/* ---------------------------------------------------------------------------
 * Kernel 2: SHARED MEMORY box blur with 2D halo
 * ---------------------------------------------------------------------------
 *
 * Strategy:
 *   1. Cooperatively load a (TILE_W + 2R) x (TILE_H + 2R) tile from
 *      global memory into shared memory
 *   2. __syncthreads() -- wait for all threads to finish loading
 *   3. Each thread computes its output pixel from shared memory
 *
 * Loading the tile:
 *   The shared memory tile has SMEM_W * SMEM_H elements.
 *   The block has TILE_W * TILE_H threads.
 *   Since SMEM_W * SMEM_H > TILE_W * TILE_H (due to halo), each thread
 *   may need to load MORE than one element.
 *
 *   We use a linear mapping: thread with linear index 'tid' loads elements
 *   tid, tid + blockSize, tid + 2*blockSize, etc.
 *
 * ---------------------------------------------------------------------------*/
__global__ void blur_shared(const float *input, float *output,
                            int width, int height, int radius) {
    /*
     * Shared memory tile: (TILE_W + 2*RADIUS) x (TILE_H + 2*RADIUS)
     *
     * For TILE_W=16, TILE_H=16, RADIUS=3:
     *   SMEM_W = 22, SMEM_H = 22
     *   22 * 22 = 484 floats = 1936 bytes per block
     *   (well within the 48 KB limit per SM)
     */
    __shared__ float tile[SMEM_H][SMEM_W];

    /* ----- Step 1: Cooperatively load the tile from global memory ----- */

    /*
     * Compute the top-left corner of this block's INPUT region
     * (including halo). The halo extends RADIUS pixels before the
     * first output pixel.
     *
     * Block (bx, by) produces output pixels starting at:
     *   out_col_start = bx * TILE_W
     *   out_row_start = by * TILE_H
     *
     * The input region starts RADIUS pixels earlier:
     *   in_col_start = out_col_start - RADIUS
     *   in_row_start = out_row_start - RADIUS
     */
    int tile_col_start = blockIdx.x * TILE_W - radius;
    int tile_row_start = blockIdx.y * TILE_H - radius;

    /*
     * Each thread has a linear index within the block.
     * We use this to assign shared memory loading work.
     */
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;  /* TILE_W * TILE_H = 256 */
    int smem_size = SMEM_W * SMEM_H;           /* 22 * 22 = 484 */

    /*
     * Load loop: each thread loads elements at positions
     *   tid, tid + blockSize, tid + 2*blockSize, ...
     * until all SMEM_W * SMEM_H elements are loaded.
     *
     * For 256 threads loading 484 elements:
     *   First pass:  threads 0-255 load elements 0-255
     *   Second pass: threads 0-227 load elements 256-483
     *   (threads 228-255 are idle in the second pass)
     */
    for (int idx = tid; idx < smem_size; idx += block_size) {
        /* Convert linear shared memory index to 2D coordinates */
        int s_row = idx / SMEM_W;
        int s_col = idx % SMEM_W;

        /* Corresponding global image coordinates */
        int g_row = tile_row_start + s_row;
        int g_col = tile_col_start + s_col;

        /* Clamp to image boundaries (replicate edge pixels) */
        g_row = clamp_coord(g_row, height);
        g_col = clamp_coord(g_col, width);

        tile[s_row][s_col] = input[g_row * width + g_col];
    }

    /*
     * CRITICAL: Wait for ALL threads to finish loading before any
     * thread reads from shared memory. Without this barrier, a thread
     * might read a shared memory location that hasn't been loaded yet!
     */
    __syncthreads();

    /* ----- Step 2: Compute blur from shared memory ----- */

    /*
     * Each thread computes one output pixel.
     * Its position in shared memory is offset by RADIUS from the halo edge.
     */
    int out_col = blockIdx.x * TILE_W + threadIdx.x;
    int out_row = blockIdx.y * TILE_H + threadIdx.y;

    if (out_col >= width || out_row >= height) return;

    /*
     * The thread's center pixel in shared memory is at:
     *   s_col = threadIdx.x + RADIUS
     *   s_row = threadIdx.y + RADIUS
     *
     * We read the neighborhood [s_row-R .. s_row+R][s_col-R .. s_col+R]
     * entirely from shared memory -- no global memory accesses needed!
     *
     * Shared memory has ~5 cycle latency vs ~400+ for global memory.
     */
    int s_center_col = threadIdx.x + radius;
    int s_center_row = threadIdx.y + radius;

    float sum = 0.0f;
    int count = 0;

    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            sum += tile[s_center_row + dy][s_center_col + dx];
            count++;
        }
    }

    output[out_row * width + out_col] = sum / (float)count;
}


/* ---------------------------------------------------------------------------
 * Generate a synthetic test image
 * ---------------------------------------------------------------------------
 *
 * Creates a pattern with:
 *   - Horizontal and vertical gradients
 *   - A bright circle in the center
 *   - Some sharp edges (to show blur effect)
 *
 * Values are in [0, 1] representing grayscale intensity.
 * ---------------------------------------------------------------------------*/
void generate_test_image(float *img, int width, int height) {
    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float max_r = fminf(cx, cy) * 0.4f;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            /* Start with a gradient */
            float val = (float)col / width * 0.3f + (float)row / height * 0.3f;

            /* Add a bright circle */
            float dx = col - cx;
            float dy = row - cy;
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < max_r) {
                val += 0.5f * (1.0f - dist / max_r);
            }

            /* Add a checkerboard pattern for sharp edges */
            if ((col / 64 + row / 64) % 2 == 0) {
                val += 0.1f;
            }

            /* Clamp to [0, 1] */
            img[row * width + col] = fminf(fmaxf(val, 0.0f), 1.0f);
        }
    }
}


/* ---------------------------------------------------------------------------
 * Verify that two images match (within floating-point tolerance)
 * ---------------------------------------------------------------------------*/
int verify_results(const float *a, const float *b, int width, int height,
                   float tolerance) {
    int mismatches = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < width * height; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tolerance) {
            if (mismatches < 5) {
                int row = i / width;
                int col = i % width;
                printf("  Mismatch at (%d,%d): naive=%.6f shared=%.6f diff=%.6f\n",
                       row, col, a[i], b[i], diff);
            }
            mismatches++;
        }
    }

    printf("  Max difference: %.9f\n", max_diff);
    return mismatches;
}


/* ---------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------------*/
int main() {
    printf("==========================================================\n");
    printf("  Chapter 03: Image Box Blur (2D Grid Example)\n");
    printf("==========================================================\n");
    printf("Image size:   %d x %d (%d pixels, %.1f MB)\n",
           IMG_WIDTH, IMG_HEIGHT,
           IMG_WIDTH * IMG_HEIGHT,
           (float)(IMG_WIDTH * IMG_HEIGHT) * sizeof(float) / (1024 * 1024));
    printf("Blur radius:  %d (window size: %dx%d = %d pixels)\n",
           BLUR_RADIUS, 2*BLUR_RADIUS+1, 2*BLUR_RADIUS+1,
           (2*BLUR_RADIUS+1)*(2*BLUR_RADIUS+1));
    printf("Tile size:    %d x %d (output pixels per block)\n", TILE_W, TILE_H);
    printf("Block size:   %d x %d = %d threads\n",
           TILE_W, TILE_H, TILE_W * TILE_H);
    printf("Shared mem:   %d x %d = %d floats = %d bytes per block\n",
           SMEM_W, SMEM_H, SMEM_W * SMEM_H,
           (int)(SMEM_W * SMEM_H * sizeof(float)));

    int pixels = IMG_WIDTH * IMG_HEIGHT;
    size_t img_bytes = pixels * sizeof(float);

    /* -----------------------------------------------------------------------
     * Allocate memory
     * -----------------------------------------------------------------------*/
    float *h_input    = (float *)malloc(img_bytes);
    float *h_naive    = (float *)malloc(img_bytes);
    float *h_shared   = (float *)malloc(img_bytes);

    float *d_input, *d_output_naive, *d_output_shared;
    CUDA_CHECK(cudaMalloc(&d_input,         img_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_naive,  img_bytes));
    CUDA_CHECK(cudaMalloc(&d_output_shared, img_bytes));

    /* -----------------------------------------------------------------------
     * Generate and upload test image
     * -----------------------------------------------------------------------*/
    printf("\nGenerating synthetic test image...\n");
    generate_test_image(h_input, IMG_WIDTH, IMG_HEIGHT);

    /* Print a small sample of the input */
    printf("Sample input values (top-left 5x5 corner):\n");
    for (int r = 0; r < 5; r++) {
        printf("  ");
        for (int c = 0; c < 5; c++) {
            printf("%.3f ", h_input[r * IMG_WIDTH + c]);
        }
        printf("\n");
    }

    CUDA_CHECK(cudaMemcpy(d_input, h_input, img_bytes, cudaMemcpyHostToDevice));

    /* -----------------------------------------------------------------------
     * Launch configuration
     * -----------------------------------------------------------------------
     *
     * Grid dimensions: enough blocks to cover the entire image.
     *   gridDim.x = ceil(IMG_WIDTH  / TILE_W)
     *   gridDim.y = ceil(IMG_HEIGHT / TILE_H)
     *
     * For 1024x1024 with 16x16 tiles: gridDim = (64, 64) = 4096 blocks
     * 4096 blocks / 18 SMs = ~227 blocks per SM (wave after wave)
     * -----------------------------------------------------------------------*/
    dim3 blockDim_naive(TILE_W, TILE_H);
    dim3 gridDim_naive(
        (IMG_WIDTH  + TILE_W - 1) / TILE_W,
        (IMG_HEIGHT + TILE_H - 1) / TILE_H
    );

    dim3 blockDim_shared(TILE_W, TILE_H);
    dim3 gridDim_shared(
        (IMG_WIDTH  + TILE_W - 1) / TILE_W,
        (IMG_HEIGHT + TILE_H - 1) / TILE_H
    );

    printf("\nGrid:  (%d, %d) = %d blocks\n",
           gridDim_naive.x, gridDim_naive.y,
           gridDim_naive.x * gridDim_naive.y);
    printf("Block: (%d, %d) = %d threads\n",
           blockDim_naive.x, blockDim_naive.y,
           blockDim_naive.x * blockDim_naive.y);

    /* -----------------------------------------------------------------------
     * Warm-up runs
     * -----------------------------------------------------------------------*/
    printf("\nWarm-up...\n");
    blur_naive<<<gridDim_naive, blockDim_naive>>>(d_input, d_output_naive,
                                                  IMG_WIDTH, IMG_HEIGHT,
                                                  BLUR_RADIUS);
    blur_shared<<<gridDim_shared, blockDim_shared>>>(d_input, d_output_shared,
                                                      IMG_WIDTH, IMG_HEIGHT,
                                                      BLUR_RADIUS);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* -----------------------------------------------------------------------
     * Timed runs
     * -----------------------------------------------------------------------*/
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* --- Time naive kernel --- */
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        blur_naive<<<gridDim_naive, blockDim_naive>>>(d_input, d_output_naive,
                                                      IMG_WIDTH, IMG_HEIGHT,
                                                      BLUR_RADIUS);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    ms_naive /= ITERATIONS;

    /* --- Time shared memory kernel --- */
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERATIONS; i++) {
        blur_shared<<<gridDim_shared, blockDim_shared>>>(d_input, d_output_shared,
                                                          IMG_WIDTH, IMG_HEIGHT,
                                                          BLUR_RADIUS);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_shared = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_shared, start, stop));
    ms_shared /= ITERATIONS;

    /* -----------------------------------------------------------------------
     * Results
     * -----------------------------------------------------------------------*/
    printf("\n----------------------------------------------------------\n");
    printf("Timing Results (average over %d iterations):\n", ITERATIONS);
    printf("----------------------------------------------------------\n");
    printf("  Naive  (global memory): %8.3f ms\n", ms_naive);
    printf("  Shared (shared memory): %8.3f ms\n", ms_shared);
    printf("  Speedup:                %8.2fx\n", ms_naive / ms_shared);

    /* -----------------------------------------------------------------------
     * Verify correctness: both kernels should produce the same output
     * -----------------------------------------------------------------------*/
    printf("\n----------------------------------------------------------\n");
    printf("Verification (naive vs shared):\n");
    printf("----------------------------------------------------------\n");

    CUDA_CHECK(cudaMemcpy(h_naive,  d_output_naive,  img_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_shared, d_output_shared, img_bytes,
                          cudaMemcpyDeviceToHost));

    int mismatches = verify_results(h_naive, h_shared, IMG_WIDTH, IMG_HEIGHT,
                                    1e-5f);
    if (mismatches == 0) {
        printf("  PASSED: Both kernels produce identical results.\n");
    } else {
        printf("  WARNING: %d mismatches found (may be floating-point order).\n",
               mismatches);
    }

    /* Print a sample of the blurred output */
    printf("\nSample output values (top-left 5x5 corner, shared memory version):\n");
    for (int r = 0; r < 5; r++) {
        printf("  ");
        for (int c = 0; c < 5; c++) {
            printf("%.3f ", h_shared[r * IMG_WIDTH + c]);
        }
        printf("\n");
    }

    /* -----------------------------------------------------------------------
     * Analysis
     * -----------------------------------------------------------------------*/
    printf("\n----------------------------------------------------------\n");
    printf("Analysis:\n");
    printf("----------------------------------------------------------\n");
    printf("\nNAIVE version:\n");
    printf("  Each thread reads %d pixels from global memory.\n",
           (2*BLUR_RADIUS+1) * (2*BLUR_RADIUS+1));
    printf("  Neighboring threads share most of their reads,\n");
    printf("  but each thread independently fetches from global memory.\n");
    printf("  Total global reads: %d x %d = %d M reads\n",
           pixels, (2*BLUR_RADIUS+1)*(2*BLUR_RADIUS+1),
           pixels * (2*BLUR_RADIUS+1)*(2*BLUR_RADIUS+1) / 1000000);

    printf("\nSHARED MEMORY version:\n");
    printf("  Each block loads a %dx%d tile (%d elements) once.\n",
           SMEM_W, SMEM_H, SMEM_W * SMEM_H);
    printf("  Then all %d threads read from fast shared memory.\n",
           TILE_W * TILE_H);
    printf("  Global reads per block: %d (vs %d for naive)\n",
           SMEM_W * SMEM_H,
           TILE_W * TILE_H * (2*BLUR_RADIUS+1)*(2*BLUR_RADIUS+1));
    printf("  Reduction: %.1fx fewer global memory reads!\n",
           (float)(TILE_W * TILE_H * (2*BLUR_RADIUS+1)*(2*BLUR_RADIUS+1)) /
           (SMEM_W * SMEM_H));

    printf("\nShared memory bandwidth: ~1.5 TB/s (per SM)\n");
    printf("Global memory bandwidth: ~76 GB/s (Quadro P4200)\n");
    printf("Shared memory is ~20x faster per access!\n");

    /* Cleanup */
    free(h_input);
    free(h_naive);
    free(h_shared);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_naive));
    CUDA_CHECK(cudaFree(d_output_shared));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\n==========================================================\n");
    printf("  Done!\n");
    printf("==========================================================\n");

    return 0;
}
