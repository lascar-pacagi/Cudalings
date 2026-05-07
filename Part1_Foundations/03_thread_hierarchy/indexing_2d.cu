/**
 * =============================================================================
 * Chapter 03: 2D Thread Indexing Demo
 * =============================================================================
 *
 * This program demonstrates how CUDA's 2D grid/block structure maps threads
 * to positions in a 2D array (like an image or matrix).
 *
 * Key concepts:
 *   - dim3 for 2D grid and block dimensions
 *   - Computing global (row, col) from block/thread indices
 *   - Row-major vs column-major memory layout
 *   - Boundary checking for grids that don't divide evenly
 *
 * Hardware: Quadro P4200 (CC 6.1, max 1024 threads/block)
 *
 * =============================================================================
 *
 * 2D INDEXING DIAGRAM
 * ===================
 *
 * Suppose we have a 10x8 matrix and launch with blockDim(4,4), gridDim(2,3):
 *
 *                    col -->
 *          0   1   2   3   4   5   6   7
 *       +---+---+---+---+---+---+---+---+
 *    0  |   |   |   |   |   |   |   |   |  Block(0,0)    Block(1,0)
 *    1  |   |   |   |   |   |   |   |   |  covers        covers
 *    2  |   |   |   |   |   |   |   |   |  cols 0-3,     cols 4-7,
 *    3  |   |   |   |   |   |   |   |   |  rows 0-3      rows 0-3
 *       +---+---+---+---+---+---+---+---+
 *    4  |   |   |   |   |   |   |   |   |
 * row   5  |   |   |   |   |   |   |   |   |
 *  |    6  |   |   |   |   |   |   |   |   |
 *  v    7  |   |   |   |   |   |   |   |   |
 *       +---+---+---+---+---+---+---+---+
 *    8  |   |   |   |   |   |   |   |   |  Block(0,2) covers rows 8-9
 *    9  |   |   |   |   |   |   |   |   |  (threads with row >= 10 idle)
 *       +---+---+---+---+---+---+---+---+
 *
 * ROW-MAJOR LAYOUT (C/C++ default):
 *   Elements in the same row are contiguous in memory.
 *   index = row * width + col
 *
 *   Memory: [row0,col0] [row0,col1] [row0,col2] ... [row1,col0] [row1,col1] ...
 *
 * COLUMN-MAJOR LAYOUT (Fortran, MATLAB):
 *   Elements in the same column are contiguous in memory.
 *   index = col * height + row
 *
 *   Memory: [row0,col0] [row1,col0] [row2,col0] ... [row0,col1] [row1,col1] ...
 *
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* ---------------------------------------------------------------------------
 * Error checking macro (from Chapter 04, but useful everywhere)
 * ---------------------------------------------------------------------------*/
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
 * Kernel 1: Fill a 2D matrix using ROW-MAJOR layout
 * ---------------------------------------------------------------------------
 *
 * Each thread computes its global (row, col) and writes a unique value.
 * The value encodes the position: row * 1000 + col, so we can verify
 * that the indexing is correct.
 *
 * Global coordinates:
 *   col = blockIdx.x * blockDim.x + threadIdx.x
 *   row = blockIdx.y * blockDim.y + threadIdx.y
 *
 * Row-major linear index:
 *   idx = row * width + col
 * ---------------------------------------------------------------------------*/
__global__ void fill_row_major(int *matrix, int width, int height) {
    /* Compute global column and row for this thread */
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    /* Boundary check: grid may be larger than the matrix */
    if (row < height && col < width) {
        /*
         * Row-major index:
         *   Row 0 occupies indices [0, width-1]
         *   Row 1 occupies indices [width, 2*width-1]
         *   etc.
         */
        int idx = row * width + col;

        /* Encode position as value for verification */
        matrix[idx] = row * 1000 + col;
    }
}


/* ---------------------------------------------------------------------------
 * Kernel 2: Fill a 2D matrix using COLUMN-MAJOR layout
 * ---------------------------------------------------------------------------
 *
 * Same thread mapping, but different memory layout.
 *
 * Column-major linear index:
 *   idx = col * height + row
 *
 * NOTE: Column-major with 2D blocks where threadIdx.x maps to columns
 * means threads in the same warp access non-contiguous memory (bad for
 * coalescing). We'll discuss memory coalescing in a later chapter.
 * ---------------------------------------------------------------------------*/
__global__ void fill_col_major(int *matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        /*
         * Column-major index:
         *   Col 0 occupies indices [0, height-1]
         *   Col 1 occupies indices [height, 2*height-1]
         *   etc.
         */
        int idx = col * height + row;

        matrix[idx] = row * 1000 + col;
    }
}


/* ---------------------------------------------------------------------------
 * Kernel 3: Print thread assignment info (small grid only!)
 * ---------------------------------------------------------------------------
 *
 * Each thread prints its block index, thread index, global position,
 * and which warp it belongs to. Only use on small grids!
 * ---------------------------------------------------------------------------*/
__global__ void print_thread_info(int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        /*
         * Compute the linear thread index within the block.
         * For a 2D block: linear = threadIdx.y * blockDim.x + threadIdx.x
         * Warp ID = linear / 32
         * Lane ID = linear % 32
         */
        int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
        int warp_id = linear_tid / 32;
        int lane_id = linear_tid % 32;

        printf("Block(%d,%d) Thread(%d,%d) -> Global(%d,%d) "
               "| LinearTid=%2d Warp=%d Lane=%2d | RowMajorIdx=%d\n",
               blockIdx.x, blockIdx.y,
               threadIdx.x, threadIdx.y,
               col, row,
               linear_tid, warp_id, lane_id,
               row * width + col);
    }
}


/* ---------------------------------------------------------------------------
 * Helper: Print a matrix stored in row-major order
 * ---------------------------------------------------------------------------*/
void print_matrix_row_major(const int *matrix, int width, int height,
                            const char *label) {
    printf("\n%s (row-major layout):\n", label);
    printf("       ");
    for (int c = 0; c < width; c++) {
        printf("col%-5d", c);
    }
    printf("\n");

    for (int r = 0; r < height; r++) {
        printf("row %d: ", r);
        for (int c = 0; c < width; c++) {
            printf("%-8d", matrix[r * width + c]);
        }
        printf("\n");
    }
}


/* ---------------------------------------------------------------------------
 * Helper: Print a matrix stored in column-major order
 * ---------------------------------------------------------------------------
 * The data is stored column-by-column, but we display it as a 2D grid.
 * ---------------------------------------------------------------------------*/
void print_matrix_col_major(const int *matrix, int width, int height,
                            const char *label) {
    printf("\n%s (column-major layout):\n", label);
    printf("       ");
    for (int c = 0; c < width; c++) {
        printf("col%-5d", c);
    }
    printf("\n");

    for (int r = 0; r < height; r++) {
        printf("row %d: ", r);
        for (int c = 0; c < width; c++) {
            /* Column-major: element (r,c) is at index c * height + r */
            printf("%-8d", matrix[c * height + r]);
        }
        printf("\n");
    }
}


/* ---------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------------*/
int main() {
    printf("==========================================================\n");
    printf("  Chapter 03: 2D Thread Indexing Demo\n");
    printf("==========================================================\n");

    /* -----------------------------------------------------------------------
     * Part 1: Small matrix to visualize thread assignments
     * -----------------------------------------------------------------------
     * We use a tiny 6x4 matrix (width=6, height=4) with 4x2 thread blocks.
     * This gives us a 2x2 grid of blocks.
     *
     *   gridDim  = (ceil(6/4), ceil(4/2)) = (2, 2)
     *   blockDim = (4, 2)
     *
     *         col 0  col 1  col 2  col 3  col 4  col 5
     *        +------+------+------+------+------+------+
     * row 0  |      Block(0,0)     |      Block(1,0)     |
     * row 1  |                     |                     |
     *        +------+------+------+------+------+------+
     * row 2  |      Block(0,1)     |      Block(1,1)     |
     * row 3  |                     |                     |
     *        +------+------+------+------+------+------+
     *
     * Block(1,0) covers cols 4-7, but cols 6-7 are out of bounds (width=6),
     * so those threads will be masked by the boundary check.
     * -----------------------------------------------------------------------*/
    {
        const int WIDTH  = 6;
        const int HEIGHT = 4;

        printf("\n----------------------------------------------------------\n");
        printf("Part 1: Thread Assignment Visualization (%dx%d matrix)\n",
               WIDTH, HEIGHT);
        printf("----------------------------------------------------------\n");

        dim3 blockDim(4, 2);  /* 4 cols x 2 rows = 8 threads per block */
        dim3 gridDim(
            (WIDTH  + blockDim.x - 1) / blockDim.x,   /* ceil(6/4) = 2 */
            (HEIGHT + blockDim.y - 1) / blockDim.y     /* ceil(4/2) = 2 */
        );

        printf("Block dimensions: (%d, %d) = %d threads per block\n",
               blockDim.x, blockDim.y, blockDim.x * blockDim.y);
        printf("Grid dimensions:  (%d, %d) = %d blocks total\n",
               gridDim.x, gridDim.y, gridDim.x * gridDim.y);
        printf("\nThread assignments:\n");

        print_thread_info<<<gridDim, blockDim>>>(WIDTH, HEIGHT);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* -----------------------------------------------------------------------
     * Part 2: Row-major indexing on a larger matrix
     * -----------------------------------------------------------------------*/
    {
        const int WIDTH  = 8;
        const int HEIGHT = 6;
        const int SIZE   = WIDTH * HEIGHT;
        const int BYTES  = SIZE * sizeof(int);

        printf("\n----------------------------------------------------------\n");
        printf("Part 2: Row-Major Layout (%dx%d matrix)\n", WIDTH, HEIGHT);
        printf("----------------------------------------------------------\n");

        /* Allocate host and device memory */
        int *h_matrix = (int *)malloc(BYTES);
        int *d_matrix;
        CUDA_CHECK(cudaMalloc(&d_matrix, BYTES));

        /* Launch configuration:
         * blockDim = (4, 4) = 16 threads per block
         * gridDim  = (ceil(8/4), ceil(6/4)) = (2, 2) */
        dim3 blockDim(4, 4);
        dim3 gridDim(
            (WIDTH  + blockDim.x - 1) / blockDim.x,
            (HEIGHT + blockDim.y - 1) / blockDim.y
        );

        printf("Block dimensions: (%d, %d)\n", blockDim.x, blockDim.y);
        printf("Grid dimensions:  (%d, %d)\n", gridDim.x, gridDim.y);

        fill_row_major<<<gridDim, blockDim>>>(d_matrix, WIDTH, HEIGHT);
        CUDA_CHECK(cudaDeviceSynchronize());

        /* Copy result back and display */
        CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, BYTES, cudaMemcpyDeviceToHost));
        print_matrix_row_major(h_matrix, WIDTH, HEIGHT, "Result");

        printf("\nVerification: element at (row=2, col=5) should be 2005.\n");
        printf("  matrix[2*8 + 5] = matrix[21] = %d %s\n",
               h_matrix[2 * WIDTH + 5],
               h_matrix[2 * WIDTH + 5] == 2005 ? "(CORRECT)" : "(WRONG!)");

        /* Memory layout in linear memory:
         *
         * Address: 0     1     2     3     4     5     6     7     8  ...
         * Value:   0     1     2     3     4     5     6     7     1000 ...
         *          |<--------- row 0 --------->|<--------- row 1 -----...
         */
        printf("\nFirst 16 elements in linear memory (row-major):\n  ");
        for (int i = 0; i < 16 && i < SIZE; i++) {
            printf("%d ", h_matrix[i]);
        }
        printf("...\n");

        free(h_matrix);
        CUDA_CHECK(cudaFree(d_matrix));
    }

    /* -----------------------------------------------------------------------
     * Part 3: Column-major indexing on the same matrix
     * -----------------------------------------------------------------------*/
    {
        const int WIDTH  = 8;
        const int HEIGHT = 6;
        const int SIZE   = WIDTH * HEIGHT;
        const int BYTES  = SIZE * sizeof(int);

        printf("\n----------------------------------------------------------\n");
        printf("Part 3: Column-Major Layout (%dx%d matrix)\n", WIDTH, HEIGHT);
        printf("----------------------------------------------------------\n");

        int *h_matrix = (int *)malloc(BYTES);
        int *d_matrix;
        CUDA_CHECK(cudaMalloc(&d_matrix, BYTES));

        dim3 blockDim(4, 4);
        dim3 gridDim(
            (WIDTH  + blockDim.x - 1) / blockDim.x,
            (HEIGHT + blockDim.y - 1) / blockDim.y
        );

        fill_col_major<<<gridDim, blockDim>>>(d_matrix, WIDTH, HEIGHT);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, BYTES, cudaMemcpyDeviceToHost));
        print_matrix_col_major(h_matrix, WIDTH, HEIGHT, "Result");

        printf("\nVerification: element at (row=2, col=5) should be 2005.\n");
        printf("  matrix[5*6 + 2] = matrix[32] = %d %s\n",
               h_matrix[5 * HEIGHT + 2],
               h_matrix[5 * HEIGHT + 2] == 2005 ? "(CORRECT)" : "(WRONG!)");

        /* Memory layout in linear memory:
         *
         * Address: 0     1     2     3     4     5     6     7  ...
         * Value:   0     1000  2000  3000  4000  5000  1     1001 ...
         *          |<------- col 0 -------->|<------- col 1 -----...
         */
        printf("\nFirst 16 elements in linear memory (column-major):\n  ");
        for (int i = 0; i < 16 && i < SIZE; i++) {
            printf("%d ", h_matrix[i]);
        }
        printf("...\n");

        free(h_matrix);
        CUDA_CHECK(cudaFree(d_matrix));
    }

    /* -----------------------------------------------------------------------
     * Part 4: Performance note on memory coalescing
     * -----------------------------------------------------------------------*/
    printf("\n----------------------------------------------------------\n");
    printf("IMPORTANT: Row-Major vs Column-Major and Coalescing\n");
    printf("----------------------------------------------------------\n");
    printf("In a 2D block, threads with consecutive threadIdx.x are in\n");
    printf("the same warp. With row-major layout, these threads access\n");
    printf("consecutive memory addresses -> COALESCED (fast!).\n\n");
    printf("With column-major layout, consecutive threadIdx.x threads\n");
    printf("access addresses 'height' apart -> NOT COALESCED (slow!).\n\n");
    printf("Rule: Use row-major layout with threadIdx.x mapping to\n");
    printf("the fastest-varying dimension (columns).\n");

    printf("\n==========================================================\n");
    printf("  Done!\n");
    printf("==========================================================\n");

    return 0;
}
