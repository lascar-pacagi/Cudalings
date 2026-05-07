// ===========================================================================
// Chapter 11: Scan (Prefix Sum) -- scan_large.cu
// ===========================================================================
// Scan for arrays LARGER than one block using the three-phase approach:
//
//   Phase 1: Each block performs an exclusive scan on its chunk.
//            Save each block's total sum into an auxiliary array.
//
//   Phase 2: Exclusive scan of the block totals (recursive).
//
//   Phase 3: Add each block's offset (from phase 2) to every element
//            in that block.
//
// DIAGRAM -- Three-phase scan for 4 blocks:
// -----------------------------------------------------------------------
//
// Input:  [---- block 0 ----][---- block 1 ----][---- block 2 ----][---- block 3 ----]
//         [ a  b  c  d  e  f][ g  h  i  j  k  l][ m  n  o  p  q  r][ s  t  u  v  w  x]
//
// Phase 1 -- Block-level exclusive scan + save totals:
//
//   Block 0 scans:  [0, a, a+b, a+b+c, ...]   total T0 = a+b+c+d+e+f
//   Block 1 scans:  [0, g, g+h, g+h+i, ...]   total T1 = g+h+i+j+k+l
//   Block 2 scans:  [0, m, m+n, m+n+o, ...]   total T2 = m+n+o+p+q+r
//   Block 3 scans:  [0, s, s+t, s+t+u, ...]   total T3 = s+t+u+v+w+x
//
//   Block totals array: [T0, T1, T2, T3]
//
// Phase 2 -- Exclusive scan of block totals:
//
//   [T0, T1, T2, T3]  -->  [0, T0, T0+T1, T0+T1+T2]
//     = offsets for each block
//
// Phase 3 -- Add offsets back:
//
//   Block 0: each element += 0          (no change)
//   Block 1: each element += T0
//   Block 2: each element += T0+T1
//   Block 3: each element += T0+T1+T2
//
// Result: complete exclusive scan of the entire array!
// -----------------------------------------------------------------------
//
// Target: Quadro P4200 (CC 6.1), CUDA 11.7
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
// Each block scans ELEMENTS_PER_BLOCK elements.
// Each thread handles 2 elements (coalesced load), so we launch
// ELEMENTS_PER_BLOCK/2 threads per block.
// Using 512 threads * 2 = 1024 elements per block.
#define THREADS_PER_BLOCK  512
#define ELEMENTS_PER_BLOCK (2 * THREADS_PER_BLOCK)  // 1024

// ===========================================================================
// CPU Reference: sequential exclusive scan
// ===========================================================================
void cpu_exclusive_scan(const int *input, int *output, int n) {
    output[0] = 0;
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// ===========================================================================
// Kernel 1: Block-level Blelloch exclusive scan
// ===========================================================================
// Each block:
//   - Loads ELEMENTS_PER_BLOCK elements into shared memory
//   - Performs a Blelloch exclusive scan in shared memory
//   - Saves the block's total sum to d_block_sums[blockIdx.x]
//   - Writes the scanned elements back to global memory
//
// Each thread loads 2 elements to maximize shared memory utilization.
// This kernel handles the padding for blocks that extend beyond the array.
// ---------------------------------------------------------------------------
__global__ void blelloch_scan_block(const int *d_input,
                                     int       *d_output,
                                     int       *d_block_sums,
                                     int        n) {
    __shared__ int temp[ELEMENTS_PER_BLOCK];

    int tid       = threadIdx.x;
    int block_off = blockIdx.x * ELEMENTS_PER_BLOCK;

    // ---------------------------------------------------------------
    // Load 2 elements per thread into shared memory.
    // Pad with 0 if the index is out of bounds (0 is identity for +).
    // ---------------------------------------------------------------
    int ai = block_off + 2 * tid;
    int bi = block_off + 2 * tid + 1;
    temp[2 * tid]     = (ai < n) ? d_input[ai] : 0;
    temp[2 * tid + 1] = (bi < n) ? d_input[bi] : 0;
    __syncthreads();

    // ---------------------------------------------------------------
    // UP-SWEEP (reduce phase)
    // ---------------------------------------------------------------
    // Build partial sums in a tree. After this, temp[ELEMENTS_PER_BLOCK-1]
    // holds the total sum of this block.
    //
    // Step d=0: stride=2   -> add pairs (0,1), (2,3), (4,5), ...
    // Step d=1: stride=4   -> add pairs (1,3), (5,7), (9,11), ...
    // ...
    // Step d=log2(N)-1: stride=N -> single addition at the root
    // ---------------------------------------------------------------
    for (int stride = 1; stride < ELEMENTS_PER_BLOCK; stride <<= 1) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < ELEMENTS_PER_BLOCK) {
            temp[idx] += temp[idx - stride];
        }
        __syncthreads();
    }

    // ---------------------------------------------------------------
    // Save block total and set root to 0
    // ---------------------------------------------------------------
    if (tid == 0) {
        // Save the total sum for this block (before we overwrite it with 0).
        if (d_block_sums != nullptr) {
            d_block_sums[blockIdx.x] = temp[ELEMENTS_PER_BLOCK - 1];
        }
        // Set root to identity (0) for exclusive scan.
        temp[ELEMENTS_PER_BLOCK - 1] = 0;
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // DOWN-SWEEP (distribute phase)
    // ---------------------------------------------------------------
    // Traverse the tree from root to leaves, distributing prefix sums.
    //
    // At each node:
    //   left_child  = parent_value
    //   right_child = parent_value + old_left_child
    // ---------------------------------------------------------------
    for (int stride = ELEMENTS_PER_BLOCK / 2; stride >= 1; stride >>= 1) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < ELEMENTS_PER_BLOCK) {
            int left_idx   = idx - stride;
            int t          = temp[left_idx];
            temp[left_idx] = temp[idx];
            temp[idx]     += t;
        }
        __syncthreads();
    }

    // ---------------------------------------------------------------
    // Write results back to global memory
    // ---------------------------------------------------------------
    if (ai < n) d_output[ai] = temp[2 * tid];
    if (bi < n) d_output[bi] = temp[2 * tid + 1];
}

// ===========================================================================
// Kernel 2: Add block offsets (Phase 3)
// ===========================================================================
// After Phase 2 computes the prefix sum of block totals, this kernel adds
// each block's offset to every element in that block.
//
// block_offsets[blockIdx.x] = sum of all elements in blocks 0..blockIdx.x-1
// ---------------------------------------------------------------------------
__global__ void add_block_offsets(int *d_data,
                                  const int *d_block_offsets,
                                  int n) {
    int idx = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x;

    // Each thread adds the block offset to its element.
    // We launch ELEMENTS_PER_BLOCK threads per block.
    if (idx < n) {
        d_data[idx] += d_block_offsets[blockIdx.x];
    }
}

// ===========================================================================
// Recursive large-array exclusive scan
// ===========================================================================
// This function implements the three-phase approach recursively:
//   1. Scan each block -> partial scans + block totals
//   2. Recursively scan the block totals
//   3. Add block offsets back
//
// The recursion bottoms out when the array fits in a single block.
// ---------------------------------------------------------------------------
void exclusive_scan_recursive(const int *d_input, int *d_output, int n) {
    // How many blocks do we need?
    int num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    if (num_blocks == 1) {
        // Base case: fits in one block, just scan directly.
        blelloch_scan_block<<<1, THREADS_PER_BLOCK>>>(
            d_input, d_output, nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // -----------------------------------------------------------------
    // Phase 1: Scan each block independently and save block totals.
    // -----------------------------------------------------------------
    int *d_block_sums;
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));

    blelloch_scan_block<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_input, d_output, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());

    // -----------------------------------------------------------------
    // Phase 2: Recursively scan the block totals.
    // This gives us the offset for each block (exclusive prefix sum
    // of the block totals).
    // -----------------------------------------------------------------
    int *d_block_offsets;
    CUDA_CHECK(cudaMalloc(&d_block_offsets, num_blocks * sizeof(int)));

    // Recursive call! If num_blocks > ELEMENTS_PER_BLOCK, this will
    // recurse further. Each level reduces the problem by a factor of
    // ELEMENTS_PER_BLOCK, so the recursion depth is
    // log_{ELEMENTS_PER_BLOCK}(N) -- typically just 2-3 levels.
    exclusive_scan_recursive(d_block_sums, d_block_offsets, num_blocks);

    // -----------------------------------------------------------------
    // Phase 3: Add block offsets to each element.
    // -----------------------------------------------------------------
    // We launch ELEMENTS_PER_BLOCK threads per block (one thread per
    // element) for the addition.
    add_block_offsets<<<num_blocks, ELEMENTS_PER_BLOCK>>>(
        d_output, d_block_offsets, n);
    CUDA_CHECK(cudaGetLastError());

    // Cleanup intermediate arrays
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_block_offsets));
}

// ===========================================================================
// Verification helper
// ===========================================================================
int verify(const int *ref, const int *test, int n, const char *label) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (ref[i] != test[i]) {
            if (errors < 5) {
                printf("  MISMATCH [%s] at index %d: expected %d, got %d\n",
                       label, i, ref[i], test[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("  [%s] PASSED -- all %d elements match.\n", label, n);
    } else {
        printf("  [%s] FAILED -- %d / %d mismatches.\n", label, errors, n);
    }
    return errors;
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    printf("===========================================================\n");
    printf("Chapter 11: Large-Array Scan (Three-Phase Approach)\n");
    printf("===========================================================\n\n");

    // Test with various sizes, including non-power-of-2
    int test_sizes[] = {
        1024,          // 1 block (base case)
        2048,          // 2 blocks
        10000,         // ~10 blocks, non-power-of-2
        100000,        // ~98 blocks
        1000000,       // ~977 blocks
        10000000       // ~9766 blocks (10 million)
    };
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];
        printf("--- N = %d", n);
        if (n >= 1000000) printf(" (%.1fM)", n / 1e6);
        printf(" ---\n");

        int num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        printf("  Blocks needed: %d (ELEMENTS_PER_BLOCK = %d)\n",
               num_blocks, ELEMENTS_PER_BLOCK);

        // Allocate host arrays
        int *h_input  = (int *)malloc(n * sizeof(int));
        int *h_ref    = (int *)malloc(n * sizeof(int));
        int *h_result = (int *)malloc(n * sizeof(int));

        // Fill with small random values (avoid overflow for large N)
        srand(42);
        for (int i = 0; i < n; i++) {
            h_input[i] = rand() % 4;  // values 0-3, sum won't overflow int
        }

        // CPU reference exclusive scan
        cpu_exclusive_scan(h_input, h_ref, n);

        // Allocate device arrays
        int *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input,  n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(int),
                              cudaMemcpyHostToDevice));

        // -----------------------------------------------------------------
        // Run the recursive three-phase scan
        // -----------------------------------------------------------------
        exclusive_scan_recursive(d_input, d_output, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_result, d_output, n * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // Verify
        verify(h_ref, h_result, n, "Large exclusive scan");

        // Print first and last few elements for spot check
        if (n <= 32) {
            printf("  Input:  ");
            for (int i = 0; i < n; i++) printf("%d ", h_input[i]);
            printf("\n  Output: ");
            for (int i = 0; i < n; i++) printf("%d ", h_result[i]);
            printf("\n");
        } else {
            printf("  First 8: ");
            for (int i = 0; i < 8; i++) printf("%d ", h_result[i]);
            printf("...\n");
            printf("  Last  8: ");
            for (int i = n - 8; i < n; i++) printf("%d ", h_result[i]);
            printf("...\n");
        }

        // -----------------------------------------------------------------
        // Benchmark
        // -----------------------------------------------------------------
        if (n >= 10000) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            int nruns = 100;

            // Warm up
            exclusive_scan_recursive(d_input, d_output, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < nruns; r++) {
                exclusive_scan_recursive(d_input, d_output, n);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= nruns;

            // Compute effective bandwidth
            // We read N ints and write N ints
            float bytes    = 2.0f * n * sizeof(int);
            float gbps     = (bytes / 1e9) / (ms / 1e3);

            printf("  Benchmark (%d runs): %.3f ms  (%.2f GB/s effective)\n",
                   nruns, ms, gbps);

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }

        printf("\n");

        // Cleanup
        free(h_input);
        free(h_ref);
        free(h_result);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    printf("===========================================================\n");
    printf("Summary:\n");
    printf("  - Three-phase scan handles arbitrary array sizes.\n");
    printf("  - Phase 1: block-level Blelloch scan + save totals.\n");
    printf("  - Phase 2: recursive scan of block totals.\n");
    printf("  - Phase 3: add block offsets to each element.\n");
    printf("  - Recursion depth is typically 2-3 for millions of elements.\n");
    printf("  - For production use, consider CUB's DeviceScan or\n");
    printf("    thrust::exclusive_scan which are highly optimized.\n");
    printf("===========================================================\n");

    return 0;
}
