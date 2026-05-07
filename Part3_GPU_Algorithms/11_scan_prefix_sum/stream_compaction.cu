// ===========================================================================
// Chapter 11: Scan (Prefix Sum) -- stream_compaction.cu
// ===========================================================================
// Stream compaction: remove elements that do not satisfy a predicate.
//
// This is one of the most important practical applications of prefix sum.
// It is used in:
//   - Physics simulation: remove dead particles
//   - Ray tracing: compact active rays
//   - ML / sparse ops: select non-zero elements
//   - Graphics: compact visible primitives after culling
//   - Database queries: filter rows matching a condition
//
// ALGORITHM (3 steps):
// -----------------------------------------------------------------------
//
// Example: remove all zeros from [3, 0, 7, 0, 0, 1, 6, 0, 4, 0]
//
// Step 1: Compute predicate array (1 if we keep the element, 0 if not)
//
//   Input:      [3,  0,  7,  0,  0,  1,  6,  0,  4,  0]
//   Predicate:  [1,  0,  1,  0,  0,  1,  1,  0,  1,  0]
//                ^       ^             ^   ^       ^
//              keep    keep          keep keep   keep
//
// Step 2: Exclusive prefix sum of predicate -> scatter addresses
//
//   Predicate:  [1,  0,  1,  0,  0,  1,  1,  0,  1,  0]
//   Exc. scan:  [0,  1,  1,  2,  2,  2,  3,  4,  4,  5]
//                ^       ^             ^   ^       ^
//               "write  "write       "write "write "write
//               to 0"   to 1"       to 2"  to 3"  to 4"
//
//   The exclusive scan tells each kept element WHERE to write in the
//   output array. The total count of kept elements = last scan value
//   + last predicate value = 5 + 0 = 5.
//
// Step 3: Scatter -- each element with predicate=1 writes to output[scan[i]]
//
//   i=0: pred=1, scan=0 -> output[0] = input[0] = 3
//   i=1: pred=0          -> skip
//   i=2: pred=1, scan=1 -> output[1] = input[2] = 7
//   i=3: pred=0          -> skip
//   i=4: pred=0          -> skip
//   i=5: pred=1, scan=2 -> output[2] = input[5] = 1
//   i=6: pred=1, scan=3 -> output[3] = input[6] = 6
//   i=7: pred=0          -> skip
//   i=8: pred=1, scan=4 -> output[4] = input[8] = 4
//   i=9: pred=0          -> skip
//
//   Output: [3, 7, 1, 6, 4]  -- all zeros removed!
//   Count:  5
//
// VISUAL DIAGRAM of the complete pipeline:
// -----------------------------------------------------------------------
//
//   Input:     [ 3 | 0 | 7 | 0 | 0 | 1 | 6 | 0 | 4 | 0 ]
//                |   |   |   |   |   |   |   |   |   |
//                v   v   v   v   v   v   v   v   v   v
//   Predicate: [ 1 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 ]   (keep non-zeros)
//                |   |   |   |   |   |   |   |   |   |
//                v   v   v   v   v   v   v   v   v   v
//   Exc. Scan: [ 0 | 1 | 1 | 2 | 2 | 2 | 3 | 4 | 4 | 5 ]   (scatter addresses)
//                |       |               |   |       |
//   Scatter:    [0]=3   [1]=7          [2]=1 [3]=6  [4]=4
//                |       |               |   |       |
//                v       v               v   v       v
//   Output:    [ 3 | 7 | 1 | 6 | 4 ]    count = 5
//
// -----------------------------------------------------------------------
//
// Target: Quadro P4200 (CC 6.1), CUDA 11.7
// ===========================================================================

#include <cstdio>
#include <cstdlib>
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
// Configuration (must match scan_large.cu approach)
// ---------------------------------------------------------------------------
#define THREADS_PER_BLOCK  512
#define ELEMENTS_PER_BLOCK (2 * THREADS_PER_BLOCK)  // 1024

// ===========================================================================
// CPU Reference: stream compaction (remove zeros)
// ===========================================================================
int cpu_stream_compaction(const int *input, int *output, int n) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (input[i] != 0) {
            output[count++] = input[i];
        }
    }
    return count;
}

// ===========================================================================
// Kernel 1: Compute predicate array
// ===========================================================================
// For each element, write 1 if we keep it, 0 if we discard it.
// Here our predicate is "element != 0", but this could be any condition.
// ---------------------------------------------------------------------------
__global__ void compute_predicate(const int *d_input,
                                   int       *d_predicate,
                                   int        n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Predicate: keep non-zero elements
        d_predicate[idx] = (d_input[idx] != 0) ? 1 : 0;
    }
}

// ===========================================================================
// Kernel 2: Block-level Blelloch exclusive scan (same as scan_large.cu)
// ===========================================================================
// Each block scans ELEMENTS_PER_BLOCK elements.
// Each thread handles 2 elements.
// Saves block totals to d_block_sums.
// ---------------------------------------------------------------------------
__global__ void blelloch_scan_block(const int *d_input,
                                     int       *d_output,
                                     int       *d_block_sums,
                                     int        n) {
    __shared__ int temp[ELEMENTS_PER_BLOCK];

    int tid       = threadIdx.x;
    int block_off = blockIdx.x * ELEMENTS_PER_BLOCK;

    // Load 2 elements per thread (pad with 0 for out-of-bounds)
    int ai = block_off + 2 * tid;
    int bi = block_off + 2 * tid + 1;
    temp[2 * tid]     = (ai < n) ? d_input[ai] : 0;
    temp[2 * tid + 1] = (bi < n) ? d_input[bi] : 0;
    __syncthreads();

    // Up-sweep (reduce)
    for (int stride = 1; stride < ELEMENTS_PER_BLOCK; stride <<= 1) {
        int idx = (tid + 1) * 2 * stride - 1;
        if (idx < ELEMENTS_PER_BLOCK) {
            temp[idx] += temp[idx - stride];
        }
        __syncthreads();
    }

    // Save block total and set root to 0
    if (tid == 0) {
        if (d_block_sums != nullptr) {
            d_block_sums[blockIdx.x] = temp[ELEMENTS_PER_BLOCK - 1];
        }
        temp[ELEMENTS_PER_BLOCK - 1] = 0;
    }
    __syncthreads();

    // Down-sweep (distribute)
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

    // Write back
    if (ai < n) d_output[ai] = temp[2 * tid];
    if (bi < n) d_output[bi] = temp[2 * tid + 1];
}

// ===========================================================================
// Kernel 3: Add block offsets (Phase 3 of large scan)
// ===========================================================================
__global__ void add_block_offsets(int *d_data,
                                  const int *d_block_offsets,
                                  int n) {
    int idx = blockIdx.x * ELEMENTS_PER_BLOCK + threadIdx.x;
    if (idx < n) {
        d_data[idx] += d_block_offsets[blockIdx.x];
    }
}

// ===========================================================================
// Recursive large-array exclusive scan (same as scan_large.cu)
// ===========================================================================
void exclusive_scan_recursive(const int *d_input, int *d_output, int n) {
    int num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    if (num_blocks == 1) {
        blelloch_scan_block<<<1, THREADS_PER_BLOCK>>>(
            d_input, d_output, nullptr, n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    int *d_block_sums, *d_block_offsets;
    CUDA_CHECK(cudaMalloc(&d_block_sums,    num_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, num_blocks * sizeof(int)));

    // Phase 1: block-level scan
    blelloch_scan_block<<<num_blocks, THREADS_PER_BLOCK>>>(
        d_input, d_output, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());

    // Phase 2: scan block totals (recursive)
    exclusive_scan_recursive(d_block_sums, d_block_offsets, num_blocks);

    // Phase 3: add offsets
    add_block_offsets<<<num_blocks, ELEMENTS_PER_BLOCK>>>(
        d_output, d_block_offsets, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_block_offsets));
}

// ===========================================================================
// Kernel 4: Scatter -- write kept elements to their computed addresses
// ===========================================================================
// For each element where predicate[i] == 1:
//   output[scan_result[i]] = input[i]
//
// This is the final step of stream compaction.
// ---------------------------------------------------------------------------
__global__ void scatter(const int *d_input,
                        const int *d_predicate,
                        const int *d_scan_result,
                        int       *d_output,
                        int        n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (d_predicate[idx] == 1) {
            // The exclusive scan tells us WHERE to write this element
            int write_pos = d_scan_result[idx];
            d_output[write_pos] = d_input[idx];
        }
    }
}

// ===========================================================================
// GPU stream compaction (orchestrator)
// ===========================================================================
// Performs the full three-step pipeline:
//   1. Compute predicate
//   2. Exclusive scan of predicate
//   3. Scatter
// Returns the number of elements that passed the predicate.
// ---------------------------------------------------------------------------
int gpu_stream_compaction(const int *d_input,
                           int       *d_output,
                           int        n) {
    // -----------------------------------------------------------------
    // Step 1: Compute predicate array
    // -----------------------------------------------------------------
    int *d_predicate;
    CUDA_CHECK(cudaMalloc(&d_predicate, n * sizeof(int)));

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    compute_predicate<<<blocks, threads>>>(d_input, d_predicate, n);
    CUDA_CHECK(cudaGetLastError());

    // -----------------------------------------------------------------
    // Step 2: Exclusive scan of predicate array
    // -----------------------------------------------------------------
    // This gives us the scatter addresses. The total count of kept
    // elements = scan[n-1] + predicate[n-1].
    int *d_scan_result;
    CUDA_CHECK(cudaMalloc(&d_scan_result, n * sizeof(int)));

    exclusive_scan_recursive(d_predicate, d_scan_result, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get the total count of kept elements:
    //   count = scan_result[n-1] + predicate[n-1]
    int last_scan, last_pred;
    CUDA_CHECK(cudaMemcpy(&last_scan, d_scan_result + (n - 1),
                          sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_pred, d_predicate + (n - 1),
                          sizeof(int), cudaMemcpyDeviceToHost));
    int count = last_scan + last_pred;

    // -----------------------------------------------------------------
    // Step 3: Scatter kept elements to output
    // -----------------------------------------------------------------
    scatter<<<blocks, threads>>>(d_input, d_predicate, d_scan_result,
                                  d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    CUDA_CHECK(cudaFree(d_predicate));
    CUDA_CHECK(cudaFree(d_scan_result));

    return count;
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    printf("===========================================================\n");
    printf("Chapter 11: Stream Compaction Using Scan\n");
    printf("===========================================================\n\n");

    // =====================================================================
    // Test 1: Small example (matches the diagram in comments)
    // =====================================================================
    {
        printf("--- Test 1: Small example (remove zeros) ---\n");
        int h_input[] = {3, 0, 7, 0, 0, 1, 6, 0, 4, 0};
        int n = 10;

        printf("  Input:    ");
        for (int i = 0; i < n; i++) printf("%d ", h_input[i]);
        printf("\n");

        // CPU reference
        int h_cpu_output[10];
        int cpu_count = cpu_stream_compaction(h_input, h_cpu_output, n);
        printf("  CPU out:  ");
        for (int i = 0; i < cpu_count; i++) printf("%d ", h_cpu_output[i]);
        printf(" (count=%d)\n", cpu_count);

        // GPU
        int *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input,  n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(int),
                              cudaMemcpyHostToDevice));

        int gpu_count = gpu_stream_compaction(d_input, d_output, n);

        int h_gpu_output[10];
        CUDA_CHECK(cudaMemcpy(h_gpu_output, d_output,
                              gpu_count * sizeof(int),
                              cudaMemcpyDeviceToHost));

        printf("  GPU out:  ");
        for (int i = 0; i < gpu_count; i++) printf("%d ", h_gpu_output[i]);
        printf(" (count=%d)\n", gpu_count);

        // Verify
        bool pass = (cpu_count == gpu_count);
        for (int i = 0; i < cpu_count && pass; i++) {
            if (h_cpu_output[i] != h_gpu_output[i]) pass = false;
        }
        printf("  Result:   %s\n\n", pass ? "PASSED" : "FAILED");

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    // =====================================================================
    // Test 2: Large random array with ~50% zeros
    // =====================================================================
    {
        int test_sizes[] = {10000, 100000, 1000000, 10000000};
        int num_tests    = sizeof(test_sizes) / sizeof(test_sizes[0]);

        for (int t = 0; t < num_tests; t++) {
            int n = test_sizes[t];
            printf("--- Test: N = %d", n);
            if (n >= 1000000) printf(" (%.1fM)", n / 1e6);
            printf(", ~50%% zeros ---\n");

            // Allocate and fill with random values (0-9, lots of zeros)
            int *h_input  = (int *)malloc(n * sizeof(int));
            int *h_cpu_out = (int *)malloc(n * sizeof(int));
            int *h_gpu_out = (int *)malloc(n * sizeof(int));

            srand(12345);
            int zero_count = 0;
            for (int i = 0; i < n; i++) {
                // ~50% chance of zero
                h_input[i] = (rand() % 2 == 0) ? 0 : (rand() % 100 + 1);
                if (h_input[i] == 0) zero_count++;
            }
            printf("  Actual zeros: %d (%.1f%%)\n",
                   zero_count, 100.0f * zero_count / n);

            // CPU reference
            int cpu_count = cpu_stream_compaction(h_input, h_cpu_out, n);
            printf("  CPU kept: %d elements\n", cpu_count);

            // GPU
            int *d_input, *d_output;
            CUDA_CHECK(cudaMalloc(&d_input,  n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(int),
                                  cudaMemcpyHostToDevice));

            int gpu_count = gpu_stream_compaction(d_input, d_output, n);
            printf("  GPU kept: %d elements\n", gpu_count);

            // Copy back and verify
            CUDA_CHECK(cudaMemcpy(h_gpu_out, d_output,
                                  gpu_count * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            int errors = 0;
            if (cpu_count != gpu_count) {
                printf("  COUNT MISMATCH: CPU=%d, GPU=%d\n",
                       cpu_count, gpu_count);
                errors++;
            } else {
                for (int i = 0; i < cpu_count; i++) {
                    if (h_cpu_out[i] != h_gpu_out[i]) {
                        if (errors < 5) {
                            printf("  MISMATCH at output[%d]: "
                                   "CPU=%d, GPU=%d\n",
                                   i, h_cpu_out[i], h_gpu_out[i]);
                        }
                        errors++;
                    }
                }
            }
            printf("  Verification: %s\n",
                   errors == 0 ? "PASSED" : "FAILED");

            // Benchmark
            if (n >= 100000) {
                cudaEvent_t start, stop;
                CUDA_CHECK(cudaEventCreate(&start));
                CUDA_CHECK(cudaEventCreate(&stop));

                int nruns = 50;

                // Warm up
                gpu_stream_compaction(d_input, d_output, n);

                CUDA_CHECK(cudaEventRecord(start));
                for (int r = 0; r < nruns; r++) {
                    gpu_stream_compaction(d_input, d_output, n);
                }
                CUDA_CHECK(cudaEventRecord(stop));
                CUDA_CHECK(cudaEventSynchronize(stop));

                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                ms /= nruns;

                // Effective throughput: read input + write output + intermediates
                float input_bytes  = n * sizeof(int);
                float output_bytes = gpu_count * sizeof(int);
                float gbps = ((input_bytes + output_bytes) / 1e9) / (ms / 1e3);

                printf("  Benchmark (%d runs): %.3f ms  (%.2f GB/s)\n",
                       nruns, ms, gbps);

                CUDA_CHECK(cudaEventDestroy(start));
                CUDA_CHECK(cudaEventDestroy(stop));
            }

            printf("\n");

            free(h_input);
            free(h_cpu_out);
            free(h_gpu_out);
            CUDA_CHECK(cudaFree(d_input));
            CUDA_CHECK(cudaFree(d_output));
        }
    }

    printf("===========================================================\n");
    printf("Summary:\n");
    printf("  Stream compaction uses scan as its core building block:\n");
    printf("    1. Compute predicate (which elements to keep)\n");
    printf("    2. Exclusive scan -> scatter addresses\n");
    printf("    3. Scatter kept elements to output\n");
    printf("\n");
    printf("  Real-world uses:\n");
    printf("    - Physics: remove dead particles from simulation\n");
    printf("    - Rendering: compact active rays in path tracer\n");
    printf("    - ML: select non-zero gradients (sparse training)\n");
    printf("    - Databases: filter rows matching a WHERE clause\n");
    printf("===========================================================\n");

    return 0;
}
