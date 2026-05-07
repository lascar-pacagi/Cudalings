// ===========================================================================
// Chapter 11: Scan (Prefix Sum) -- scan_algorithms.cu
// ===========================================================================
// Two classic parallel scan algorithms implemented side by side:
//
//   1. Hillis-Steele  (inclusive scan, O(N log N) work, log N steps)
//   2. Blelloch       (exclusive scan, O(N) work, 2 log N steps)
//
// Both use shared memory within a single block.
// Target: Quadro P4200 (CC 6.1), CUDA 11.7
// ===========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// Constants
// ---------------------------------------------------------------------------
// Maximum elements per block for shared-memory scan.
// We use one block, so N must be <= BLOCK_SIZE.
// For large N, see scan_large.cu.
#define BLOCK_SIZE 1024

// ===========================================================================
// CPU Reference Implementations
// ===========================================================================

// ---------------------------------------------------------------------------
// cpu_inclusive_scan: Simple sequential inclusive prefix sum.
//   output[0] = input[0]
//   output[i] = output[i-1] + input[i]    for i > 0
// ---------------------------------------------------------------------------
void cpu_inclusive_scan(const int *input, int *output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// ---------------------------------------------------------------------------
// cpu_exclusive_scan: Simple sequential exclusive prefix sum.
//   output[0] = 0  (identity for addition)
//   output[i] = output[i-1] + input[i-1]  for i > 0
// ---------------------------------------------------------------------------
void cpu_exclusive_scan(const int *input, int *output, int n) {
    output[0] = 0;
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// ===========================================================================
// Kernel 1: Hillis-Steele Inclusive Scan
// ===========================================================================
//
// Algorithm (for N elements, one block):
//   Use TWO shared-memory buffers (double buffering) to avoid read-after-
//   write hazards within a step.
//
//   for d = 0 to log2(N)-1:
//       offset = 2^d
//       for each thread i in parallel:
//           if i >= offset:
//               out[i] = in[i] + in[i - offset]
//           else:
//               out[i] = in[i]
//       swap in/out buffers
//
// DIAGRAM for 8 elements [3, 1, 7, 0, 4, 1, 6, 3]:
// -----------------------------------------------------------------------
//
// Input:         3    1    7    0    4    1    6    3
//                0    1    2    3    4    5    6    7   <- indices
//
// d=0 (offset=1):
//   i=0: 3              (no i-1)
//   i=1: 1+3  = 4
//   i=2: 7+1  = 8
//   i=3: 0+7  = 7
//   i=4: 4+0  = 4
//   i=5: 1+4  = 5
//   i=6: 6+1  = 7
//   i=7: 3+6  = 9
// Result:        3    4    8    7    4    5    7    9
//              (sums of 1) (sums of 2 consecutive elements)
//
// d=1 (offset=2):
//   i=0: 3              (no i-2)
//   i=1: 4              (no i-2, i-2 = -1)
//   i=2: 8+3  = 11
//   i=3: 7+4  = 11
//   i=4: 4+8  = 12
//   i=5: 5+7  = 12
//   i=6: 7+4  = 11
//   i=7: 9+5  = 14
// Result:        3    4   11   11   12   12   11   14
//              (sums of up to 4 consecutive elements)
//
// d=2 (offset=4):
//   i=0: 3              (no i-4)
//   i=1: 4              (no i-4)
//   i=2: 11             (no i-4)
//   i=3: 11             (no i-4)
//   i=4: 12+3  = 15
//   i=5: 12+4  = 16
//   i=6: 11+11 = 22
//   i=7: 14+11 = 25
// Result:        3    4   11   11   15   16   22   25
//                                              DONE!
//
// Each element i now contains sum(input[0..i]) -- inclusive scan.
// -----------------------------------------------------------------------
__global__ void hillis_steele_inclusive_scan(const int *g_input,
                                             int       *g_output,
                                             int        n) {
    // We use two shared memory buffers and ping-pong between them.
    // This avoids the need for a separate __syncthreads() between
    // reading old values and writing new values within a step.
    __shared__ int buf[2][BLOCK_SIZE];

    int tid = threadIdx.x;
    if (tid >= n) return;

    // Load input into the first buffer.
    buf[0][tid] = g_input[tid];
    __syncthreads();

    // ping-pong index: read from buf[in_idx], write to buf[out_idx]
    int in_idx = 0;

    // log2(N) steps
    for (int offset = 1; offset < n; offset <<= 1) {
        int out_idx = 1 - in_idx;  // swap buffers

        if (tid >= offset) {
            // Add the element "offset" positions to the left
            buf[out_idx][tid] = buf[in_idx][tid] + buf[in_idx][tid - offset];
        } else {
            // Just copy -- this element's prefix sum is unchanged
            buf[out_idx][tid] = buf[in_idx][tid];
        }
        __syncthreads();

        in_idx = out_idx;  // the output becomes the input for next step
    }

    // Write final result to global memory
    g_output[tid] = buf[in_idx][tid];
}

// ===========================================================================
// Kernel 2: Blelloch Exclusive Scan
// ===========================================================================
//
// Algorithm (for N elements, N must be power of 2):
//
//   Phase 1 -- UP-SWEEP (reduce):
//     Build partial sums bottom-up, like a reduction tree.
//     for d = 0 to log2(N)-1:
//         stride = 2^(d+1)
//         for each k = stride-1, 2*stride-1, 3*stride-1, ...:
//             a[k] += a[k - stride/2]
//
//   Phase 2 -- DOWN-SWEEP (distribute):
//     Set root to 0, then push prefix sums down the tree.
//     for d = log2(N)-1 down to 0:
//         stride = 2^(d+1)
//         for each k = stride-1, 2*stride-1, 3*stride-1, ...:
//             temp   = a[k - stride/2]
//             a[k - stride/2] = a[k]       // left child = parent value
//             a[k]  += temp                 // right child = parent + old left
//
// DIAGRAM for 8 elements [3, 1, 7, 0, 4, 1, 6, 3]:
// -----------------------------------------------------------------------
//
// UP-SWEEP:
//
// Input:        [3]   [1]   [7]   [0]   [4]   [1]   [6]   [3]
//                0     1     2     3     4     5     6     7
//
// d=0 (stride=2): a[1]+=a[0], a[3]+=a[2], a[5]+=a[4], a[7]+=a[6]
//               [3]   [4]   [7]   [7]   [4]   [5]   [6]   [9]
//                      ^           ^           ^           ^
//
// d=1 (stride=4): a[3]+=a[1], a[7]+=a[5]
//               [3]   [4]   [7]  [11]   [4]   [5]   [6]  [14]
//                                  ^                        ^
//
// d=2 (stride=8): a[7]+=a[3]
//               [3]   [4]   [7]  [11]   [4]   [5]   [6]  [25]
//                                                           ^
//                                                      Total sum
//
// DOWN-SWEEP:
//
// Set a[7] = 0:
//               [3]   [4]   [7]  [11]   [4]   [5]   [6]   [0]
//
// d=2 (stride=8): swap-and-add at index 7
//   temp=a[3]=11, a[3]=a[7]=0, a[7]=0+11=11
//               [3]   [4]   [7]   [0]   [4]   [5]   [6]  [11]
//
// d=1 (stride=4): swap-and-add at indices 3 and 7
//   idx 3: temp=a[1]=4,  a[1]=a[3]=0,  a[3]=0+4=4
//   idx 7: temp=a[5]=5,  a[5]=a[7]=11, a[7]=11+5=16
//               [3]   [0]   [7]   [4]   [4]  [11]   [6]  [16]
//
// d=0 (stride=2): swap-and-add at indices 1, 3, 5, 7
//   idx 1: temp=a[0]=3,  a[0]=a[1]=0,   a[1]=0+3=3
//   idx 3: temp=a[2]=7,  a[2]=a[3]=4,   a[3]=4+7=11
//   idx 5: temp=a[4]=4,  a[4]=a[5]=11,  a[5]=11+4=15
//   idx 7: temp=a[6]=6,  a[6]=a[7]=16,  a[7]=16+6=22
//               [0]   [3]   [4]  [11]  [11]  [15]  [16]  [22]
//                                                    DONE!
//
// Exclusive scan of [3,1,7,0,4,1,6,3] = [0,3,4,11,11,15,16,22]  (correct!)
// -----------------------------------------------------------------------
__global__ void blelloch_exclusive_scan(const int *g_input,
                                         int       *g_output,
                                         int        n) {
    __shared__ int temp[BLOCK_SIZE];

    int tid = threadIdx.x;
    if (tid >= n) return;

    // Load input into shared memory
    temp[tid] = g_input[tid];
    __syncthreads();

    // ---------------------------------------------------------------
    // PHASE 1: Up-sweep (reduce)
    // ---------------------------------------------------------------
    // Build partial sums in a binary tree fashion.
    // After this phase, temp[n-1] contains the total sum.
    for (int stride = 2; stride <= n; stride <<= 1) {
        // Only threads at positions (stride-1), (2*stride-1), ... participate
        int idx = (tid + 1) * stride - 1;  // map thread to array index
        if (idx < n) {
            temp[idx] += temp[idx - stride / 2];
        }
        __syncthreads();
    }

    // ---------------------------------------------------------------
    // Set the root (last element) to 0 -- the identity for addition.
    // This is what makes it an EXCLUSIVE scan.
    // ---------------------------------------------------------------
    if (tid == 0) {
        temp[n - 1] = 0;
    }
    __syncthreads();

    // ---------------------------------------------------------------
    // PHASE 2: Down-sweep (distribute prefix sums)
    // ---------------------------------------------------------------
    // Traverse from root to leaves. At each node:
    //   left_child  = parent_value
    //   right_child = parent_value + old_left_child
    for (int stride = n; stride >= 2; stride >>= 1) {
        int idx = (tid + 1) * stride - 1;
        if (idx < n) {
            int left_idx = idx - stride / 2;
            int t        = temp[left_idx];   // save old left child
            temp[left_idx] = temp[idx];       // left child = parent value
            temp[idx]     += t;               // right child = parent + old left
        }
        __syncthreads();
    }

    // Write result to global memory
    g_output[tid] = temp[tid];
}

// ===========================================================================
// Verification helper
// ===========================================================================
// Returns the number of mismatches between two arrays.
int verify(const int *ref, const int *test, int n, const char *label) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (ref[i] != test[i]) {
            if (errors < 5) {  // Print at most 5 mismatches
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
// Benchmarking helper
// ===========================================================================
// Runs a kernel `nruns` times and returns the average time in milliseconds.
float benchmark_kernel(void (*launcher)(const int*, int*, int,
                                         int*, int*),
                       const int *d_input, int *d_output, int n,
                       int *d_dummy1, int *d_dummy2,
                       int nruns) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up
    launcher(d_input, d_output, n, d_dummy1, d_dummy2);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < nruns; i++) {
        launcher(d_input, d_output, n, d_dummy1, d_dummy2);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / nruns;
}

// ---------------------------------------------------------------------------
// Launcher wrappers (uniform signature for benchmark_kernel)
// ---------------------------------------------------------------------------
void launch_hillis_steele(const int *d_in, int *d_out, int n,
                           int * /*unused1*/, int * /*unused2*/) {
    hillis_steele_inclusive_scan<<<1, n>>>(d_in, d_out, n);
}

void launch_blelloch(const int *d_in, int *d_out, int n,
                      int * /*unused1*/, int * /*unused2*/) {
    blelloch_exclusive_scan<<<1, n>>>(d_in, d_out, n);
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    printf("===========================================================\n");
    printf("Chapter 11: Scan Algorithms -- Hillis-Steele vs. Blelloch\n");
    printf("===========================================================\n\n");

    // We test with multiple sizes (all powers of 2, <= BLOCK_SIZE)
    int test_sizes[] = {8, 32, 128, 256, 512, 1024};
    int num_tests    = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_tests; t++) {
        int n = test_sizes[t];
        printf("--- N = %d ---\n", n);

        // Allocate host arrays
        int *h_input       = (int *)malloc(n * sizeof(int));
        int *h_inc_ref     = (int *)malloc(n * sizeof(int));  // CPU inclusive
        int *h_exc_ref     = (int *)malloc(n * sizeof(int));  // CPU exclusive
        int *h_gpu_result  = (int *)malloc(n * sizeof(int));

        // Fill with small random values to avoid overflow
        srand(42);
        for (int i = 0; i < n; i++) {
            h_input[i] = rand() % 10;
        }

        // Print input for small sizes
        if (n <= 32) {
            printf("  Input: ");
            for (int i = 0; i < n; i++) printf("%d ", h_input[i]);
            printf("\n");
        }

        // CPU reference scans
        cpu_inclusive_scan(h_input, h_inc_ref, n);
        cpu_exclusive_scan(h_input, h_exc_ref, n);

        if (n <= 32) {
            printf("  CPU inclusive: ");
            for (int i = 0; i < n; i++) printf("%d ", h_inc_ref[i]);
            printf("\n");
            printf("  CPU exclusive: ");
            for (int i = 0; i < n; i++) printf("%d ", h_exc_ref[i]);
            printf("\n");
        }

        // Allocate device arrays
        int *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input,  n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_input, h_input, n * sizeof(int),
                              cudaMemcpyHostToDevice));

        // -----------------------------------------------------------------
        // Test 1: Hillis-Steele inclusive scan
        // -----------------------------------------------------------------
        CUDA_CHECK(cudaMemset(d_output, 0, n * sizeof(int)));
        hillis_steele_inclusive_scan<<<1, n>>>(d_input, d_output, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_gpu_result, d_output, n * sizeof(int),
                              cudaMemcpyDeviceToHost));

        if (n <= 32) {
            printf("  GPU Hillis-Steele (inclusive): ");
            for (int i = 0; i < n; i++) printf("%d ", h_gpu_result[i]);
            printf("\n");
        }
        verify(h_inc_ref, h_gpu_result, n, "Hillis-Steele inclusive");

        // -----------------------------------------------------------------
        // Test 2: Blelloch exclusive scan
        // -----------------------------------------------------------------
        CUDA_CHECK(cudaMemset(d_output, 0, n * sizeof(int)));
        blelloch_exclusive_scan<<<1, n>>>(d_input, d_output, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_gpu_result, d_output, n * sizeof(int),
                              cudaMemcpyDeviceToHost));

        if (n <= 32) {
            printf("  GPU Blelloch (exclusive): ");
            for (int i = 0; i < n; i++) printf("%d ", h_gpu_result[i]);
            printf("\n");
        }
        verify(h_exc_ref, h_gpu_result, n, "Blelloch exclusive");

        // -----------------------------------------------------------------
        // Benchmark (only for larger sizes to get meaningful times)
        // -----------------------------------------------------------------
        if (n >= 256) {
            int nruns = 10000;
            float ms_hs = benchmark_kernel(launch_hillis_steele,
                                           d_input, d_output, n,
                                           nullptr, nullptr, nruns);
            float ms_bl = benchmark_kernel(launch_blelloch,
                                           d_input, d_output, n,
                                           nullptr, nullptr, nruns);
            printf("  Benchmark (%d runs):\n", nruns);
            printf("    Hillis-Steele:  %.4f us\n", ms_hs * 1000.0f);
            printf("    Blelloch:       %.4f us\n", ms_bl * 1000.0f);
            printf("    Speedup (Blelloch/HS): %.2fx\n", ms_hs / ms_bl);
        }

        printf("\n");

        // Cleanup
        free(h_input);
        free(h_inc_ref);
        free(h_exc_ref);
        free(h_gpu_result);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
    }

    printf("===========================================================\n");
    printf("Key observations:\n");
    printf("  - Hillis-Steele does O(N log N) work but only log N steps.\n");
    printf("  - Blelloch does O(N) work but 2*log N steps.\n");
    printf("  - For small N (fits in one warp), Hillis-Steele may win.\n");
    printf("  - For large N, Blelloch's work-efficiency wins.\n");
    printf("  - For truly large arrays, see scan_large.cu.\n");
    printf("===========================================================\n");

    return 0;
}
