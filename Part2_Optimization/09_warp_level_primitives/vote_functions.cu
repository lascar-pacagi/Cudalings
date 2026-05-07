/* ============================================================================
 * vote_functions.cu  --  Chapter 09: Warp Vote Functions
 * ============================================================================
 *
 * Warp vote functions let all 32 lanes collectively evaluate a boolean
 * condition.  They are single-instruction operations -- as fast as shuffles.
 *
 * Three vote functions:
 *   __all_sync(mask, predicate)    → true if ALL lanes have predicate true
 *   __any_sync(mask, predicate)    → true if ANY lane has predicate true
 *   __ballot_sync(mask, predicate) → 32-bit bitmask of which lanes are true
 *
 * Plus:
 *   __popc(x)       → population count (number of 1-bits in x)
 *   __activemask()  → bitmask of currently active lanes
 *
 * Target GPU : Quadro P4200  (CC 6.1, Pascal, 18 SMs)
 * CUDA       : 11.7
 * Compile    : nvcc -arch=sm_61 -O2 -lineinfo -ccbin g++-11 vote_functions.cu -o vote_functions
 * ========================================================================= */

#include <cstdio>

#define FULL_MASK 0xFFFFFFFF

/* ---------------------------------------------------------------------------
 * CUDA error checking macro.
 * ------------------------------------------------------------------------ */
#define CHECK_CUDA(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ===========================================================================
 * Demo 1: __all_sync — "are ALL threads in the warp true?"
 *
 * Scenario: checking if all elements in a warp satisfy a condition.
 *
 * Example: are all values positive?
 *
 *   Lane:    0   1   2   3   4   5   6   7   ...  31
 *   Value:   5   3   7   2   8   1   4   9   ...  6
 *   Pred:    T   T   T   T   T   T   T   T   ...  T
 *   __all_sync → 1 (all true)
 *
 *   Lane:    0   1   2   3   4   5   6   7   ...  31
 *   Value:   5   3  -1   2   8   1   4   9   ...  6
 *   Pred:    T   T   F   T   T   T   T   T   ...  T
 *   __all_sync → 0 (lane 2 is false)
 * ========================================================================= */
__global__ void demo_all_sync(const int* values, int* results) {
    int lane = threadIdx.x;    // 0-31

    /* Each lane checks if its value is positive */
    int pred = (values[lane] > 0);

    /* __all_sync returns non-zero if ALL participating lanes have pred != 0 */
    int allPositive = __all_sync(FULL_MASK, pred);

    /* Only lane 0 reports the result */
    if (lane == 0) {
        results[0] = allPositive;
    }
}

/* ===========================================================================
 * Demo 2: __any_sync — "is ANY thread in the warp true?"
 *
 * Scenario: checking if at least one element needs special handling.
 *
 *   Lane:    0   1   2   3   4   5   6   7   ...  31
 *   Value:   0   0   0   0   0   0   0   0   ...  0
 *   Pred:    F   F   F   F   F   F   F   F   ...  F
 *   __any_sync → 0 (none true)
 *
 *   Lane:    0   1   2   3   4   5   6   7   ...  31
 *   Value:   0   0   0   0   0   42  0   0   ...  0
 *   Pred:    F   F   F   F   F   T   F   F   ...  F
 *   __any_sync → 1 (lane 5 is true)
 * ========================================================================= */
__global__ void demo_any_sync(const int* values, int* results) {
    int lane = threadIdx.x;

    /* Check if this lane's value is non-zero */
    int pred = (values[lane] != 0);

    /* __any_sync returns non-zero if ANY participating lane has pred != 0 */
    int anyNonZero = __any_sync(FULL_MASK, pred);

    if (lane == 0) {
        results[0] = anyNonZero;
    }
}

/* ===========================================================================
 * Demo 3: __ballot_sync — "give me a bitmask of which lanes are true"
 *
 * This is the most informative vote function.  It returns a 32-bit integer
 * where bit i is set if lane i's predicate is non-zero.
 *
 *   Lane:    0   1   2   3   4   5   6   7   ...
 *   Pred:    T   F   T   T   F   T   F   F   ...
 *                                                │
 *   Result:  bit0=1, bit1=0, bit2=1, bit3=1, bit4=0, bit5=1, ...
 *            = 0x...00101101 (binary) = 0x2D (if upper bits are 0)
 *
 * Combined with __popc() (population count), you can count how many lanes
 * satisfy a condition without any shared memory or atomics.
 * ========================================================================= */
__global__ void demo_ballot_sync(const int* values, unsigned int* ballotResult,
                                  int* popcount, int threshold) {
    int lane = threadIdx.x;

    /* Predicate: is this lane's value above the threshold? */
    int pred = (values[lane] > threshold);

    /* __ballot_sync returns a 32-bit mask of which lanes are true */
    unsigned int ballot = __ballot_sync(FULL_MASK, pred);

    /* __popc counts the number of 1-bits (population count).
     * This gives us the count of lanes that satisfy the predicate,
     * computed in a SINGLE INSTRUCTION -- no reduction needed!
     */
    int count = __popc(ballot);

    if (lane == 0) {
        ballotResult[0] = ballot;
        popcount[0]     = count;
    }
}

/* ===========================================================================
 * Demo 4: Practical example — early exit when all threads are done
 *
 * Scenario: iterative convergence.  Each thread updates a value until it
 * converges (change < epsilon).  We want ALL threads in the warp to exit
 * the loop together (warp-level convergence check).
 *
 * Without __all_sync, you'd need shared memory and __syncthreads().
 * With __all_sync, it's a single instruction per iteration.
 * ========================================================================= */
__global__ void demo_early_exit(float* values, int* iterCounts) {
    int lane = threadIdx.x;
    float val = values[lane];
    float epsilon = 0.001f;
    int iter = 0;

    /*
     * Simulate an iterative process:
     * - Each lane starts with a different value (its initial value)
     * - Each iteration, the value is halved
     * - A lane "converges" when val < epsilon
     * - The warp exits when ALL lanes have converged
     *
     * Key insight: __all_sync checks convergence across the entire warp
     * in a single instruction -- no shared memory, no __syncthreads().
     */
    while (true) {
        /* Check if THIS lane has converged */
        int converged = (val < epsilon);

        /* Check if ALL lanes have converged.
         * If so, we can break -- the entire warp is done.
         * This is safe because warps execute in lockstep:
         * all lanes evaluate __all_sync at the same time.
         */
        if (__all_sync(FULL_MASK, converged)) {
            break;
        }

        /* If not all converged, keep iterating */
        val *= 0.5f;   // halve the value each iteration
        iter++;
    }

    /* Store how many iterations each lane took to be part of global convergence */
    iterCounts[lane] = iter;
    values[lane]     = val;
}

/* ===========================================================================
 * Demo 5: Using __ballot_sync for population count
 *
 * Practical example: count how many elements in an array satisfy a predicate,
 * using ballot + popc.  Much faster than atomicAdd for small counts.
 *
 * Strategy:
 *   1. Each lane checks its predicate
 *   2. __ballot_sync gets a bitmask of true lanes
 *   3. __popc counts the 1-bits → number of matching lanes
 *   4. One lane per warp does atomicAdd to the global count
 *
 * This reduces atomicAdd calls from 32 per warp to just 1 per warp.
 * ========================================================================= */
__global__ void countAboveThreshold(const float* data, int N,
                                     float threshold, int* count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;

    /* Each lane checks its element (OOB threads have predicate = false) */
    int pred = 0;
    if (tid < N) {
        pred = (data[tid] > threshold) ? 1 : 0;
    }

    /* Ballot: get bitmask of which lanes in this warp satisfy the predicate */
    unsigned int ballot = __ballot_sync(FULL_MASK, pred);

    /* Population count: how many 1-bits in the ballot?
     * This is the number of elements in THIS WARP above the threshold.
     */
    int warpCount = __popc(ballot);

    /* Only lane 0 of each warp atomically adds to the global counter.
     * This is 32x fewer atomics than having every thread do atomicAdd!
     *
     *   Without ballot: 32 atomicAdd per warp (serialized)
     *   With ballot:     1 atomicAdd per warp (32x fewer conflicts)
     */
    if (lane == 0) {
        atomicAdd(count, warpCount);
    }
}

/* ===========================================================================
 * Helper: print a 32-bit value as binary, grouped in nibbles
 * ========================================================================= */
void printBinary(unsigned int val) {
    for (int i = 31; i >= 0; i--) {
        printf("%d", (val >> i) & 1);
        if (i % 4 == 0 && i > 0) printf("_");
    }
}

/* ===========================================================================
 * main()
 * ========================================================================= */
int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Chapter 09: Warp Vote Functions\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    /* ── Demo 1: __all_sync ────────────────────────────────────────────── */
    {
        printf("── Demo 1: __all_sync (are ALL values positive?) ──\n\n");

        int h_values[32];
        int h_result;

        /* Test A: all positive */
        for (int i = 0; i < 32; i++) h_values[i] = i + 1;   // 1,2,3,...,32

        int *d_values, *d_result;
        CHECK_CUDA(cudaMalloc(&d_values, 32 * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));

        CHECK_CUDA(cudaMemcpy(d_values, h_values, 32 * sizeof(int), cudaMemcpyHostToDevice));
        demo_all_sync<<<1, 32>>>(d_values, d_result);
        CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  Test A (all positive 1-32):       __all_sync = %d  %s\n",
               h_result, h_result ? "(all positive)" : "(not all positive)");

        /* Test B: one negative value */
        h_values[15] = -5;   // lane 15 is negative
        CHECK_CUDA(cudaMemcpy(d_values, h_values, 32 * sizeof(int), cudaMemcpyHostToDevice));
        demo_all_sync<<<1, 32>>>(d_values, d_result);
        CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  Test B (lane 15 = -5):            __all_sync = %d  %s\n\n",
               h_result, h_result ? "(all positive)" : "(not all positive)");

        CHECK_CUDA(cudaFree(d_values));
        CHECK_CUDA(cudaFree(d_result));
    }

    /* ── Demo 2: __any_sync ────────────────────────────────────────────── */
    {
        printf("── Demo 2: __any_sync (is ANY value non-zero?) ──\n\n");

        int h_values[32];
        int h_result;

        int *d_values, *d_result;
        CHECK_CUDA(cudaMalloc(&d_values, 32 * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_result, sizeof(int)));

        /* Test A: all zeros */
        for (int i = 0; i < 32; i++) h_values[i] = 0;
        CHECK_CUDA(cudaMemcpy(d_values, h_values, 32 * sizeof(int), cudaMemcpyHostToDevice));
        demo_any_sync<<<1, 32>>>(d_values, d_result);
        CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  Test A (all zeros):               __any_sync = %d  %s\n",
               h_result, h_result ? "(found non-zero)" : "(all zero)");

        /* Test B: one non-zero */
        h_values[23] = 42;   // lane 23 has a non-zero value
        CHECK_CUDA(cudaMemcpy(d_values, h_values, 32 * sizeof(int), cudaMemcpyHostToDevice));
        demo_any_sync<<<1, 32>>>(d_values, d_result);
        CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  Test B (lane 23 = 42):            __any_sync = %d  %s\n\n",
               h_result, h_result ? "(found non-zero)" : "(all zero)");

        CHECK_CUDA(cudaFree(d_values));
        CHECK_CUDA(cudaFree(d_result));
    }

    /* ── Demo 3: __ballot_sync ─────────────────────────────────────────── */
    {
        printf("── Demo 3: __ballot_sync (bitmask of which lanes are > threshold) ──\n\n");

        int h_values[32];
        unsigned int h_ballot;
        int h_popcount;

        unsigned int *d_ballot;
        int *d_values, *d_popcount;
        CHECK_CUDA(cudaMalloc(&d_values,   32 * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_ballot,   sizeof(unsigned int)));
        CHECK_CUDA(cudaMalloc(&d_popcount, sizeof(int)));

        /* Set up: values = lane ID.  Threshold = 20.
         * Lanes 21-31 should be true (11 lanes).
         */
        for (int i = 0; i < 32; i++) h_values[i] = i;
        int threshold = 20;

        CHECK_CUDA(cudaMemcpy(d_values, h_values, 32 * sizeof(int), cudaMemcpyHostToDevice));
        demo_ballot_sync<<<1, 32>>>(d_values, d_ballot, d_popcount, threshold);
        CHECK_CUDA(cudaMemcpy(&h_ballot,   d_ballot,   sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_popcount, d_popcount, sizeof(int),          cudaMemcpyDeviceToHost));

        printf("  Values: lane ID (0-31), threshold = %d\n", threshold);
        printf("  Lanes with value > %d: lanes 21-31\n\n", threshold);
        printf("  Ballot bitmask: 0x%08X\n", h_ballot);
        printf("  Binary:         ");
        printBinary(h_ballot);
        printf("\n");
        printf("                  ^                          ^\n");
        printf("                  bit 31                     bit 0\n\n");
        printf("  Population count (__popc): %d lanes above threshold\n", h_popcount);
        printf("  Expected: 11 lanes (21,22,...,31)\n\n");

        /* Second test: even lanes only (values = lane*2, threshold = 30) */
        for (int i = 0; i < 32; i++) h_values[i] = (i % 2 == 0) ? 100 : 0;
        threshold = 50;

        CHECK_CUDA(cudaMemcpy(d_values, h_values, 32 * sizeof(int), cudaMemcpyHostToDevice));
        demo_ballot_sync<<<1, 32>>>(d_values, d_ballot, d_popcount, threshold);
        CHECK_CUDA(cudaMemcpy(&h_ballot,   d_ballot,   sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_popcount, d_popcount, sizeof(int),          cudaMemcpyDeviceToHost));

        printf("  Values: even lanes = 100, odd lanes = 0, threshold = %d\n", threshold);
        printf("  Ballot bitmask: 0x%08X\n", h_ballot);
        printf("  Binary:         ");
        printBinary(h_ballot);
        printf("\n");
        printf("  Population count: %d lanes above threshold\n", h_popcount);
        printf("  Expected: 16 lanes (all even lanes: 0,2,4,...,30)\n\n");

        CHECK_CUDA(cudaFree(d_values));
        CHECK_CUDA(cudaFree(d_ballot));
        CHECK_CUDA(cudaFree(d_popcount));
    }

    /* ── Demo 4: Early exit with __all_sync ────────────────────────────── */
    {
        printf("── Demo 4: Early exit with __all_sync (iterative convergence) ──\n\n");

        /* Each lane starts with a different value.
         * Lane i starts with (i+1) * 10.0.
         * Each iteration halves the value.
         * The lane converges when value < 0.001.
         * The warp exits when ALL lanes converge.
         *
         * Lane 0 starts at 10.0 → converges at ~14 iterations (10/2^14 ≈ 0.0006)
         * Lane 31 starts at 320.0 → converges at ~19 iterations (320/2^19 ≈ 0.0006)
         *
         * All lanes run for max(iterations) because __all_sync waits for ALL.
         */
        float h_values[32];
        for (int i = 0; i < 32; i++) h_values[i] = (float)(i + 1) * 10.0f;

        float *d_values;
        int   *d_iters;
        int    h_iters[32];
        CHECK_CUDA(cudaMalloc(&d_values, 32 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_iters,  32 * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_values, h_values, 32 * sizeof(float), cudaMemcpyHostToDevice));

        demo_early_exit<<<1, 32>>>(d_values, d_iters);
        CHECK_CUDA(cudaMemcpy(h_iters,  d_iters,  32 * sizeof(int),   cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_values, d_values, 32 * sizeof(float), cudaMemcpyDeviceToHost));

        printf("  Each lane starts with (lane+1)*10.0, halves each iteration.\n");
        printf("  Warp exits when ALL lanes have value < 0.001.\n\n");
        printf("  Lane  Start    Iters  Final Value\n");
        printf("  ────  ─────    ─────  ───────────\n");
        for (int i = 0; i < 32; i += 4) {
            printf("  %3d   %6.1f    %3d    %.6f\n",
                   i, (float)(i + 1) * 10.0f, h_iters[i], h_values[i]);
        }
        printf("\n  Notice: ALL lanes ran for the same number of iterations\n");
        printf("  (the maximum required by any lane), because __all_sync\n");
        printf("  keeps the warp running until every lane has converged.\n\n");

        CHECK_CUDA(cudaFree(d_values));
        CHECK_CUDA(cudaFree(d_iters));
    }

    /* ── Demo 5: ballot + popc for counting ────────────────────────────── */
    {
        printf("── Demo 5: Ballot + popc for fast counting ──\n\n");

        const int N = 1 << 20;   // 1M elements
        const int blockSize = 256;
        const int gridSize  = (N + blockSize - 1) / blockSize;

        /* Generate random data in [0, 100) */
        float* h_data = (float*)malloc(N * sizeof(float));
        srand(42);
        int cpuCount = 0;
        float threshold = 75.0f;
        for (int i = 0; i < N; i++) {
            h_data[i] = (float)(rand() % 100);
            if (h_data[i] > threshold) cpuCount++;
        }

        float *d_data;
        int   *d_count;
        CHECK_CUDA(cudaMalloc(&d_data,  N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_count, 0, sizeof(int)));

        countAboveThreshold<<<gridSize, blockSize>>>(d_data, N, threshold, d_count);

        int gpuCount;
        CHECK_CUDA(cudaMemcpy(&gpuCount, d_count, sizeof(int), cudaMemcpyDeviceToHost));

        printf("  Counting elements > %.0f in array of %d elements\n", threshold, N);
        printf("  CPU count: %d\n", cpuCount);
        printf("  GPU count: %d (using ballot + popc)\n", gpuCount);
        printf("  Match: %s\n\n", (cpuCount == gpuCount) ? "[PASS]" : "[FAIL]");

        printf("  Why ballot + popc is better than naive atomicAdd:\n");
        printf("  ┌────────────────────────────────────────────────────────┐\n");
        printf("  │  Naive: each thread does atomicAdd(count, pred)       │\n");
        printf("  │         = %d atomic ops (one per thread)          │\n", N);
        printf("  │                                                        │\n");
        printf("  │  Ballot: one atomicAdd per warp                        │\n");
        printf("  │         = %d atomic ops (32x fewer!)              │\n", N / 32);
        printf("  └────────────────────────────────────────────────────────┘\n\n");

        CHECK_CUDA(cudaFree(d_data));
        CHECK_CUDA(cudaFree(d_count));
        free(h_data);
    }

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Key takeaway: vote functions give warp-wide boolean queries\n");
    printf("  in a single instruction.  Combined with __popc, they enable\n");
    printf("  fast counting, early exit, and branch optimization.\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}
