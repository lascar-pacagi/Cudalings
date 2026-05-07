/* ============================================================================
 * shuffle_demo.cu  --  Chapter 09: Warp Shuffle Operations Demonstrated
 * ============================================================================
 *
 * This program demonstrates all four warp shuffle variants:
 *   1. __shfl_sync       -- broadcast / direct read from any lane
 *   2. __shfl_up_sync    -- shift data toward lower-numbered lanes
 *   3. __shfl_down_sync  -- shift data toward higher-numbered lanes
 *   4. __shfl_xor_sync   -- butterfly (XOR) exchange between lanes
 *
 * We launch exactly 1 warp (32 threads) so we can clearly see the data
 * movement.  Each lane starts with its own lane ID as its value, then we
 * apply each shuffle and print the results.
 *
 * Target GPU : Quadro P4200  (CC 6.1, Pascal, 18 SMs)
 * CUDA       : 11.7
 * Compile    : nvcc -arch=sm_61 -O2 -lineinfo -ccbin g++-11 shuffle_demo.cu -o shuffle_demo
 * ========================================================================= */

#include <cstdio>

/* ---------------------------------------------------------------------------
 * FULL_MASK: every lane participates in the shuffle.
 *
 * 0xFFFFFFFF = 32 bits all set = all 32 lanes.  This is the most common
 * mask.  You must ensure that every lane executing the shuffle instruction
 * has its corresponding bit set in the mask, or behaviour is undefined.
 * ------------------------------------------------------------------------ */
#define FULL_MASK 0xFFFFFFFF

/* ===========================================================================
 * Helper: print an array of 32 ints from device memory, 8 per line.
 * We copy to host first (only 32 ints, negligible overhead).
 * ========================================================================= */
void printWarp(const char* label, const int* d_data) {
    int h[32];
    cudaMemcpy(h, d_data, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("  %s:\n", label);
    for (int i = 0; i < 32; i++) {
        if (i % 8 == 0) printf("    lane %2d-%2d: ", i, i + 7);
        printf("%3d ", h[i]);
        if (i % 8 == 7) printf("\n");
    }
    printf("\n");
}

/* ===========================================================================
 * Kernel: demonstrate __shfl_sync  (broadcast from a specific lane)
 *
 * Every lane reads the value from srcLane.
 *
 *   Before:  lane i holds value i
 *   After:   every lane holds value srcLane
 *
 *   Diagram (srcLane = 3):
 *   Lane:   0   1   2  [3]  4   5   6   7 ...
 *   Before: 0   1   2   3   4   5   6   7 ...
 *                        │
 *           ┌────┬────┬──┼──┬────┬────┬────┬────┐
 *           ↓    ↓    ↓  ↓  ↓    ↓    ↓    ↓
 *   After:  3    3    3  3  3    3    3    3 ...
 * ========================================================================= */
__global__ void demo_shfl_broadcast(int* out, int srcLane) {
    int lane = threadIdx.x;          // lane ID within warp (0-31)
    int val  = lane;                 // each lane starts with its own ID

    /* __shfl_sync(mask, value, srcLane, width=32)
     *
     * Parameters:
     *   mask    : 0xFFFFFFFF = all 32 lanes participate
     *   value   : the value THIS lane contributes (other lanes may read it)
     *   srcLane : which lane's value every participating lane receives
     *
     * Returns: the value held by srcLane
     */
    int result = __shfl_sync(FULL_MASK, val, srcLane);

    out[lane] = result;
}

/* ===========================================================================
 * Kernel: demonstrate __shfl_up_sync  (shift toward lower lanes)
 *
 * Each lane i reads from lane (i - delta).
 * Lanes where (i - delta) < 0 keep their original value.
 *
 *   Diagram (delta = 2):
 *   Lane:   0   1   2   3   4   5   6   7
 *   Before: 0   1   2   3   4   5   6   7
 *                   ↗   ↗   ↗   ↗   ↗   ↗
 *   After:  0   1   0   1   2   3   4   5
 *           ↑   ↑
 *         unchanged (no lane -2 or -1 to read from)
 *
 * Use case: prefix sums / exclusive scans.
 * ========================================================================= */
__global__ void demo_shfl_up(int* out, int delta) {
    int lane = threadIdx.x;
    int val  = lane;

    /* __shfl_up_sync(mask, value, delta, width=32)
     *
     * Each lane reads from (laneID - delta).
     * If (laneID - delta) < 0, the lane keeps its own value.
     */
    int result = __shfl_up_sync(FULL_MASK, val, delta);

    out[lane] = result;
}

/* ===========================================================================
 * Kernel: demonstrate __shfl_down_sync  (shift toward higher lanes)
 *
 * Each lane i reads from lane (i + delta).
 * Lanes where (i + delta) >= warpSize keep their original value.
 *
 *   Diagram (delta = 2):
 *   Lane:   0   1   2   3   4   5   6   7
 *   Before: 0   1   2   3   4   5   6   7
 *           ↘   ↘   ↘   ↘   ↘   ↘
 *   After:  2   3   4   5   6   7   6   7
 *                                   ↑   ↑
 *                         unchanged (no lane 8 or 9)
 *
 * Use case: REDUCTIONS -- the most important application.
 * ========================================================================= */
__global__ void demo_shfl_down(int* out, int delta) {
    int lane = threadIdx.x;
    int val  = lane;

    /* __shfl_down_sync(mask, value, delta, width=32)
     *
     * Each lane reads from (laneID + delta).
     * If (laneID + delta) >= 32, the lane keeps its own value.
     */
    int result = __shfl_down_sync(FULL_MASK, val, delta);

    out[lane] = result;
}

/* ===========================================================================
 * Kernel: demonstrate __shfl_xor_sync  (butterfly / XOR exchange)
 *
 * Each lane i reads from lane (i ^ laneMask).
 * Since XOR is symmetric (a^b = c  ⟹  c^b = a), pairs of lanes swap.
 *
 *   Diagram (laneMask = 1):
 *   Lane:   0   1   2   3   4   5   6   7
 *   Before: 0   1   2   3   4   5   6   7
 *           ↕       ↕       ↕       ↕
 *   After:  1   0   3   2   5   4   7   6
 *
 *   0^1=1, 1^1=0: lanes 0,1 swap
 *   2^1=3, 3^1=2: lanes 2,3 swap
 *   ...
 *
 *   Diagram (laneMask = 2):
 *   Lane:   0   1   2   3   4   5   6   7
 *   Before: 0   1   2   3   4   5   6   7
 *           ↕   ↕               ↕   ↕
 *   After:  2   3   0   1   6   7   4   5
 *
 *   0^2=2, 2^2=0: lanes 0,2 swap
 *   1^2=3, 3^2=1: lanes 1,3 swap
 *
 * Use case: butterfly reductions, parallel prefix, all-reduce patterns.
 * ========================================================================= */
__global__ void demo_shfl_xor(int* out, int laneMask) {
    int lane = threadIdx.x;
    int val  = lane;

    /* __shfl_xor_sync(mask, value, laneMask, width=32)
     *
     * Each lane reads from (laneID ^ laneMask).
     * The XOR creates symmetric exchange: if lane A reads from B,
     * then lane B reads from A.
     */
    int result = __shfl_xor_sync(FULL_MASK, val, laneMask);

    out[lane] = result;
}

/* ===========================================================================
 * Kernel: use shuffle for rotation (rotate left by N)
 *
 * Rotate is not a built-in shuffle op, but we can build it from __shfl_sync
 * by computing the source lane with modular arithmetic:
 *
 *   source = (laneID + N) % 32
 *
 *   Diagram (rotate left by 3):
 *   Lane:   0   1   2   3   4   5  ...  29  30  31
 *   Before: 0   1   2   3   4   5  ...  29  30  31
 *   After:  3   4   5   6   7   8  ...   0   1   2
 *           ↑                             ↑
 *         reads from lane 3            reads from lane 29+3=32%32=0
 * ========================================================================= */
__global__ void demo_rotate(int* out, int N) {
    int lane = threadIdx.x;
    int val  = lane;

    /* Compute source lane for a left-rotation:
     * Lane i should receive the value from lane (i + N) % 32.
     * __shfl_sync can read from an arbitrary source lane.
     */
    int srcLane = (lane + N) & 31;   // & 31 is equivalent to % 32
    int result  = __shfl_sync(FULL_MASK, val, srcLane);

    out[lane] = result;
}

/* ===========================================================================
 * Kernel: use XOR shuffle to reverse the warp
 *
 * To reverse 32 elements, lane i should get the value from lane (31 - i).
 * Note: 31 - i = i ^ 31  when we stay within 5 bits (0-31).
 * Wait -- that's only true when i XOR 31 = 31-i, which IS true for 0-31.
 *
 *   31 ^ 0  = 31,  31 ^ 1  = 30,  31 ^ 2  = 29  ... 31 ^ 31 = 0   ✓
 *
 *   Diagram:
 *   Lane:   0   1   2   3  ...  28  29  30  31
 *   Before: 0   1   2   3  ...  28  29  30  31
 *   After: 31  30  29  28  ...   3   2   1   0
 * ========================================================================= */
__global__ void demo_reverse(int* out) {
    int lane = threadIdx.x;
    int val  = lane;

    /* XOR with 31 (0b11111) flips all 5 lane-ID bits → reversal! */
    int result = __shfl_xor_sync(FULL_MASK, val, 31);

    out[lane] = result;
}

/* ===========================================================================
 * main()
 * ========================================================================= */
int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Chapter 09: Warp Shuffle Demonstration\n");
    printf("  Launching 1 warp (32 threads) for each demo\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    /* Allocate device memory for 32 ints (one per lane) */
    int* d_out;
    cudaMalloc(&d_out, 32 * sizeof(int));

    /* ── Demo 1: __shfl_sync (broadcast from lane 3) ────────────────── */
    printf("── Demo 1: __shfl_sync -- broadcast from lane 3 ──\n");
    printf("  Initial: each lane holds its own lane ID (0-31)\n\n");
    demo_shfl_broadcast<<<1, 32>>>(d_out, 3);
    cudaDeviceSynchronize();
    printWarp("After __shfl_sync(FULL_MASK, val, 3)", d_out);

    /* ── Demo 2: __shfl_up_sync (shift up by 2) ────────────────────── */
    printf("── Demo 2: __shfl_up_sync -- shift up by delta=2 ──\n");
    printf("  Initial: each lane holds its own lane ID (0-31)\n\n");
    demo_shfl_up<<<1, 32>>>(d_out, 2);
    cudaDeviceSynchronize();
    printWarp("After __shfl_up_sync(FULL_MASK, val, 2)", d_out);

    /* ── Demo 3: __shfl_down_sync (shift down by 2) ───────────────── */
    printf("── Demo 3: __shfl_down_sync -- shift down by delta=2 ──\n");
    printf("  Initial: each lane holds its own lane ID (0-31)\n\n");
    demo_shfl_down<<<1, 32>>>(d_out, 2);
    cudaDeviceSynchronize();
    printWarp("After __shfl_down_sync(FULL_MASK, val, 2)", d_out);

    /* ── Demo 4a: __shfl_xor_sync (XOR with 1 → neighbor swap) ────── */
    printf("── Demo 4a: __shfl_xor_sync -- XOR with laneMask=1 ──\n");
    printf("  Adjacent pairs swap: (0,1), (2,3), (4,5), ...\n\n");
    demo_shfl_xor<<<1, 32>>>(d_out, 1);
    cudaDeviceSynchronize();
    printWarp("After __shfl_xor_sync(FULL_MASK, val, 1)", d_out);

    /* ── Demo 4b: __shfl_xor_sync (XOR with 2 → stride-2 swap) ───── */
    printf("── Demo 4b: __shfl_xor_sync -- XOR with laneMask=2 ──\n");
    printf("  Stride-2 swap: (0,2), (1,3), (4,6), (5,7), ...\n\n");
    demo_shfl_xor<<<1, 32>>>(d_out, 2);
    cudaDeviceSynchronize();
    printWarp("After __shfl_xor_sync(FULL_MASK, val, 2)", d_out);

    /* ── Demo 4c: __shfl_xor_sync (XOR with 4 → stride-4 swap) ───── */
    printf("── Demo 4c: __shfl_xor_sync -- XOR with laneMask=4 ──\n");
    printf("  Stride-4 swap: (0,4), (1,5), (2,6), (3,7), ...\n\n");
    demo_shfl_xor<<<1, 32>>>(d_out, 4);
    cudaDeviceSynchronize();
    printWarp("After __shfl_xor_sync(FULL_MASK, val, 4)", d_out);

    /* ── Demo 5: Rotate left by 3 (built from __shfl_sync) ─────────── */
    printf("── Demo 5: Rotate left by 3 (using __shfl_sync) ──\n");
    printf("  Lane i reads from lane (i+3) %% 32\n\n");
    demo_rotate<<<1, 32>>>(d_out, 3);
    cudaDeviceSynchronize();
    printWarp("After rotate-left-3", d_out);

    /* ── Demo 6: Reverse (using __shfl_xor_sync with 31) ──────────── */
    printf("── Demo 6: Reverse warp (using __shfl_xor_sync with 31) ──\n");
    printf("  Lane i reads from lane (i ^ 31) = lane (31 - i)\n\n");
    demo_reverse<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    printWarp("After reverse", d_out);

    /* Clean up */
    cudaFree(d_out);

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Key takeaway: shuffles move data between registers within\n");
    printf("  a warp -- no shared memory, no synchronization, ~1-2 cycles.\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}
