/*
 * occupancy_demo.cu
 * =================
 * Explore GPU occupancy limits using CUDA runtime APIs.
 *
 * This demo shows:
 *   1. How to query a kernel's register usage with cudaFuncGetAttributes
 *   2. How to compute occupancy with cudaOccupancyMaxActiveBlocksPerMultiprocessor
 *   3. How register pressure (controlled via __launch_bounds__) affects occupancy
 *   4. How shared memory usage affects occupancy
 *
 * Target: Quadro P4200 (CC 6.1, 18 SMs, 2048 threads/SM, 64K regs/SM, 48KB shmem/SM)
 * Build:  nvcc -arch=sm_61 -O2 -lineinfo -ccbin g++-11 occupancy_demo.cu -o occupancy_demo
 */

#include <cstdio>
#include <cuda_runtime.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * ERROR CHECKING MACRO
 * ═══════════════════════════════════════════════════════════════════════════ */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 1: Low register pressure
 * ─────────────────────────────────────────────────────────────────────────
 * __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
 *
 * __launch_bounds__ is a hint to the compiler:
 *   - maxThreadsPerBlock: we promise never to launch with more threads
 *   - minBlocksPerMultiprocessor: we want at least this many blocks per SM
 *
 * When minBlocks is high, the compiler uses FEWER registers per thread
 * to allow more blocks to fit on the SM. This may cause register spilling
 * to local memory (slow), but increases occupancy.
 *
 * Here: 256 threads/block, want at least 8 blocks/SM = 2048 threads = 100%
 * The compiler will try to use at most 65536 / (8 * 256) = 32 regs/thread
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void __launch_bounds__(256, 8)
kernel_low_regs(const float* __restrict__ input, float* __restrict__ output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        /* Simple operation: very few registers needed.
         * The compiler only needs registers for idx, input[idx], and the result. */
        float val = input[idx];
        output[idx] = val * 2.0f + 1.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 2: Medium register pressure
 * ─────────────────────────────────────────────────────────────────────────
 * We use more local variables, which the compiler must keep in registers.
 * With __launch_bounds__(256, 4), the compiler targets at most:
 *   65536 / (4 * 256) = 64 regs/thread
 * This allows 4 blocks per SM = 1024 threads = 50% occupancy.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void __launch_bounds__(256, 4)
kernel_med_regs(const float* __restrict__ input, float* __restrict__ output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        /* More computation = more intermediate values = more registers.
         * Each of these variables may occupy a register. */
        float a = input[idx];
        float b = a * a;                    // square
        float c = b * a;                    // cube
        float d = sqrtf(fabsf(a));           // square root
        float e = sinf(a);                   // trig (uses several regs internally)
        float f = cosf(a);
        float g = expf(-a * a);             // Gaussian
        float h = logf(fabsf(a) + 1.0f);    // log
        float i = fmaf(b, c, d);            // fused multiply-add
        float j = fmaf(e, f, g);
        float k = fmaf(h, i, j);
        output[idx] = k;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 3: High register pressure
 * ─────────────────────────────────────────────────────────────────────────
 * Many live variables simultaneously → compiler needs many registers.
 * __launch_bounds__(256, 2) targets: 65536 / (2 * 256) = 128 regs/thread
 * (CC 6.1 max is 255 regs/thread)
 * This limits us to 2 blocks/SM = 512 threads = 25% occupancy.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void __launch_bounds__(256, 2)
kernel_high_regs(const float* __restrict__ input, float* __restrict__ output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        /* Intentionally create many live variables.
         * "Live" means the variable is defined but not yet consumed,
         * so the compiler must keep it in a register.
         *
         * The array below will likely be placed in registers (not memory)
         * because the indices are compile-time constants. */
        float v[16];
        float x = input[idx];

        /* Fill all 16 values — all are "live" simultaneously */
        v[0]  = x * 1.1f;
        v[1]  = x * 2.2f + v[0];
        v[2]  = x * 3.3f + v[1];
        v[3]  = sinf(v[0]) + cosf(v[1]);
        v[4]  = expf(-v[2] * v[2]);
        v[5]  = logf(fabsf(v[3]) + 1.0f);
        v[6]  = sqrtf(fabsf(v[4]));
        v[7]  = v[5] * v[6] + v[0];
        v[8]  = fmaf(v[1], v[2], v[3]);
        v[9]  = fmaf(v[4], v[5], v[6]);
        v[10] = fmaf(v[7], v[8], v[9]);
        v[11] = sinf(v[10]) * cosf(v[7]);
        v[12] = v[11] + v[0] + v[1] + v[2] + v[3];
        v[13] = v[12] * v[4] + v[5] * v[6];
        v[14] = v[13] - v[7] + v[8] * v[9];
        v[15] = v[14] + v[10] + v[11] + v[12] + v[13];

        /* Sum all — forces all 16 values to stay live until here */
        float result = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            result += v[i];
        }
        output[idx] = result;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 4: Shared memory hog
 * ─────────────────────────────────────────────────────────────────────────
 * Uses a large static shared memory array to demonstrate how shared memory
 * limits occupancy even when registers are not the bottleneck.
 *
 * 48 KB shared per SM on CC 6.1:
 *   - If block uses 24 KB: 48/24 = 2 blocks/SM max from shared mem
 *   - 2 blocks × 256 threads = 512 threads = 25% occupancy
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void __launch_bounds__(256, 8)
kernel_shared_heavy(const float* __restrict__ input, float* __restrict__ output, int N)
{
    /* 24 KB of shared memory per block.
     * This is a static allocation — the size is known at compile time.
     * 6144 floats × 4 bytes = 24,576 bytes = 24 KB */
    __shared__ float smem[6144];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    /* Each thread loads multiple elements into shared memory */
    for (int i = tid; i < 6144; i += blockDim.x) {
        int gidx = blockIdx.x * 6144 + i;
        smem[i] = (gidx < N) ? input[gidx] : 0.0f;
    }
    __syncthreads();

    /* Simple reduction-like operation using shared data */
    if (idx < N) {
        float sum = 0.0f;
        /* Access a few shared memory locations */
        for (int offset = 0; offset < 6144; offset += 256) {
            sum += smem[(tid + offset) % 6144];
        }
        output[idx] = sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 5: Shared memory via dynamic allocation
 * ─────────────────────────────────────────────────────────────────────────
 * Dynamic shared memory is specified at launch time via the third <<<>>>
 * parameter. This lets us vary the amount at runtime and see how it
 * affects occupancy.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void __launch_bounds__(256, 8)
kernel_dynamic_shared(const float* __restrict__ input, float* __restrict__ output, int N)
{
    /* Dynamic shared memory — size is set in the <<<grid, block, shmem>>> call.
     * Declared as "extern __shared__" with no size. */
    extern __shared__ float dyn_smem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    /* Load into dynamic shared memory */
    if (tid < blockDim.x) {
        dyn_smem[tid] = (idx < N) ? input[idx] : 0.0f;
    }
    __syncthreads();

    if (idx < N) {
        /* Simple operation using shared data */
        float val = dyn_smem[tid];
        output[idx] = val * 2.0f;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HELPER: Print device properties relevant to occupancy
 * ═══════════════════════════════════════════════════════════════════════════ */
void print_device_info()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║             GPU OCCUPANCY LIMITS (Device 0)             ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Device:                    %-27s ║\n", prop.name);
    printf("║  Compute Capability:        %-27d.%d║\n", prop.major, prop.minor);
    printf("║  Number of SMs:             %-27d ║\n", prop.multiProcessorCount);
    printf("║  Max threads per SM:        %-27d ║\n", prop.maxThreadsPerMultiProcessor);
    printf("║  Max warps per SM:          %-27d ║\n", maxWarpsPerSM);
    printf("║  Max blocks per SM:         %-27d ║\n", prop.maxBlocksPerMultiProcessor);
    printf("║  Max threads per block:     %-27d ║\n", prop.maxThreadsPerBlock);
    printf("║  Registers per SM:          %-27d ║\n", prop.regsPerMultiprocessor);
    printf("║  Registers per block:       %-27d ║\n", prop.regsPerBlock);
    printf("║  Shared memory per SM:      %-23d bytes║\n", (int)prop.sharedMemPerMultiprocessor);
    printf("║  Shared memory per block:   %-23d bytes║\n", (int)prop.sharedMemPerBlock);
    printf("║  Warp size:                 %-27d ║\n", prop.warpSize);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HELPER: Analyze occupancy for a given kernel
 * ─────────────────────────────────────────────────────────────────────────
 * This function:
 *   1. Queries the kernel's register count via cudaFuncGetAttributes
 *   2. Queries max active blocks per SM via the occupancy API
 *   3. Calculates and prints the theoretical occupancy
 *
 * Parameters:
 *   kernel_func  — pointer to the __global__ function
 *   name         — human-readable name for printing
 *   blockSize    — threads per block
 *   dynSharedMem — bytes of dynamic shared memory per block
 * ═══════════════════════════════════════════════════════════════════════════ */
template <typename KernelFunc>
void analyze_occupancy(KernelFunc kernel_func, const char* name,
                       int blockSize, size_t dynSharedMem)
{
    /* ── Step 1: Query function attributes ──
     * cudaFuncGetAttributes tells us how many registers the compiler
     * assigned to this kernel, and how much static shared memory it uses. */
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_func));

    /* ── Step 2: Query occupancy ──
     * This is the key API. It considers ALL three limiters:
     *   - registers per thread
     *   - shared memory (static + dynamic) per block
     *   - thread/block slot limits
     * and returns the maximum number of blocks that can be active on one SM. */
    int maxActiveBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,     // output
        kernel_func,          // kernel to analyze
        blockSize,            // threads per block
        dynSharedMem          // dynamic shared memory bytes
    ));

    /* ── Step 3: Calculate occupancy ──
     * Occupancy = (active warps) / (max warps per SM) */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / 32;

    int activeWarps    = maxActiveBlocks * (blockSize / 32);
    float occupancy    = 100.0f * activeWarps / (float)maxWarpsPerSM;
    int activeThreads  = maxActiveBlocks * blockSize;

    /* Total shared memory = static (compiled into kernel) + dynamic (passed at launch) */
    size_t totalSharedPerBlock = attr.sharedSizeBytes + dynSharedMem;

    printf("  %-24s │ %4d regs │ %6zu B shmem │ %2d blocks │ %4d threads │ %5.1f%%\n",
           name,
           attr.numRegs,
           totalSharedPerBlock,
           maxActiveBlocks,
           activeThreads,
           occupancy);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main()
{
    printf("\n");
    print_device_info();

    /* ──────────────────────────────────────────────────────────────────────
     * SECTION 1: Register pressure comparison
     * ────────────────────────────────────────────────────────────────────── */
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  OCCUPANCY ANALYSIS: Varying Register Pressure (blockSize = 256)\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  %-24s │ %-9s │ %-13s │ %-9s │ %-12s │ %-5s\n",
           "Kernel", "Regs/Thd", "Shared/Block", "Blks/SM", "Threads/SM", "Occ");
    printf("  ────────────────────────┼───────────┼───────────────┼───────────┼──────────────┼──────\n");

    /* All kernels use blockSize=256, no dynamic shared memory.
     * The ONLY difference is register pressure, controlled by __launch_bounds__
     * and the complexity of the kernel code. */
    analyze_occupancy(kernel_low_regs,     "Low regs (simple)",     256, 0);
    analyze_occupancy(kernel_med_regs,     "Med regs (math)",       256, 0);
    analyze_occupancy(kernel_high_regs,    "High regs (16 vars)",   256, 0);

    printf("\n");
    printf("  Observation: More registers per thread → fewer blocks fit → lower occupancy\n");
    printf("  The compiler assigns registers based on kernel complexity and __launch_bounds__.\n\n");

    /* ──────────────────────────────────────────────────────────────────────
     * SECTION 2: Shared memory effects
     * ──────────────────────────────────────────────────────────────────────
     * We use kernel_dynamic_shared and vary the dynamic shared memory size
     * to show how shared memory usage limits occupancy.
     *
     * CC 6.1 has 48 KB = 49,152 bytes shared memory per SM.
     *   - 1 KB dynamic:  49152/1024  = 48 blocks possible (but capped by other limits)
     *   - 8 KB dynamic:  49152/8192  = 6 blocks
     *   - 16 KB dynamic: 49152/16384 = 3 blocks
     *   - 24 KB dynamic: 49152/24576 = 2 blocks
     *   - 48 KB dynamic: 49152/49152 = 1 block
     */
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  OCCUPANCY ANALYSIS: Varying Shared Memory (blockSize = 256)\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  %-24s │ %-9s │ %-13s │ %-9s │ %-12s │ %-5s\n",
           "Config", "Regs/Thd", "Shared/Block", "Blks/SM", "Threads/SM", "Occ");
    printf("  ────────────────────────┼───────────┼───────────────┼───────────┼──────────────┼──────\n");

    /* Static shared memory kernel (24 KB hardcoded) */
    analyze_occupancy(kernel_shared_heavy,   "Static 24KB shared",   256, 0);

    /* Dynamic shared memory kernel with varying sizes */
    analyze_occupancy(kernel_dynamic_shared, "Dynamic 1 KB",         256,  1024);
    analyze_occupancy(kernel_dynamic_shared, "Dynamic 8 KB",         256,  8192);
    analyze_occupancy(kernel_dynamic_shared, "Dynamic 16 KB",        256, 16384);
    analyze_occupancy(kernel_dynamic_shared, "Dynamic 24 KB",        256, 24576);
    analyze_occupancy(kernel_dynamic_shared, "Dynamic 48 KB",        256, 49152);

    printf("\n");
    printf("  Observation: More shared memory per block → fewer blocks fit per SM.\n");
    printf("  At 48 KB (the entire SM budget), only 1 block can run at a time.\n\n");

    /* ──────────────────────────────────────────────────────────────────────
     * SECTION 3: Block size effects
     * ──────────────────────────────────────────────────────────────────────
     * Using kernel_low_regs (minimal register pressure, no shared memory)
     * to isolate the effect of block size on occupancy.
     *
     * Key insight: CC 6.1 allows max 32 blocks per SM.
     *   - blockSize=32:  32 blocks × 32 = 1024 threads → 50% (block limit!)
     *   - blockSize=64:  32 blocks × 64 = 2048 threads → 100%
     *   - blockSize=128: 16 blocks × 128 = 2048 → 100%
     *   - blockSize=256: 8 blocks × 256 = 2048 → 100%
     *   - blockSize=512: 4 blocks × 512 = 2048 → 100%
     *   - blockSize=1024: 2 blocks × 1024 = 2048 → 100%
     */
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  OCCUPANCY ANALYSIS: Varying Block Size (low-register kernel)\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  %-24s │ %-9s │ %-13s │ %-9s │ %-12s │ %-5s\n",
           "Block Size", "Regs/Thd", "Shared/Block", "Blks/SM", "Threads/SM", "Occ");
    printf("  ────────────────────────┼───────────┼───────────────┼───────────┼──────────────┼──────\n");

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    char label[64];
    for (int i = 0; i < 6; i++) {
        snprintf(label, sizeof(label), "blockSize = %d", blockSizes[i]);
        analyze_occupancy(kernel_low_regs, label, blockSizes[i], 0);
    }

    printf("\n");
    printf("  Observation: blockSize=32 hits the 32-blocks-per-SM limit on CC 6.1.\n");
    printf("  Beyond blockSize=64, occupancy is the same — but more blocks per SM\n");
    printf("  gives better load balancing and scheduling flexibility.\n\n");

    /* ──────────────────────────────────────────────────────────────────────
     * SUMMARY DIAGRAM
     * ────────────────────────────────────────────────────────────────────── */
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  OCCUPANCY LIMITERS — Which one is active?\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────┐\n");
    printf("  │                   SM Resource Pool                      │\n");
    printf("  │                                                         │\n");
    printf("  │  Registers: 65536   Shared Mem: 48KB   Thread Slots: 2048│\n");
    printf("  │       │                  │                    │          │\n");
    printf("  │       ▼                  ▼                    ▼          │\n");
    printf("  │  ┌──────────┐    ┌──────────────┐    ┌──────────────┐   │\n");
    printf("  │  │ Regs/thd │    │ Shared/block │    │ Threads/blk  │   │\n");
    printf("  │  │ ×threads │    │ ×num blocks  │    │ ×num blocks  │   │\n");
    printf("  │  │ ×blocks  │    │ ≤ 48KB       │    │ ≤ 2048       │   │\n");
    printf("  │  └────┬─────┘    └──────┬───────┘    └──────┬───────┘   │\n");
    printf("  │       └────────────────┬┘                   │           │\n");
    printf("  │                   ┌────▼────────────────────▼┐          │\n");
    printf("  │                   │ Occupancy = min of all   │          │\n");
    printf("  │                   │ three constraints        │          │\n");
    printf("  │                   └──────────────────────────┘          │\n");
    printf("  └──────────────────────────────────────────────────────────┘\n");
    printf("\n");

    return 0;
}
