/*
 * register_pressure.cu
 * ====================
 * Demonstrate the effects of register pressure on GPU performance.
 *
 * This demo shows:
 *   1. Kernels with LOW vs HIGH register usage
 *   2. The effect of -maxrregcount (capping registers at compile time)
 *   3. The tradeoff: capping registers → higher occupancy BUT register spilling
 *   4. Finding the "sweet spot" between occupancy and register spilling
 *
 * WHAT IS REGISTER SPILLING?
 * ─────────────────────────────────────────────────────────────────────────
 * When a kernel needs more registers than the compiler can allocate
 * (due to hardware limits or -maxrregcount), the compiler "spills" some
 * register values to LOCAL MEMORY.
 *
 * Local memory is actually in GLOBAL MEMORY (DRAM) — it's per-thread
 * private memory that is cached in L1/L2, but MUCH slower than registers:
 *
 *   Register access:     ~0 cycles (same cycle as instruction)
 *   L1 cache (local):    ~28 cycles
 *   L2 cache:            ~200 cycles
 *   DRAM (uncached):     ~400-800 cycles
 *
 * So spilling trades occupancy for memory latency. Sometimes the higher
 * occupancy wins; sometimes the spilling cost dominates.
 *
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │              Register Pressure Tradeoff                     │
 *   │                                                             │
 *   │  Few regs/thread:  Many threads fit → high occupancy        │
 *   │                    But may spill → slow memory accesses     │
 *   │                                                             │
 *   │  Many regs/thread: Fewer threads fit → low occupancy        │
 *   │                    But all data in registers → fast!        │
 *   │                                                             │
 *   │  The sweet spot depends on the kernel.                      │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * HOW TO SEE REGISTER USAGE AT COMPILE TIME:
 *   nvcc -Xptxas -v -arch=sm_61 register_pressure.cu
 *
 * This prints lines like:
 *   ptxas info: Used 32 registers, 0 bytes smem, 0 bytes cmem[0]
 *
 * HOW TO CAP REGISTERS:
 *   nvcc -maxrregcount=32 ...    (global cap: all kernels in the file)
 *   __launch_bounds__(256, 8)    (per-kernel hint: compiler infers max regs)
 *
 * Target: Quadro P4200 (CC 6.1, 18 SMs)
 * Build:  nvcc -arch=sm_61 -O2 -lineinfo -ccbin g++-11 register_pressure.cu -o register_pressure
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * ERROR CHECKING
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
 * KERNEL 1: Low register usage — simple element-wise operation
 * ─────────────────────────────────────────────────────────────────────────
 * Only a handful of registers needed: idx, val, output address.
 * Compiler will use ~16-20 registers per thread.
 * With 65536 regs/SM, this allows ~2048+ threads = 100% occupancy.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_low_regs(const float* __restrict__ input,
                                float* __restrict__ output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = input[idx];
        output[idx] = val * val + val;   /* 2 FLOPs, ~3 registers */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 2: High register usage — many live variables
 * ─────────────────────────────────────────────────────────────────────────
 * This kernel intentionally keeps many values "live" (defined but not yet
 * consumed). Each live value needs a register. The compiler cannot reuse
 * a register until its current value is no longer needed.
 *
 * We also use #pragma unroll to force loop unrolling, which creates
 * even more live variables (all loop iterations exist simultaneously).
 *
 * Expected: ~60-128+ registers per thread depending on compiler decisions.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void kernel_high_regs(const float* __restrict__ input,
                                 float* __restrict__ output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];

        /* ── Stage 1: Create many independent values ──
         * These are all "live" at the same time because they're all
         * needed in the final sum. The compiler must keep them in registers. */
        float a0 = sinf(x * 1.0f);
        float a1 = cosf(x * 1.1f);
        float a2 = sinf(x * 1.2f);
        float a3 = cosf(x * 1.3f);
        float a4 = sinf(x * 1.4f);
        float a5 = cosf(x * 1.5f);
        float a6 = sinf(x * 1.6f);
        float a7 = cosf(x * 1.7f);

        /* ── Stage 2: More values derived from stage 1 ──
         * These depend on stage 1 values, which must ALSO remain live. */
        float b0 = a0 * a1 + expf(-a2 * a2);
        float b1 = a2 * a3 + sqrtf(fabsf(a4));
        float b2 = a4 * a5 + logf(fabsf(a6) + 1.0f);
        float b3 = a6 * a7 + a0;
        float b4 = fmaf(a1, a3, a5);
        float b5 = fmaf(a0, a2, a4);
        float b6 = fmaf(a1, a7, a6);
        float b7 = fmaf(a3, a5, a7);

        /* ── Stage 3: More derived values ──
         * Now a0-a7 AND b0-b7 all need registers simultaneously. */
        float c0 = b0 * b1 + a0 * a1;
        float c1 = b2 * b3 + a2 * a3;
        float c2 = b4 * b5 + a4 * a5;
        float c3 = b6 * b7 + a6 * a7;

        /* ── Stage 4: Unrolled loop to add more pressure ── */
        float accum = c0 + c1 + c2 + c3;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            /* Each iteration creates temporaries that may be kept in registers.
             * With #pragma unroll, ALL iterations expand simultaneously. */
            float t1 = sinf(accum + (float)i);
            float t2 = cosf(accum - (float)i);
            float t3 = fmaf(t1, t2, accum);
            accum = t3 * 0.999f + 0.001f;
        }

        /* ── Final: sum everything (forces all values to be "live" at stage 3) ── */
        output[idx] = accum + a0 + a7 + b0 + b7;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 3: Same as kernel_high_regs but with __launch_bounds__ to cap regs
 * ─────────────────────────────────────────────────────────────────────────
 * __launch_bounds__(256, 8) means:
 *   - Max 256 threads per block
 *   - Want at least 8 blocks per SM
 *   - 8 × 256 = 2048 threads → 100% occupancy target
 *   - Register budget: 65536 / 2048 = 32 regs/thread
 *
 * The compiler will TRY to fit in 32 registers. If the kernel needs more,
 * it will SPILL to local memory. This is the classic tradeoff:
 *   Higher occupancy (good) vs. spilling overhead (bad)
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void __launch_bounds__(256, 8)
kernel_high_regs_capped(const float* __restrict__ input,
                        float* __restrict__ output, int N)
{
    /* IDENTICAL code to kernel_high_regs — only __launch_bounds__ differs.
     * The compiler generates different register allocation / spilling. */

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];

        float a0 = sinf(x * 1.0f);
        float a1 = cosf(x * 1.1f);
        float a2 = sinf(x * 1.2f);
        float a3 = cosf(x * 1.3f);
        float a4 = sinf(x * 1.4f);
        float a5 = cosf(x * 1.5f);
        float a6 = sinf(x * 1.6f);
        float a7 = cosf(x * 1.7f);

        float b0 = a0 * a1 + expf(-a2 * a2);
        float b1 = a2 * a3 + sqrtf(fabsf(a4));
        float b2 = a4 * a5 + logf(fabsf(a6) + 1.0f);
        float b3 = a6 * a7 + a0;
        float b4 = fmaf(a1, a3, a5);
        float b5 = fmaf(a0, a2, a4);
        float b6 = fmaf(a1, a7, a6);
        float b7 = fmaf(a3, a5, a7);

        float c0 = b0 * b1 + a0 * a1;
        float c1 = b2 * b3 + a2 * a3;
        float c2 = b4 * b5 + a4 * a5;
        float c3 = b6 * b7 + a6 * a7;

        float accum = c0 + c1 + c2 + c3;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float t1 = sinf(accum + (float)i);
            float t2 = cosf(accum - (float)i);
            float t3 = fmaf(t1, t2, accum);
            accum = t3 * 0.999f + 0.001f;
        }

        output[idx] = accum + a0 + a7 + b0 + b7;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNEL 4: Medium register pressure with __launch_bounds__ targeting 50%
 * ─────────────────────────────────────────────────────────────────────────
 * __launch_bounds__(256, 4) → 4 × 256 = 1024 threads → 50% occupancy
 * Register budget: 65536 / 1024 = 64 regs/thread
 *
 * This is a compromise: more registers available than the 100%-occupancy
 * version, but still constrained. Often this is the sweet spot.
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void __launch_bounds__(256, 4)
kernel_high_regs_medium_cap(const float* __restrict__ input,
                            float* __restrict__ output, int N)
{
    /* Same code, different __launch_bounds__ → different register allocation */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];

        float a0 = sinf(x * 1.0f);
        float a1 = cosf(x * 1.1f);
        float a2 = sinf(x * 1.2f);
        float a3 = cosf(x * 1.3f);
        float a4 = sinf(x * 1.4f);
        float a5 = cosf(x * 1.5f);
        float a6 = sinf(x * 1.6f);
        float a7 = cosf(x * 1.7f);

        float b0 = a0 * a1 + expf(-a2 * a2);
        float b1 = a2 * a3 + sqrtf(fabsf(a4));
        float b2 = a4 * a5 + logf(fabsf(a6) + 1.0f);
        float b3 = a6 * a7 + a0;
        float b4 = fmaf(a1, a3, a5);
        float b5 = fmaf(a0, a2, a4);
        float b6 = fmaf(a1, a7, a6);
        float b7 = fmaf(a3, a5, a7);

        float c0 = b0 * b1 + a0 * a1;
        float c1 = b2 * b3 + a2 * a3;
        float c2 = b4 * b5 + a4 * a5;
        float c3 = b6 * b7 + a6 * a7;

        float accum = c0 + c1 + c2 + c3;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float t1 = sinf(accum + (float)i);
            float t2 = cosf(accum - (float)i);
            float t3 = fmaf(t1, t2, accum);
            accum = t3 * 0.999f + 0.001f;
        }

        output[idx] = accum + a0 + a7 + b0 + b7;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCHMARK HELPER
 * ═══════════════════════════════════════════════════════════════════════════ */
template <typename KernelFunc>
float benchmark_kernel(KernelFunc kernel, const float* d_in, float* d_out,
                       int N, int blockSize, int warmupRuns, int timedRuns)
{
    int gridSize = (N + blockSize - 1) / blockSize;

    for (int i = 0; i < warmupRuns; i++) {
        kernel<<<gridSize, blockSize>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < timedRuns; i++) {
        kernel<<<gridSize, blockSize>>>(d_in, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / timedRuns;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main()
{
    const int N = 1 << 24;   /* 16M elements */
    const int warmup = 5;
    const int runs   = 20;
    const int blockSize = 256;
    size_t bytes = N * sizeof(float);

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  REGISTER PRESSURE DEMO\n");
    printf("  N = %d elements, blockSize = %d\n", N, blockSize);
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");

    /* ── Device info ── */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int maxWarps = prop.maxThreadsPerMultiProcessor / 32;

    printf("  Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("  Max threads per SM: %d\n\n", prop.maxThreadsPerMultiProcessor);

    /* ── Allocate ── */
    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = 0.5f + 0.001f * (i % 1000);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    /* ══════════════════════════════════════════════════════════════════════
     * SECTION 1: Query register usage for all kernel variants
     * ══════════════════════════════════════════════════════════════════════
     *
     * cudaFuncGetAttributes returns the actual register count assigned by
     * the compiler. This is what limits occupancy at runtime.
     *
     * NOTE: To see register counts at COMPILE time, use:
     *   nvcc -Xptxas -v -arch=sm_61 register_pressure.cu
     *
     * This prints per-kernel info like:
     *   ptxas info    : Compiling entry function '_Z15kernel_low_regsPKfPfi'
     *   ptxas info    : Used 16 registers, 0 bytes smem, 380 bytes cmem[0]
     *
     * The register count from -Xptxas -v should match cudaFuncGetAttributes.
     */
    printf("─── Register Usage and Occupancy ──────────────────────────────────────────\n\n");

    /* Define kernel info for table output */
    struct KernelInfo {
        const char* name;
        const char* description;
        void* func;   /* we need to cast to query attributes */
    };

    /* Query attributes for each kernel */
    cudaFuncAttributes attr_low, attr_high, attr_capped, attr_medium;
    CUDA_CHECK(cudaFuncGetAttributes(&attr_low,    kernel_low_regs));
    CUDA_CHECK(cudaFuncGetAttributes(&attr_high,   kernel_high_regs));
    CUDA_CHECK(cudaFuncGetAttributes(&attr_capped, kernel_high_regs_capped));
    CUDA_CHECK(cudaFuncGetAttributes(&attr_medium, kernel_high_regs_medium_cap));

    /* Query occupancy for each kernel */
    int blocks_low, blocks_high, blocks_capped, blocks_medium;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_low, kernel_low_regs, blockSize, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_high, kernel_high_regs, blockSize, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_capped, kernel_high_regs_capped, blockSize, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_medium, kernel_high_regs_medium_cap, blockSize, 0));

    printf("  %-28s │ %5s │ %8s │ %7s │ %10s\n",
           "Kernel", "Regs", "Blks/SM", "Occ %%", "Spill Risk");
    printf("  ────────────────────────────┼───────┼──────────┼─────────┼───────────\n");

    auto print_row = [&](const char* name, cudaFuncAttributes& attr,
                         int blocks, const char* spillRisk) {
        float occ = 100.0f * blocks * (blockSize / 32) / (float)maxWarps;
        printf("  %-28s │ %5d │ %8d │ %6.1f%% │ %10s\n",
               name, attr.numRegs, blocks, occ, spillRisk);
    };

    print_row("Low regs (simple)",          attr_low,    blocks_low,    "None");
    print_row("High regs (no cap)",         attr_high,   blocks_high,   "None");
    print_row("High regs (cap for 100%)",   attr_capped, blocks_capped, "HIGH");
    print_row("High regs (cap for 50%)",    attr_medium, blocks_medium, "Medium");

    printf("\n");
    printf("  Register budget analysis (CC 6.1, 65536 regs/SM, blockSize=%d):\n", blockSize);
    printf("    100%% occupancy (8 blocks): 65536 / (8 × 256) = 32 regs/thread max\n");
    printf("     75%% occupancy (6 blocks): 65536 / (6 × 256) = 42 regs/thread max\n");
    printf("     50%% occupancy (4 blocks): 65536 / (4 × 256) = 64 regs/thread max\n");
    printf("     25%% occupancy (2 blocks): 65536 / (2 × 256) = 128 regs/thread max\n\n");

    /* Query and display local memory (spill) usage */
    printf("  Local memory (spill) usage:\n");
    printf("    Low regs:          %5zu bytes/thread\n", attr_low.localSizeBytes);
    printf("    High regs (no cap):%5zu bytes/thread\n", attr_high.localSizeBytes);
    printf("    High regs (100%%):  %5zu bytes/thread  ← spilling to hit 32 regs!\n",
           attr_capped.localSizeBytes);
    printf("    High regs (50%%):   %5zu bytes/thread\n\n", attr_medium.localSizeBytes);

    /* ══════════════════════════════════════════════════════════════════════
     * SECTION 2: Benchmark all variants
     * ══════════════════════════════════════════════════════════════════════
     *
     * The critical question: does higher occupancy (from capping registers)
     * compensate for the cost of register spilling?
     *
     * Expected results:
     *   - kernel_low_regs: fastest (simple, 100% occupancy, no spilling)
     *   - kernel_high_regs: moderate (complex, lower occupancy, no spilling)
     *   - kernel_high_regs_medium_cap: potentially fastest for this workload
     *     (compromise: 50% occupancy, moderate spilling)
     *   - kernel_high_regs_capped: may be SLOWER despite 100% occupancy
     *     (heavy spilling to local memory can overwhelm the occupancy gain)
     */
    printf("─── Performance Benchmark ─────────────────────────────────────────────────\n\n");

    float ms_low    = benchmark_kernel(kernel_low_regs,            d_in, d_out, N, blockSize, warmup, runs);
    float ms_high   = benchmark_kernel(kernel_high_regs,           d_in, d_out, N, blockSize, warmup, runs);
    float ms_capped = benchmark_kernel(kernel_high_regs_capped,    d_in, d_out, N, blockSize, warmup, runs);
    float ms_medium = benchmark_kernel(kernel_high_regs_medium_cap, d_in, d_out, N, blockSize, warmup, runs);

    printf("  %-28s │ %10s │ %10s\n", "Kernel", "Time (ms)", "Rel. Speed");
    printf("  ────────────────────────────┼────────────┼────────────\n");

    /* Use kernel_high_regs (uncapped) as the baseline for relative comparison */
    printf("  %-28s │ %10.3f │ %10s\n", "Low regs (simple)",
           ms_low, "(different kernel)");
    printf("  %-28s │ %10.3f │ %9.2fx\n", "High regs (no cap)",
           ms_high, 1.0f);
    printf("  %-28s │ %10.3f │ %9.2fx\n", "High regs (cap for 100%%)",
           ms_capped, ms_high / ms_capped);
    printf("  %-28s │ %10.3f │ %9.2fx\n", "High regs (cap for 50%%)",
           ms_medium, ms_high / ms_medium);

    printf("\n");

    /* ══════════════════════════════════════════════════════════════════════
     * SECTION 3: Analysis — find the sweet spot
     * ══════════════════════════════════════════════════════════════════════ */
    printf("─── Analysis ──────────────────────────────────────────────────────────────\n\n");

    /* Determine which variant was fastest */
    float times[] = {ms_high, ms_capped, ms_medium};
    const char* names[] = {"No cap (compiler default)", "Capped for 100%", "Capped for 50%"};
    int bestIdx = 0;
    for (int i = 1; i < 3; i++) {
        if (times[i] < times[bestIdx]) bestIdx = i;
    }

    printf("  Fastest variant: %s (%.3f ms)\n\n", names[bestIdx], times[bestIdx]);

    if (bestIdx == 0) {
        printf("  The uncapped version won! This means:\n");
        printf("    - The kernel is compute-bound, not memory-bound\n");
        printf("    - Having more registers (faster per-thread) beat higher occupancy\n");
        printf("    - Register spilling from capping was too expensive\n");
    } else if (bestIdx == 1) {
        printf("  The 100%% occupancy cap won! This means:\n");
        printf("    - The kernel is memory-bound\n");
        printf("    - Higher occupancy (more warp-level parallelism) hides latency\n");
        printf("    - Register spilling cost was acceptable\n");
    } else {
        printf("  The 50%% occupancy cap won! This is the sweet spot:\n");
        printf("    - Enough occupancy to hide most memory latency\n");
        printf("    - Enough registers to avoid excessive spilling\n");
        printf("    - This is a common result in practice\n");
    }

    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────┐\n");
    printf("  │  REGISTER PRESSURE VISUALIZATION                                │\n");
    printf("  │                                                                 │\n");
    printf("  │  Performance                                                    │\n");
    printf("  │  ▲                                                              │\n");
    printf("  │  │            ╱╲                                                │\n");
    printf("  │  │           ╱  ╲           Sweet spot: enough regs for speed,  │\n");
    printf("  │  │          ╱    ╲          enough threads for latency hiding   │\n");
    printf("  │  │         ╱      ╲                                             │\n");
    printf("  │  │        ╱        ╲                                            │\n");
    printf("  │  │  Slow ╱          ╲ Slow                                      │\n");
    printf("  │  │  (spilling)       (low occupancy)                            │\n");
    printf("  │  └────────────────────────────────► Regs/thread                 │\n");
    printf("  │     few regs              many regs                             │\n");
    printf("  │     high occupancy        low occupancy                         │\n");
    printf("  │     heavy spilling        no spilling                           │\n");
    printf("  └──────────────────────────────────────────────────────────────────┘\n");

    printf("\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n");
    printf("  HOW TO INVESTIGATE REGISTER PRESSURE IN YOUR OWN KERNELS\n");
    printf("══════════════════════════════════════════════════════════════════════════════\n\n");
    printf("  1. Compile with verbose PTX info to see register counts:\n");
    printf("       nvcc -Xptxas -v -arch=sm_61 your_kernel.cu\n\n");
    printf("  2. Use cudaFuncGetAttributes() at runtime (as shown above)\n\n");
    printf("  3. Try different __launch_bounds__ and benchmark:\n");
    printf("       __launch_bounds__(256, 8)  → targets 100%% occupancy\n");
    printf("       __launch_bounds__(256, 4)  → targets 50%% occupancy\n");
    printf("       __launch_bounds__(256, 2)  → targets 25%% occupancy\n\n");
    printf("  4. Use -maxrregcount=N for a global cap across all kernels:\n");
    printf("       nvcc -maxrregcount=32 ...  (cap at 32 regs for all kernels)\n");
    printf("     Warning: this affects ALL kernels in the file!\n\n");
    printf("  5. Use NVIDIA's Occupancy Calculator spreadsheet or\n");
    printf("     cudaOccupancyMaxActiveBlocksPerMultiprocessor for precise analysis.\n\n");
    printf("  6. Profile with nsight compute (ncu) or nvprof to see actual spilling:\n");
    printf("       ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_read ./your_app\n\n");

    /* ── Cleanup ── */
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
