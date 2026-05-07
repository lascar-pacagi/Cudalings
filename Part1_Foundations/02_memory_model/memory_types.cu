/*
 * memory_types.cu -- Demonstrating All CUDA Memory Types
 * ======================================================
 *
 * This program showcases every memory type in the CUDA hierarchy:
 *   1. Registers      (local variables -- fastest)
 *   2. Shared memory   (__shared__ -- per-block scratchpad)
 *   3. Constant memory (__constant__ -- cached, broadcast to warp)
 *   4. Global memory   (device pointers -- large but slow)
 *   5. Local memory    (register spill -- same speed as global)
 *
 * We also query the device for memory hierarchy attributes so you
 * can see the actual numbers for your hardware.
 *
 * Compile: nvcc -arch=sm_61 -O2 -o memory_types memory_types.cu
 * Run:     ./memory_types
 *
 * Target: Quadro P4000 (CC 6.1, 14 SMs)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Problem size: 1M elements (big enough to see timing differences)
#define N (1 << 20)   // 1,048,576 elements

// Block size: 256 is our go-to default
#define BLOCK_SIZE 256

// Number of iterations for timing (more = more stable measurements)
#define ITERS 100

// ============================================================================
// ERROR CHECKING MACRO
// ============================================================================
// Wrap every CUDA call in this. If something fails, you want to know WHERE.
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


// ============================================================================
// CONSTANT MEMORY
// ============================================================================
//
// __constant__ declares a variable in constant memory.
//   - Lives in DRAM but cached in a dedicated per-SM constant cache
//   - 64 KB total limit for the entire program
//   - Read-only from GPU, written by CPU via cudaMemcpyToSymbol()
//   - When ALL threads in a warp read the SAME address, the value is
//     broadcast in a single cycle -- extremely fast
//   - If threads read DIFFERENT addresses, accesses are serialized
//     (worst case: 32x slower than broadcast)
//
// Perfect for: filter coefficients, lookup tables, configuration constants
//
__constant__ float d_const_coeff[256];   // constant memory: filter coefficients


// ============================================================================
// KERNEL 1: Global Memory Only (Baseline)
// ============================================================================
//
// This kernel reads from global memory, does a simple computation, and
// writes back to global memory. No optimization -- just raw DRAM access.
//
// Memory access pattern per thread:
//   Read:  d_in[i]  from global memory (~400-800 cycle latency)
//   Write: d_out[i] to global memory
//
// At least the access pattern is coalesced (consecutive threads access
// consecutive addresses), so we get reasonable bandwidth.
//
__global__ void kernel_global_only(const float *d_in, float *d_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Grid-stride loop: each thread processes multiple elements
    for (; i < n; i += stride) {
        // Read from global, compute, write to global.
        // The multiplication simulates some work. Without it, the compiler
        // might optimize away the load entirely.
        float val = d_in[i];
        d_out[i] = val * 2.0f + 1.0f;
    }
}


// ============================================================================
// KERNEL 2: Shared Memory Usage
// ============================================================================
//
// This kernel loads a tile of data into shared memory, processes it there,
// and writes results back to global memory.
//
// The idea:
//   1. Each thread loads ONE element from global memory into shared memory
//   2. __syncthreads() -- wait for all loads to complete
//   3. Each thread reads from shared memory (possibly multiple times)
//   4. Write result back to global memory
//
// For this simple example, shared memory doesn't help much (each element
// is only read once). Shared memory shines when there is DATA REUSE --
// like in stencils, matrix multiply, reductions, etc.
//
// But this demonstrates the syntax and the synchronization pattern.
//
__global__ void kernel_shared_memory(const float *d_in, float *d_out, int n) {
    // Declare shared memory array.
    // __shared__ means: allocate this in the per-block scratchpad.
    // All threads in this block can read/write s_data[].
    // Other blocks CANNOT see this -- each block gets its own copy.
    __shared__ float s_data[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Load from global memory into shared memory.
    // Each thread loads exactly one element.
    // threadIdx.x gives this thread's index WITHIN the block (0 to 255).
    if (i < n) {
        s_data[threadIdx.x] = d_in[i];
    }

    // Step 2: BARRIER -- wait for ALL threads in this block to finish loading.
    //
    // THIS IS CRITICAL. Without __syncthreads(), thread 0 might try to read
    // s_data[1] before thread 1 has written it. That is a race condition.
    //
    // __syncthreads() is a BLOCK-level barrier:
    //   - Every thread in the block must reach this point before ANY thread
    //     can proceed past it.
    //   - It does NOT synchronize across blocks (that is impossible during
    //     kernel execution -- blocks are independent).
    //
    __syncthreads();

    // Step 3: Read from shared memory and compute.
    // Here we simulate a computation that accesses the shared memory element.
    // In a real stencil or reduction, we would access NEIGHBORS, which is
    // where shared memory really pays off.
    if (i < n) {
        float val = s_data[threadIdx.x];   // read from shared (fast! ~5 cycles)
        d_out[i] = val * 2.0f + 1.0f;
    }
}


// ============================================================================
// KERNEL 3: Constant Memory Usage
// ============================================================================
//
// This kernel reads coefficients from constant memory and applies them.
// All threads in a warp read the SAME coefficient (d_const_coeff[0]),
// so the constant cache broadcasts the value efficiently.
//
// If each thread read a DIFFERENT constant memory address, the access
// would be serialized -- 32 separate reads for a warp. So constant
// memory is best when all threads need the same value.
//
__global__ void kernel_constant_memory(const float *d_in, float *d_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Read coefficient from constant memory.
    // All threads in the warp read d_const_coeff[0] at the same time.
    // This is a BROADCAST: one cache read serves all 32 threads.
    float coeff = d_const_coeff[0];   // constant memory read (broadcast)

    for (; i < n; i += stride) {
        float val = d_in[i];
        d_out[i] = val * coeff + d_const_coeff[1];  // another broadcast
    }
}


// ============================================================================
// KERNEL 4: Register-Heavy Computation
// ============================================================================
//
// This kernel does more computation per element, using many local variables.
// Local variables live in REGISTERS -- the fastest storage on the GPU.
//
// The compiler places each local variable in a register. If we use too many,
// variables "spill" to local memory (which is actually DRAM -- slow!).
//
// You can check register usage with: nvcc --ptxas-options=-v
//
__global__ void kernel_register_heavy(const float *d_in, float *d_out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        // All these local variables are in registers.
        // The compiler is smart about reusing registers, but each "live"
        // variable at any point in time needs its own register.
        float val = d_in[i];             // register r0
        float a = val * 1.1f;            // register r1
        float b = a + 0.5f;             // reuses r1 or new register
        float c = b * b;                // register
        float d = sqrtf(c + 1.0f);      // register (sqrtf is a single instruction on GPU!)
        float e = d - a;                // register
        float f = fmaxf(e, 0.0f);       // register (ReLU-like operation)
        float result = f * 0.1f + 0.9f; // register

        d_out[i] = result;
    }
}


// ============================================================================
// PRINT DEVICE MEMORY HIERARCHY INFO
// ============================================================================
//
// This function uses cudaDeviceGetAttribute() to query memory-related
// properties of the GPU. This is how you programmatically discover what
// your hardware can do.
//
void print_memory_hierarchy_info(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("============================================================\n");
    printf("MEMORY HIERARCHY INFO: %s\n", prop.name);
    printf("============================================================\n");
    printf("\n");

    // --- Global Memory ---
    printf("GLOBAL MEMORY (off-chip DRAM):\n");
    printf("  Total:              %zu MB (%zu bytes)\n",
           prop.totalGlobalMem / (1024 * 1024), prop.totalGlobalMem);
    printf("  Memory bus width:   %d bits\n", prop.memoryBusWidth);
    printf("  Memory clock:       %d MHz\n", prop.memoryClockRate / 1000);
    // Calculate theoretical peak bandwidth:
    //   BW = memory_clock * bus_width * 2 (DDR) / 8 (bits to bytes)
    //   memory_clock is in kHz from the API, so divide by 1e6 to get GHz
    double peak_bw_gb = 2.0 * prop.memoryClockRate * 1e3 *
                        (prop.memoryBusWidth / 8) / 1e9;
    printf("  Peak bandwidth:     %.1f GB/s (theoretical)\n", peak_bw_gb);
    printf("\n");

    // --- L2 Cache ---
    printf("L2 CACHE:\n");
    printf("  Size:               %d KB (%d bytes)\n",
           prop.l2CacheSize / 1024, prop.l2CacheSize);
    printf("\n");

    // --- Shared Memory ---
    printf("SHARED MEMORY (on-chip, per block):\n");
    printf("  Per block:          %zu KB (%zu bytes)\n",
           prop.sharedMemPerBlock / 1024, prop.sharedMemPerBlock);
    printf("  Per SM:             %zu KB (%zu bytes)\n",
           prop.sharedMemPerMultiprocessor / 1024,
           prop.sharedMemPerMultiprocessor);
    printf("\n");

    // --- Registers ---
    printf("REGISTERS (on-chip, per SM):\n");
    printf("  Per block:          %d (32-bit registers)\n",
           prop.regsPerBlock);
    printf("  Per SM:             %d (32-bit registers)\n",
           prop.regsPerMultiprocessor);
    printf("  That is:            %d KB of register file per SM\n",
           prop.regsPerMultiprocessor * 4 / 1024);
    printf("\n");

    // --- Constant Memory ---
    printf("CONSTANT MEMORY:\n");
    printf("  Total:              %zu KB (%zu bytes)\n",
           prop.totalConstMem / 1024, prop.totalConstMem);
    printf("\n");

    // --- Other ---
    printf("STREAMING MULTIPROCESSORS (SMs):\n");
    printf("  Count:              %d SMs\n", prop.multiProcessorCount);
    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max threads/block:  %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size:          %d\n", prop.warpSize);
    printf("\n");

    // --- Compute Capability ---
    printf("COMPUTE CAPABILITY:   %d.%d\n", prop.major, prop.minor);
    printf("\n");

    // Occupancy calculation example:
    // If a kernel uses 32 registers per thread:
    //   Max threads per SM = min(regsPerSM / 32, maxThreadsPerSM)
    //                      = min(65536/32, 2048) = min(2048, 2048) = 2048
    //   Occupancy = 2048/2048 = 100%
    //
    // If a kernel uses 64 registers per thread:
    //   Max threads per SM = min(65536/64, 2048) = min(1024, 2048) = 1024
    //   Occupancy = 1024/2048 = 50%
    printf("OCCUPANCY EXAMPLES (based on register usage):\n");
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    int regs_per_sm = prop.regsPerMultiprocessor;
    for (int regs = 16; regs <= 128; regs *= 2) {
        int threads_by_regs = regs_per_sm / regs;
        int actual = (threads_by_regs < max_threads_per_sm)
                     ? threads_by_regs : max_threads_per_sm;
        float occupancy = 100.0f * actual / max_threads_per_sm;
        printf("  %3d regs/thread -> %4d threads/SM -> %.0f%% occupancy\n",
               regs, actual, occupancy);
    }
    printf("\n");
}


// ============================================================================
// TIMING HELPER USING CUDA EVENTS
// ============================================================================
//
// CUDA events give accurate GPU timing. Unlike wall-clock timing (which
// includes CPU overhead and synchronization), CUDA events measure the
// actual GPU execution time.
//
// Usage:
//   cudaEventRecord(start)
//   launch_kernel<<<...>>>(...)
//   cudaEventRecord(stop)
//   cudaEventSynchronize(stop)
//   cudaEventElapsedTime(&ms, start, stop)  // result in milliseconds
//
typedef struct {
    cudaEvent_t start, stop;
} GpuTimer;

void timer_create(GpuTimer *t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
}

void timer_destroy(GpuTimer *t) {
    CUDA_CHECK(cudaEventDestroy(t->start));
    CUDA_CHECK(cudaEventDestroy(t->stop));
}

void timer_start(GpuTimer *t) {
    CUDA_CHECK(cudaEventRecord(t->start, 0));
}

float timer_stop(GpuTimer *t) {
    float ms;
    CUDA_CHECK(cudaEventRecord(t->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    return ms;
}


// ============================================================================
// MAIN
// ============================================================================

int main() {
    // --- Print hardware info ---
    print_memory_hierarchy_info(0);

    // --- Setup ---
    size_t bytes = N * sizeof(float);
    printf("============================================================\n");
    printf("MEMORY TYPE BENCHMARKS (N = %d, %.1f MB)\n", N,
           (float)bytes / (1024 * 1024));
    printf("Each kernel runs %d iterations for stable timing.\n", ITERS);
    printf("============================================================\n\n");

    // Allocate host memory
    float *h_in  = (float *)malloc(bytes);
    float *h_out = (float *)malloc(bytes);

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i * 0.001f;
    }

    // Allocate device memory (global memory -- this is the big DRAM)
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Setup constant memory
    // We store coefficients in constant memory. These will be used by
    // kernel_constant_memory(). Note: cudaMemcpyToSymbol copies from
    // host to the __constant__ variable on the device.
    float h_coeff[256];
    h_coeff[0] = 2.0f;   // multiplication coefficient
    h_coeff[1] = 1.0f;   // addition coefficient
    CUDA_CHECK(cudaMemcpyToSymbol(d_const_coeff, h_coeff, 256 * sizeof(float)));

    // Grid configuration
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Create timer
    GpuTimer timer;
    timer_create(&timer);

    // --- Warmup ---
    // The first kernel launch has overhead (driver initialization, JIT, etc.)
    // Run each kernel once to warm up before timing.
    kernel_global_only<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    kernel_shared_memory<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    kernel_constant_memory<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    kernel_register_heavy<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================================================================
    // Benchmark 1: Global Memory Only
    // =========================================================================
    timer_start(&timer);
    for (int iter = 0; iter < ITERS; iter++) {
        kernel_global_only<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    float ms_global = timer_stop(&timer);

    // Calculate bandwidth:
    // Each element: 1 float read + 1 float write = 8 bytes
    double total_bytes_global = (double)N * 8.0 * ITERS;
    double bw_global = total_bytes_global / (ms_global * 1e-3) / 1e9;

    printf("1. Global memory only:   %7.3f ms total  (%6.2f ms/iter)  BW: %6.1f GB/s\n",
           ms_global, ms_global / ITERS, bw_global);

    // =========================================================================
    // Benchmark 2: Shared Memory
    // =========================================================================
    timer_start(&timer);
    for (int iter = 0; iter < ITERS; iter++) {
        kernel_shared_memory<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    float ms_shared = timer_stop(&timer);

    double total_bytes_shared = (double)N * 8.0 * ITERS;
    double bw_shared = total_bytes_shared / (ms_shared * 1e-3) / 1e9;

    printf("2. Shared memory:        %7.3f ms total  (%6.2f ms/iter)  BW: %6.1f GB/s\n",
           ms_shared, ms_shared / ITERS, bw_shared);

    // =========================================================================
    // Benchmark 3: Constant Memory
    // =========================================================================
    timer_start(&timer);
    for (int iter = 0; iter < ITERS; iter++) {
        kernel_constant_memory<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    float ms_const = timer_stop(&timer);

    double total_bytes_const = (double)N * 8.0 * ITERS;
    double bw_const = total_bytes_const / (ms_const * 1e-3) / 1e9;

    printf("3. Constant memory:      %7.3f ms total  (%6.2f ms/iter)  BW: %6.1f GB/s\n",
           ms_const, ms_const / ITERS, bw_const);

    // =========================================================================
    // Benchmark 4: Register-Heavy
    // =========================================================================
    timer_start(&timer);
    for (int iter = 0; iter < ITERS; iter++) {
        kernel_register_heavy<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    }
    float ms_regs = timer_stop(&timer);

    double total_bytes_regs = (double)N * 8.0 * ITERS;
    double bw_regs = total_bytes_regs / (ms_regs * 1e-3) / 1e9;

    printf("4. Register-heavy:       %7.3f ms total  (%6.2f ms/iter)  BW: %6.1f GB/s\n",
           ms_regs, ms_regs / ITERS, bw_regs);

    // =========================================================================
    // Verify correctness (spot check)
    // =========================================================================
    // Copy results from the last kernel back to CPU and check a few values.
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    printf("\n--- Correctness check (register-heavy kernel) ---\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float val = (float)i * 0.001f;
        float a = val * 1.1f;
        float b = a + 0.5f;
        float c = b * b;
        float d = sqrtf(c + 1.0f);
        float e = d - a;
        float f = fmaxf(e, 0.0f);
        float expected = f * 0.1f + 0.9f;
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            if (errors < 5) {
                printf("  MISMATCH at i=%d: expected=%.6f, got=%.6f\n",
                       i, expected, h_out[i]);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("  PASSED -- all values correct.\n");
    } else {
        printf("  FAILED -- %d mismatches.\n", errors);
    }

    // =========================================================================
    // Analysis
    // =========================================================================
    printf("\n============================================================\n");
    printf("ANALYSIS\n");
    printf("============================================================\n");
    printf("\n");
    printf("For these simple kernels (1 read + 1 write per element), the\n");
    printf("differences between memory types are subtle because the\n");
    printf("bottleneck is always the global memory read/write.\n");
    printf("\n");
    printf("Shared memory shines when there is DATA REUSE (each element\n");
    printf("is read multiple times, like in stencils or matrix multiply).\n");
    printf("See stencil_1d.cu for a dramatic demonstration.\n");
    printf("\n");
    printf("Constant memory shines when ALL threads in a warp read the\n");
    printf("SAME address (broadcast). The coefficient read was essentially\n");
    printf("free -- cached and broadcast to all 32 threads.\n");
    printf("\n");
    printf("The register-heavy kernel does more FLOPs but runs at similar\n");
    printf("bandwidth because it is still memory-bound -- the GPU has\n");
    printf("plenty of ALUs to do the extra math while waiting for memory.\n");
    printf("\n");

    // Cleanup
    timer_destroy(&timer);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    printf("Done!\n");
    return 0;
}

/*
 * =============================================================================
 * KEY TAKEAWAYS:
 * =============================================================================
 *
 * 1. REGISTERS are the fastest storage. Local variables in your kernel live
 *    in registers. The compiler handles this automatically.
 *
 * 2. SHARED MEMORY is explicitly managed by you. It is on-chip and fast
 *    (~5 cycles), but limited (~48 KB per block). Use it for data reuse.
 *
 * 3. CONSTANT MEMORY is great for values that all threads need. It uses a
 *    special cache that can broadcast one value to an entire warp.
 *
 * 4. GLOBAL MEMORY is where your data lives. It is large (8 GB) but slow
 *    (~400-800 cycles). Coalesced access is critical.
 *
 * 5. For simple kernels (low arithmetic intensity), performance is dominated
 *    by global memory bandwidth. Shared memory only helps when you can
 *    reduce the number of global memory accesses through data reuse.
 *
 * 6. Use cudaDeviceGetAttribute() / cudaGetDeviceProperties() to query
 *    hardware limits at runtime. Never hardcode assumptions about GPU specs.
 *
 * =============================================================================
 */
