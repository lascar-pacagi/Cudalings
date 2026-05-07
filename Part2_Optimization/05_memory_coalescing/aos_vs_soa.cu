/*
 * aos_vs_soa.cu -- Array of Structures vs Structure of Arrays on GPU
 * ====================================================================
 *
 * This is one of the most important practical lessons in GPU programming.
 * The way you lay out your data in memory can make a 2-5x performance
 * difference -- with ZERO changes to the algorithm.
 *
 * We simulate a particle system with 7 fields per particle:
 *   position: x, y, z
 *   velocity: vx, vy, vz
 *   mass
 *
 * The update rule is simple: pos += vel * dt (Euler integration).
 * This kernel is purely memory-bound -- the compute is trivial.
 *
 * We compare two layouts:
 *
 * AoS (Array of Structures):
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │ Memory: [x0 y0 z0 vx0 vy0 vz0 m0 | x1 y1 z1 vx1 ... | ...│
 *   │          |----- particle 0 ------| |- particle 1 ----|      │
 *   │                                                             │
 *   │ When a warp reads x for 32 particles:                       │
 *   │   T0->x0  T1->x1  T2->x2  ...  T31->x31                   │
 *   │   But x0 and x1 are 28 bytes apart (stride = 7 floats)!    │
 *   │   32 x values span 32 * 28 = 896 bytes = 7 cache lines     │
 *   │   Efficiency: 1/7 = 14%                                    │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * SoA (Structure of Arrays):
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │ Memory: [x0 x1 x2 ... x_N-1 | y0 y1 y2 ... | z0 z1 ... ]  │
 *   │          |-- all x values --| |-- all y's -|                │
 *   │                                                             │
 *   │ When a warp reads x for 32 particles:                       │
 *   │   T0->x0  T1->x1  T2->x2  ...  T31->x31                   │
 *   │   x0 and x1 are 4 bytes apart (consecutive!)               │
 *   │   32 x values = 128 bytes = 1 cache line                   │
 *   │   Efficiency: 100%                                          │
 *   └──────────────────────────────────────────────────────────────┘
 *
 * CONNECTION TO DEEP LEARNING:
 *   This is exactly the same issue as NCHW vs NHWC tensor layouts:
 *   - NCHW = channels first = SoA-like (all R pixels, then all G, then all B)
 *   - NHWC = channels last  = AoS-like (R,G,B per pixel)
 *   cuDNN supports both, but performance differs based on the operation.
 *   Modern tensor cores on Volta+ prefer NHWC for mixed precision.
 *
 * Compile: nvcc -arch=sm_61 -O2 -lineinfo -o aos_vs_soa aos_vs_soa.cu
 * Run:     ./aos_vs_soa
 *
 * Target: Quadro P4200 (CC 6.1, 18 SMs)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

#define NUM_PARTICLES (4 * 1024 * 1024)   // 4M particles
#define BLOCK_SIZE    256
#define WARMUP_ITERS  5
#define BENCH_ITERS   20

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
// DATA STRUCTURES
// ============================================================================

/*
 * AoS: Array of Structures
 * -------------------------
 * Each particle is a contiguous struct in memory.
 * Intuitive for CPU code, but terrible for GPU coalescing.
 *
 * sizeof(ParticleAoS) = 7 * 4 = 28 bytes
 *
 * Memory layout for 4 particles:
 * [x0 y0 z0 vx0 vy0 vz0 m0 | x1 y1 z1 vx1 vy1 vz1 m1 | x2 ... | x3 ...]
 * |<------ 28 bytes ------->|<------ 28 bytes --------->|
 *
 * Problem: When all threads in a warp want to read the 'x' field:
 *   T0 reads at offset 0, T1 at offset 28, T2 at offset 56, ...
 *   This is a stride-7 access pattern!
 */
struct ParticleAoS {
    float x, y, z;        // position
    float vx, vy, vz;     // velocity
    float mass;            // mass
};

/*
 * SoA: Structure of Arrays
 * -------------------------
 * Each field is stored in its own contiguous array.
 * Perfect for GPU coalescing.
 *
 * Memory layout:
 *   x  array: [x0  x1  x2  x3  ... x_{N-1}]    (contiguous)
 *   y  array: [y0  y1  y2  y3  ... y_{N-1}]    (contiguous)
 *   z  array: [z0  z1  z2  z3  ... z_{N-1}]    (contiguous)
 *   vx array: [vx0 vx1 vx2 vx3 ...]            (contiguous)
 *   vy array: [vy0 vy1 vy2 vy3 ...]            (contiguous)
 *   vz array: [vz0 vz1 vz2 vz3 ...]            (contiguous)
 *   mass:     [m0  m1  m2  m3  ...]             (contiguous)
 *
 * When threads read 'x': T0->x0, T1->x1, T2->x2, ... = consecutive!
 */
struct ParticlesSoA {
    float *x, *y, *z;        // position arrays
    float *vx, *vy, *vz;     // velocity arrays
    float *mass;              // mass array
};


// ============================================================================
// KERNEL: AoS POSITION UPDATE
// ============================================================================
/*
 * Update particle positions using AoS layout.
 *
 * What happens in memory when a warp executes this:
 *
 *   Warp reads particles[tid].x:
 *     T0  -> particles[0].x   = offset 0
 *     T1  -> particles[1].x   = offset 28
 *     T2  -> particles[2].x   = offset 56
 *     T3  -> particles[3].x   = offset 84
 *     ...
 *     T31 -> particles[31].x  = offset 868
 *
 *     These 32 addresses span 868 + 4 = 872 bytes = 7 cache lines!
 *     But we only needed 32 * 4 = 128 bytes.
 *     Wasted bandwidth: 7x (6 out of 7 cache lines have unused data).
 *
 *   Then the warp reads particles[tid].vx:
 *     T0  -> particles[0].vx  = offset 12
 *     T1  -> particles[1].vx  = offset 40
 *     ...
 *     Same problem: 7 cache line reads for 128 bytes of useful data.
 *
 *   In total, reading x, y, z, vx, vy, vz = 6 field reads
 *   Each needs 7 cache line transactions instead of 1.
 *   Overhead: 6 * 7 = 42 transactions instead of 6 transactions.
 */
__global__ void kernel_update_aos(ParticleAoS *particles, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        // Each of these reads is strided by sizeof(ParticleAoS) = 28 bytes
        // across the warp. This means 7 cache lines per field access.
        particles[i].x += particles[i].vx * dt;   // read vx (strided), read x (strided), write x (strided)
        particles[i].y += particles[i].vy * dt;   // same pattern
        particles[i].z += particles[i].vz * dt;   // same pattern
    }
}


// ============================================================================
// KERNEL: SoA POSITION UPDATE
// ============================================================================
/*
 * Update particle positions using SoA layout.
 *
 * What happens in memory when a warp executes this:
 *
 *   Warp reads x[tid]:
 *     T0  -> x[0]   = address &x[0]
 *     T1  -> x[1]   = address &x[0] + 4
 *     T2  -> x[2]   = address &x[0] + 8
 *     ...
 *     T31 -> x[31]  = address &x[0] + 124
 *
 *     32 addresses in 128 consecutive bytes = 1 cache line transaction!
 *     Perfect coalescing. 100% efficiency.
 *
 *   Same for vx[tid]: 1 transaction.
 *   Same for writing x[tid]: 1 transaction.
 *
 *   Total: 6 reads + 3 writes = 9 transactions (one per field access)
 *   Compare to AoS: 42 reads + 21 writes = 63 transactions
 *   That's a 7x reduction in memory transactions!
 */
__global__ void kernel_update_soa(float *x, float *y, float *z,
                                   float *vx, float *vy, float *vz,
                                   float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; i < n; i += stride) {
        // Each array access is consecutive across the warp = coalesced!
        x[i] += vx[i] * dt;   // read vx[i] (coalesced), read x[i] (coalesced), write x[i] (coalesced)
        y[i] += vy[i] * dt;   // same: all coalesced
        z[i] += vz[i] * dt;   // same: all coalesced
    }
}


// ============================================================================
// HELPER: Initialize particles with test data
// ============================================================================

void init_particles_aos(ParticleAoS *h_particles, int n) {
    for (int i = 0; i < n; i++) {
        h_particles[i].x  = (float)i * 0.001f;
        h_particles[i].y  = (float)i * 0.002f;
        h_particles[i].z  = (float)i * 0.003f;
        h_particles[i].vx = 1.0f;
        h_particles[i].vy = 2.0f;
        h_particles[i].vz = 3.0f;
        h_particles[i].mass = 1.0f;
    }
}


// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("AoS vs SoA: Memory Layout Impact on GPU Performance\n");
    printf("=====================================================\n");
    printf("Particles: %d (%.1f million)\n", NUM_PARTICLES,
           NUM_PARTICLES / 1e6);
    printf("Fields per particle: 7 floats (28 bytes)\n");
    printf("Operation: pos += vel * dt (3 field updates)\n\n");

    const float dt = 0.01f;
    int grid_size = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float elapsed_ms;

    // ===================================================================
    // PART 1: AoS BENCHMARK
    // ===================================================================
    printf("--- AoS (Array of Structures) ---\n");

    /*
     * MEMORY LAYOUT DIAGRAM:
     *
     *   Address: 0    4    8    12   16   20   24   28   32   36   ...
     *   Data:    [x0 | y0 | z0 | vx0| vy0| vz0| m0 | x1 | y1 | z1 | ...]
     *            |<------ particle 0 (28 B) ------>| |<-- particle 1 ...
     *
     *   When warp reads x:  addr 0, 28, 56, 84, ... (stride = 7 floats)
     *   When warp reads vx: addr 12, 40, 68, 96, ... (stride = 7 floats)
     */

    // Allocate and initialize AoS on host
    size_t aos_size = NUM_PARTICLES * sizeof(ParticleAoS);
    ParticleAoS *h_particles = (ParticleAoS *)malloc(aos_size);
    init_particles_aos(h_particles, NUM_PARTICLES);

    printf("  Total memory: %.1f MB\n", aos_size / (1024.0f * 1024.0f));

    // Allocate on device and copy
    ParticleAoS *d_particles;
    CUDA_CHECK(cudaMalloc(&d_particles, aos_size));
    CUDA_CHECK(cudaMemcpy(d_particles, h_particles, aos_size,
                          cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_update_aos<<<grid_size, BLOCK_SIZE>>>(d_particles, dt,
                                                      NUM_PARTICLES);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        kernel_update_aos<<<grid_size, BLOCK_SIZE>>>(d_particles, dt,
                                                      NUM_PARTICLES);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float aos_time = elapsed_ms / BENCH_ITERS;

    /*
     * Effective bandwidth calculation:
     *   The kernel reads 6 floats (x, y, z, vx, vy, vz) and writes 3 (x, y, z).
     *   That's (6 reads + 3 writes) * 4 bytes = 36 bytes per particle.
     *   This is the MINIMUM data the kernel needs.
     *   The ACTUAL data fetched is much higher due to strided access.
     */
    float aos_bytes = (float)NUM_PARTICLES * 36.0f;   // 9 field accesses * 4 bytes
    float aos_bw = (aos_bytes * BENCH_ITERS) / (elapsed_ms / 1000.0f) / 1e9;

    printf("  Time per iteration: %.3f ms\n", aos_time);
    printf("  Effective bandwidth: %.1f GB/s\n", aos_bw);

    // ===================================================================
    // PART 2: SoA BENCHMARK
    // ===================================================================
    printf("\n--- SoA (Structure of Arrays) ---\n");

    /*
     * MEMORY LAYOUT DIAGRAM:
     *
     *   x  array: [x0  | x1  | x2  | x3  | ... | x_{N-1}]  <- contiguous
     *   y  array: [y0  | y1  | y2  | y3  | ... | y_{N-1}]  <- contiguous
     *   z  array: [z0  | z1  | z2  | z3  | ... | z_{N-1}]  <- contiguous
     *   vx array: [vx0 | vx1 | vx2 | vx3 | ... | vx_{N-1}] <- contiguous
     *   vy array: [vy0 | vy1 | vy2 | vy3 | ... | vy_{N-1}] <- contiguous
     *   vz array: [vz0 | vz1 | vz2 | vz3 | ... | vz_{N-1}] <- contiguous
     *   mass:     [m0  | m1  | m2  | m3  | ... | m_{N-1}]  <- contiguous
     *
     *   When warp reads x:  addr &x[0], &x[0]+4, &x[0]+8, ... (stride = 1)
     *   Perfect coalescing: 1 cache line per 32-thread access.
     */

    // Allocate SoA arrays on device
    size_t array_size = NUM_PARTICLES * sizeof(float);
    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_mass;

    CUDA_CHECK(cudaMalloc(&d_x,    array_size));
    CUDA_CHECK(cudaMalloc(&d_y,    array_size));
    CUDA_CHECK(cudaMalloc(&d_z,    array_size));
    CUDA_CHECK(cudaMalloc(&d_vx,   array_size));
    CUDA_CHECK(cudaMalloc(&d_vy,   array_size));
    CUDA_CHECK(cudaMalloc(&d_vz,   array_size));
    CUDA_CHECK(cudaMalloc(&d_mass, array_size));

    printf("  Total memory: %.1f MB (same data, different layout)\n",
           7 * array_size / (1024.0f * 1024.0f));

    // Initialize SoA from the AoS data (same values, different layout)
    float *h_x  = (float *)malloc(array_size);
    float *h_y  = (float *)malloc(array_size);
    float *h_z  = (float *)malloc(array_size);
    float *h_vx = (float *)malloc(array_size);
    float *h_vy = (float *)malloc(array_size);
    float *h_vz = (float *)malloc(array_size);

    for (int i = 0; i < NUM_PARTICLES; i++) {
        h_x[i]  = h_particles[i].x;
        h_y[i]  = h_particles[i].y;
        h_z[i]  = h_particles[i].z;
        h_vx[i] = h_particles[i].vx;
        h_vy[i] = h_particles[i].vy;
        h_vz[i] = h_particles[i].vz;
    }

    CUDA_CHECK(cudaMemcpy(d_x,  h_x,  array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y,  h_y,  array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z,  h_z,  array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz, array_size, cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        kernel_update_soa<<<grid_size, BLOCK_SIZE>>>(d_x, d_y, d_z,
                                                      d_vx, d_vy, d_vz,
                                                      dt, NUM_PARTICLES);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCH_ITERS; i++) {
        kernel_update_soa<<<grid_size, BLOCK_SIZE>>>(d_x, d_y, d_z,
                                                      d_vx, d_vy, d_vz,
                                                      dt, NUM_PARTICLES);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    float soa_time = elapsed_ms / BENCH_ITERS;
    float soa_bytes = (float)NUM_PARTICLES * 36.0f;  // Same logical bytes
    float soa_bw = (soa_bytes * BENCH_ITERS) / (elapsed_ms / 1000.0f) / 1e9;

    printf("  Time per iteration: %.3f ms\n", soa_time);
    printf("  Effective bandwidth: %.1f GB/s\n", soa_bw);

    // ===================================================================
    // COMPARISON
    // ===================================================================
    printf("\n");
    printf("=====================================================\n");
    printf("  COMPARISON: AoS vs SoA\n");
    printf("=====================================================\n");
    printf("                    AoS          SoA\n");
    printf("  Time (ms):       %8.3f     %8.3f\n", aos_time, soa_time);
    printf("  Bandwidth:       %6.1f GB/s  %6.1f GB/s\n", aos_bw, soa_bw);
    printf("  Speedup:         1.00x        %.2fx\n", aos_time / soa_time);
    printf("-----------------------------------------------------\n");

    // ASCII bar chart
    float max_bw = fmax(aos_bw, soa_bw);
    int bar_max = 40;

    int aos_bar = (int)(aos_bw / max_bw * bar_max);
    int soa_bar = (int)(soa_bw / max_bw * bar_max);
    if (aos_bar < 1) aos_bar = 1;
    if (soa_bar < 1) soa_bar = 1;

    printf("  AoS |");
    for (int i = 0; i < aos_bar; i++) printf("#");
    printf(" %.1f GB/s\n", aos_bw);

    printf("  SoA |");
    for (int i = 0; i < soa_bar; i++) printf("#");
    printf(" %.1f GB/s\n", soa_bw);

    printf("=====================================================\n\n");

    printf("WHY THIS MATTERS:\n");
    printf("  - AoS has stride-%lu access (%lu-byte struct = %lu floats).\n",
           sizeof(ParticleAoS) / sizeof(float),
           sizeof(ParticleAoS),
           sizeof(ParticleAoS) / sizeof(float));
    printf("  - Each warp field read touches %lu cache lines instead of 1.\n",
           sizeof(ParticleAoS) / sizeof(float));
    printf("  - SoA gives perfectly coalesced access: 1 cache line per field.\n");
    printf("  - The algorithm is IDENTICAL; only the data layout changed.\n\n");

    printf("DEEP LEARNING CONNECTION:\n");
    printf("  - NCHW layout (channels first) = SoA-like (contiguous channels)\n");
    printf("  - NHWC layout (channels last)  = AoS-like (interleaved channels)\n");
    printf("  - For standard convolutions, NCHW often coalesces better.\n");
    printf("  - For tensor core ops (Volta+), NHWC is required.\n");
    printf("  - cuDNN picks the best layout per operation.\n\n");

    // ===================================================================
    // CORRECTNESS CHECK
    // ===================================================================
    /*
     * Verify both kernels produce the same results.
     * Copy results back and compare.
     */
    CUDA_CHECK(cudaMemcpy(h_x, d_x, array_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_particles, d_particles, aos_size,
                          cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float diff = fabsf(h_x[i] - h_particles[i].x);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Correctness check: max |SoA.x - AoS.x| = %e %s\n",
           max_diff, max_diff < 1e-3f ? "(PASS)" : "(FAIL)");

    // ===================================================================
    // CLEANUP
    // ===================================================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_particles));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFree(d_mass));

    free(h_particles);
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_vx);
    free(h_vy);
    free(h_vz);

    return 0;
}
