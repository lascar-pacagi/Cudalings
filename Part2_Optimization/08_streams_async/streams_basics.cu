// ============================================================================
// Chapter 08: CUDA Streams — Basics
// ============================================================================
//
// This file demonstrates the fundamental difference between using the
// default stream (everything serial) and using multiple streams (overlapped
// execution).
//
// We perform a simple operation:
//   1. Copy data Host → Device
//   2. Run a kernel on it
//   3. Copy results Device → Host
//
// First we do this in the default stream (serial), then we split the data
// into chunks and process each chunk in its own stream (parallel).
//
// The key insight: with multiple streams, the GPU can overlap data transfers
// with kernel execution, dramatically reducing total wall-clock time.
//
// IMPORTANT: Async transfers require PINNED (page-locked) host memory!
// Using regular malloc/new memory with cudaMemcpyAsync will silently
// fall back to synchronous behavior and you'll get zero overlap.
//
// Target: Quadro P4200 (CC 6.1, 18 SMs, 1 copy engine)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================================
// Error checking macro
// ============================================================================
// Wraps every CUDA call to catch errors immediately. In production code,
// you might use a more sophisticated error handler, but for learning this
// is essential — silent CUDA errors are a debugging nightmare.
// ============================================================================
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ============================================================================
// Kernel: simple element-wise computation
// ============================================================================
// We intentionally make this kernel do some work (a loop) so that it takes
// a non-trivial amount of time. This makes the timing differences between
// serial and multi-stream execution clearly visible.
//
// Each thread processes one element:
//   output[i] = sin(input[i]) * cos(input[i]) * ... (repeated ITERS times)
//
// The ITERS parameter controls how compute-heavy the kernel is.
// ============================================================================
__global__ void process_kernel(const float *input, float *output,
                               int n, int iters) {
    // Global thread index — each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check: we launch enough threads to cover 'n', but the last
    // block may have extra threads beyond 'n'
    if (idx >= n) return;

    // Do some compute work. The loop makes this kernel take measurable time.
    // Without the loop, the kernel would be so fast that transfers dominate
    // and we wouldn't see any meaningful overlap.
    float val = input[idx];
    for (int i = 0; i < iters; i++) {
        val = sinf(val) * cosf(val) + 0.001f;
    }
    output[idx] = val;
}

// ============================================================================
// Helper: print a simple timeline-like summary
// ============================================================================
// This function prints a visual representation of how long each phase took.
// Each '#' represents a unit of time, making it easy to see relative durations.
// ============================================================================
void print_bar(const char *label, float ms, float scale) {
    // Each '#' represents 'scale' milliseconds
    int len = (int)(ms / scale + 0.5f);
    if (len < 1) len = 1;
    if (len > 60) len = 60;

    printf("  %-12s [", label);
    for (int i = 0; i < len; i++) printf("#");
    for (int i = len; i < 60; i++) printf(" ");
    printf("] %7.3f ms\n", ms);
}

// ============================================================================
// Version 1: Serial execution in the default stream
// ============================================================================
// All operations go into stream 0 (the default stream). They execute strictly
// in order: H2D finishes, then kernel runs, then D2H finishes.
//
// Timeline:
//   ┌──── H2D ────┐┌────── Kernel ──────┐┌──── D2H ────┐
//   └──────────────┘└───────────────────┘└──────────────┘
//   ← total = H2D + Kernel + D2H (no overlap) →
//
// This is what happens if you never create any streams. Easy to write,
// but leaves performance on the table.
// ============================================================================
float run_serial(const float *h_input, float *h_output,
                 int N, int iters) {
    // ---- Allocate device memory ----
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    // ---- Create timing events ----
    // CUDA events record GPU timestamps. We place them around operations
    // to measure how long each phase takes on the GPU timeline.
    cudaEvent_t ev_start, ev_h2d, ev_kernel, ev_d2h;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d));
    CUDA_CHECK(cudaEventCreate(&ev_kernel));
    CUDA_CHECK(cudaEventCreate(&ev_d2h));

    // ---- Kernel launch parameters ----
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // ---- Record start, then execute everything in default stream ----
    CUDA_CHECK(cudaEventRecord(ev_start));  // goes into default stream (0)

    // Phase 1: Copy input from host to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_h2d));    // mark end of H2D

    // Phase 2: Run the compute kernel
    process_kernel<<<blocks, threads>>>(d_input, d_output, N, iters);
    CUDA_CHECK(cudaEventRecord(ev_kernel)); // mark end of kernel

    // Phase 3: Copy results from device to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h));    // mark end of D2H

    // ---- Wait for everything to finish ----
    CUDA_CHECK(cudaEventSynchronize(ev_d2h));

    // ---- Compute elapsed times ----
    float t_h2d, t_kernel, t_d2h, t_total;
    CUDA_CHECK(cudaEventElapsedTime(&t_h2d,    ev_start,  ev_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&t_kernel,  ev_h2d,    ev_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&t_d2h,    ev_kernel,  ev_d2h));
    CUDA_CHECK(cudaEventElapsedTime(&t_total,  ev_start,   ev_d2h));

    // ---- Print results ----
    printf("\n  SERIAL (default stream):\n");
    float scale = t_total / 50.0f;  // scale bars to ~50 chars max
    if (scale < 0.01f) scale = 0.01f;
    print_bar("H2D",     t_h2d,    scale);
    print_bar("Kernel",  t_kernel, scale);
    print_bar("D2H",     t_d2h,    scale);
    printf("  %-12s %53s %7.3f ms\n", "TOTAL", "", t_total);

    // ---- Cleanup ----
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_h2d));
    CUDA_CHECK(cudaEventDestroy(ev_kernel));
    CUDA_CHECK(cudaEventDestroy(ev_d2h));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return t_total;
}

// ============================================================================
// Version 2: Multi-stream execution with overlapped transfers and compute
// ============================================================================
// We split the data into NUM_STREAMS chunks. Each chunk gets its own stream.
// Operations within a stream are still ordered, but operations across streams
// can overlap.
//
// We use DEPTH-FIRST issue order: for each stream, we issue H2D, kernel,
// D2H before moving to the next stream. The GPU hardware reorders to find
// overlap opportunities.
//
// Timeline (idealized, 4 streams, 1 copy engine):
//
//   Copy Engine: ┌H2D0┐┌H2D1┐┌D2H0┐┌H2D2┐┌D2H1┐┌H2D3┐┌D2H2┐┌D2H3┐
//   SMs:               ┌─K0──┐┌─K1──┐┌─K2──┐┌─K3──┐
//
// The kernel for stream 0 runs while H2D for stream 1 is transferring.
// Much better than serial!
//
// CRITICAL: h_input and h_output MUST be pinned memory for cudaMemcpyAsync
// to work asynchronously. If they're pageable, the Async call blocks anyway.
// ============================================================================
float run_multistream(const float *h_input, float *h_output,
                      int N, int iters, int num_streams) {
    // ---- Allocate device memory ----
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    // ---- Create streams ----
    // cudaStreamCreate allocates a new stream. Each stream is independent.
    // Operations issued to different streams can execute concurrently.
    cudaStream_t *streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // ---- Create timing events ----
    // We only need a start and end event for the total time measurement.
    // Individual per-stream timing would require 2 events per stream per phase
    // (that's what event_sync.cu demonstrates).
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    // ---- Compute chunk sizes ----
    // We divide N elements evenly across num_streams. The last chunk may be
    // slightly larger or smaller due to integer division.
    int chunk_size = (N + num_streams - 1) / num_streams;  // ceiling division
    int threads = 256;

    // ---- Record start event ----
    CUDA_CHECK(cudaEventRecord(ev_start));

    // ---- Issue work in DEPTH-FIRST order ----
    // For each stream: issue H2D, then kernel, then D2H.
    // This gives the GPU scheduler the best opportunity to find overlap.
    //
    // Why depth-first? Because the GPU's internal scheduler sees that:
    //   - H2D(stream 1) is independent of Kernel(stream 0)
    //   - So it can run them concurrently (one on copy engine, one on SMs)
    //
    // If we did breadth-first (all H2D, then all kernels, then all D2H),
    // the GPU would have less flexibility to overlap.
    for (int i = 0; i < num_streams; i++) {
        // Calculate offset and size for this chunk
        int offset = i * chunk_size;
        int size   = chunk_size;

        // Handle the last chunk: it may be smaller than chunk_size
        if (offset + size > N) size = N - offset;
        if (size <= 0) continue;  // edge case: more streams than needed

        int blocks = (size + threads - 1) / threads;

        // Phase 1: Async H2D copy for this chunk
        // The '+ offset' arithmetic gives us the pointer to this chunk's
        // portion of the array, both on host and device.
        CUDA_CHECK(cudaMemcpyAsync(d_input + offset,       // dst (device)
                                   h_input + offset,        // src (host)
                                   size * sizeof(float),    // size
                                   cudaMemcpyHostToDevice,
                                   streams[i]));            // which stream

        // Phase 2: Launch kernel for this chunk
        // The kernel only processes elements [offset, offset+size).
        // We pass d_input+offset so the kernel sees its chunk starting at [0].
        process_kernel<<<blocks, threads, 0, streams[i]>>>(
            d_input + offset, d_output + offset, size, iters);

        // Phase 3: Async D2H copy for this chunk
        CUDA_CHECK(cudaMemcpyAsync(h_output + offset,      // dst (host)
                                   d_output + offset,       // src (device)
                                   size * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   streams[i]));
    }

    // ---- Record end event and synchronize ----
    // The end event goes into the default stream. Since the default stream
    // blocks on all other streams, this event won't complete until all
    // streams have finished their D2H copies.
    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));

    // ---- Compute total time ----
    float t_total;
    CUDA_CHECK(cudaEventElapsedTime(&t_total, ev_start, ev_end));

    printf("\n  MULTI-STREAM (%d streams):\n", num_streams);
    printf("  %-12s %53s %7.3f ms\n", "TOTAL", "", t_total);

    // ---- Cleanup ----
    // Always destroy streams and events to avoid resource leaks.
    // This is especially important in long-running applications.
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return t_total;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Chapter 08: CUDA Streams Basics\n");
    printf("  Serial vs Multi-Stream Execution\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    // ---- Configuration ----
    // N must be large enough that transfers take non-trivial time.
    // ITERS controls how much compute the kernel does per element.
    // Together they determine the balance between transfer and compute time.
    const int N     = 1 << 22;   // ~4 million elements (~16 MB of float data)
    const int ITERS = 200;       // iterations per element in the kernel

    printf("\n  Problem size: N = %d (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("  Kernel iterations per element: %d\n", ITERS);

    // ========================================================================
    // CRITICAL: Allocate PINNED memory on the host
    // ========================================================================
    // Without pinned memory, cudaMemcpyAsync will block and we'll get no
    // overlap at all. This is the #1 mistake people make with CUDA streams.
    //
    // cudaMallocHost allocates page-locked memory that the GPU can DMA
    // directly to/from without going through a staging buffer.
    // ========================================================================
    float *h_input, *h_output;
    CUDA_CHECK(cudaMallocHost(&h_input,  N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_output, N * sizeof(float)));

    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 1000) * 0.001f;
    }

    // ---- Warmup run (not timed) ----
    // The first CUDA operation on a device triggers context initialization,
    // which can take hundreds of milliseconds. We do a warmup to avoid
    // including that in our timing.
    {
        float *d_tmp;
        CUDA_CHECK(cudaMalloc(&d_tmp, 1024));
        CUDA_CHECK(cudaFree(d_tmp));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("\n────────────────────────────────────────────────────────────\n");

    // ---- Run serial version ----
    float t_serial = run_serial(h_input, h_output, N, ITERS);

    printf("\n────────────────────────────────────────────────────────────\n");

    // ---- Run multi-stream versions with different stream counts ----
    // We test several stream counts to show how increasing the number of
    // streams improves overlap up to a point, after which more streams
    // don't help (because the pipeline is already fully utilized).
    int stream_counts[] = {2, 4, 8, 16};
    for (int s = 0; s < 4; s++) {
        float t_ms = run_multistream(h_input, h_output, N, ITERS,
                                     stream_counts[s]);
        float speedup = t_serial / t_ms;
        printf("  Speedup vs serial: %.2fx\n", speedup);
        printf("\n────────────────────────────────────────────────────────────\n");
    }

    // ---- Verify correctness ----
    // Run serial version one more time and compare outputs to multi-stream.
    // They should produce identical results since we're doing the same
    // computation, just in a different order.
    float *h_verify;
    CUDA_CHECK(cudaMallocHost(&h_verify, N * sizeof(float)));
    run_serial(h_input, h_verify, N, ITERS);

    // Re-run multi-stream to get h_output populated
    run_multistream(h_input, h_output, N, ITERS, 4);

    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_output[i] - h_verify[i]) > 1e-5f) {
            mismatches++;
            if (mismatches <= 5) {
                printf("  MISMATCH at i=%d: serial=%.6f  multi=%.6f\n",
                       i, h_verify[i], h_output[i]);
            }
        }
    }
    printf("\n  Verification: %s (%d mismatches out of %d)\n",
           mismatches == 0 ? "PASS" : "FAIL", mismatches, N);

    // ---- Cleanup ----
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaFreeHost(h_verify));

    printf("\n═══════════════════════════════════════════════════════════════\n");
    return 0;
}
