// ============================================================================
// Chapter 08: Overlapping Data Transfer and Compute
// ============================================================================
//
// This file demonstrates the core benefit of CUDA streams: overlapping
// data transfers with kernel computation to hide transfer latency.
//
// The idea:
//   - Split a large array into N chunks
//   - Create N streams
//   - For each stream i: H2D(chunk_i) → Kernel(chunk_i) → D2H(chunk_i)
//   - The GPU overlaps transfers in one stream with compute in another
//
// We measure total time for the serial case vs the pipelined case and
// show the speedup.
//
// ============================================================================
// PIPELINE DIAGRAM (4 streams, 1 copy engine like the P4200)
// ============================================================================
//
//   Time ──────────────────────────────────────────────────────────→
//
//   SERIAL (default stream):
//   ┌───H2D───┐┌────────Kernel────────┐┌───D2H───┐
//   │  all N  │ │       all N         │ │  all N  │
//   └─────────┘ └─────────────────────┘ └─────────┘
//   Total = T_h2d + T_kernel + T_d2h
//
//   PIPELINED (4 streams, depth-first):
//
//   Copy Engine: ┌H2D0┐┌H2D1┐┌D2H0┐┌H2D2┐┌D2H1┐┌H2D3┐┌D2H2┐┌D2H3┐
//   SMs:               ┌─K0──┐┌─K1──┐┌─K2──┐┌─K3──┐
//
//   With 1 copy engine, H2D and D2H cannot overlap each other, but BOTH
//   can overlap with kernel execution on the SMs.
//
//   Total ≈ T_h2d/N + T_kernel + T_d2h/N  +  small scheduling overhead
//   (In the ideal case where compute dominates transfers)
//
//   The more compute-heavy the kernel relative to transfers, the more
//   the transfer time is "hidden" behind compute.
//
// ============================================================================
//
// Target: Quadro P4200 (CC 6.1, 18 SMs, 1 copy engine)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================================
// Error checking macro
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
// Kernel: vector scaling with iterative computation
// ============================================================================
// output[i] = f(input[i])  where f is an iterative computation.
// The 'iters' parameter controls compute intensity. Higher iters means
// more time spent computing, which gives more opportunity to hide transfers.
//
// This mimics a real-world scenario where you process chunks of a large
// dataset — each chunk involves some transfer cost and some compute cost.
// ============================================================================
__global__ void scale_kernel(const float *input, float *output,
                             int n, float scale, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Do iterative work to simulate a compute-heavy kernel
    float val = input[idx] * scale;
    for (int i = 0; i < iters; i++) {
        val = val * 0.999f + 0.001f;  // simple iterative operation
    }
    output[idx] = val;
}

// ============================================================================
// Serial version: everything in the default stream
// ============================================================================
// This is the baseline. All data is transferred, then the kernel runs on
// all data, then all results are copied back. No overlap whatsoever.
// ============================================================================
float run_serial(const float *h_in, float *h_out, float *d_in, float *d_out,
                 int N, int iters) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start));

    // All three phases go into the default stream — completely serial
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float),
                          cudaMemcpyHostToDevice));
    scale_kernel<<<blocks, threads>>>(d_in, d_out, N, 2.0f, iters);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

// ============================================================================
// Pipelined version: N streams with overlapped transfers and compute
// ============================================================================
// This is where the magic happens. By splitting data into chunks and
// assigning each to a stream, we create a pipeline:
//
//   Stream 0: H2D(chunk0) → Kernel(chunk0) → D2H(chunk0)
//   Stream 1: H2D(chunk1) → Kernel(chunk1) → D2H(chunk1)
//   Stream 2: H2D(chunk2) → Kernel(chunk2) → D2H(chunk2)
//   ...
//
// The GPU sees that H2D(chunk1) is independent of Kernel(chunk0) and
// runs them concurrently — the copy engine handles the transfer while
// the SMs run the kernel.
//
// IMPORTANT: We use DEPTH-FIRST issue order (all ops for stream 0, then
// all ops for stream 1, etc.) because this gives the GPU scheduler the
// best chance to find overlap opportunities.
// ============================================================================
float run_pipelined(const float *h_in, float *h_out,
                    float *d_in, float *d_out,
                    int N, int iters, int num_streams) {
    // ---- Create streams ----
    cudaStream_t *streams = new cudaStream_t[num_streams];
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int chunk = (N + num_streams - 1) / num_streams;
    int threads = 256;

    CUDA_CHECK(cudaEventRecord(start));

    // ---- Depth-first issue order ----
    // For each stream: issue all three phases before moving to the next.
    // The GPU hardware will reorder independent operations across streams.
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk;
        int size   = chunk;
        if (offset + size > N) size = N - offset;
        if (size <= 0) continue;

        int blocks = (size + threads - 1) / threads;

        // Phase 1: Async H2D for this chunk
        // This is non-blocking to the CPU — the call returns immediately
        // and the transfer happens in the background on the copy engine.
        CUDA_CHECK(cudaMemcpyAsync(d_in + offset, h_in + offset,
                                   size * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[i]));

        // Phase 2: Kernel for this chunk
        // This kernel launch is also non-blocking. The kernel will start
        // on the SMs after H2D for this stream completes (stream ordering).
        // But it can overlap with H2D/D2H operations in OTHER streams.
        scale_kernel<<<blocks, threads, 0, streams[i]>>>(
            d_in + offset, d_out + offset, size, 2.0f, iters);

        // Phase 3: Async D2H for this chunk
        // Won't start until this stream's kernel completes (stream ordering).
        // But can overlap with operations in other streams.
        CUDA_CHECK(cudaMemcpyAsync(h_out + offset, d_out + offset,
                                   size * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // ---- Cleanup ----
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    delete[] streams;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Chapter 08: Overlap Transfer and Compute\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    // ---- Query device capabilities ----
    // We check the asyncEngineCount to know how many copy engines we have.
    // The P4200 has 1, meaning H2D and D2H share one engine (no bidirectional
    // overlap). GPUs with 2 copy engines can do H2D+D2H simultaneously.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\n  Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Async copy engines: %d\n", prop.asyncEngineCount);
    printf("  Concurrent kernels: %s\n",
           prop.concurrentKernels ? "yes" : "no");

    // ---- Configuration ----
    const int N     = 1 << 24;   // ~16 million elements (~64 MB)
    const int ITERS = 300;       // compute iterations per element

    printf("\n  Problem size: N = %d (%.1f MB)\n", N, N * sizeof(float) / 1e6);
    printf("  Kernel iterations: %d\n", ITERS);

    // ---- Allocate PINNED host memory ----
    // This is absolutely essential for async transfers.
    //
    // Mistake to avoid:
    //   float *h_in = (float*)malloc(N * sizeof(float));  // PAGEABLE — bad!
    //   cudaMemcpyAsync(d_in, h_in, ..., stream);          // silently sync!
    //
    // Correct:
    //   cudaMallocHost(&h_in, N * sizeof(float));          // PINNED — good!
    //   cudaMemcpyAsync(d_in, h_in, ..., stream);          // truly async!
    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_out, N * sizeof(float)));

    // ---- Allocate device memory ----
    // We allocate the full array on the device once. Each stream will work
    // on a different offset into this array. No need for per-stream device
    // allocations — they just use different regions of the same buffer.
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // ---- Initialize input ----
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)(i % 10000) * 0.0001f;
    }

    // ---- Warmup ----
    run_serial(h_in, h_out, d_in, d_out, N, ITERS);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\n────────────────────────────────────────────────────────────\n");

    // ---- Serial baseline ----
    float t_serial = run_serial(h_in, h_out, d_in, d_out, N, ITERS);
    printf("  Serial:              %7.3f ms\n", t_serial);

    printf("────────────────────────────────────────────────────────────\n");

    // ---- Pipelined with various stream counts ----
    // We test different stream counts to show the diminishing returns.
    // With 1 copy engine:
    //   - 2 streams: some overlap possible
    //   - 4 streams: good pipeline utilization
    //   - 8 streams: near-optimal on most workloads
    //   - 16+ streams: little additional benefit, more overhead
    printf("\n  %-12s  %10s  %10s\n", "Streams", "Time (ms)", "Speedup");
    printf("  %-12s  %10s  %10s\n", "-------", "---------", "-------");

    int stream_counts[] = {1, 2, 4, 8, 16, 32};
    int num_tests = sizeof(stream_counts) / sizeof(stream_counts[0]);

    for (int t = 0; t < num_tests; t++) {
        int ns = stream_counts[t];
        float t_pipe = run_pipelined(h_in, h_out, d_in, d_out,
                                     N, ITERS, ns);
        float speedup = t_serial / t_pipe;
        printf("  %-12d  %10.3f  %10.2fx\n", ns, t_pipe, speedup);
    }

    printf("\n────────────────────────────────────────────────────────────\n");

    // ---- Verify correctness ----
    // Run serial to populate h_out with reference values
    float *h_ref;
    CUDA_CHECK(cudaMallocHost(&h_ref, N * sizeof(float)));
    run_serial(h_in, h_ref, d_in, d_out, N, ITERS);

    // Run pipelined to populate h_out
    run_pipelined(h_in, h_out, d_in, d_out, N, ITERS, 4);

    int mismatches = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_out[i] - h_ref[i]) > 1e-5f) {
            mismatches++;
            if (mismatches <= 3) {
                printf("  MISMATCH i=%d: serial=%.6f pipe=%.6f\n",
                       i, h_ref[i], h_out[i]);
            }
        }
    }
    printf("  Verification: %s (%d mismatches)\n",
           mismatches == 0 ? "PASS" : "FAIL", mismatches);

    // ---- Explain the results ----
    printf("\n  ANALYSIS:\n");
    printf("  ─────────────────────────────────────────────────────────\n");
    printf("  With enough streams, transfer time is hidden behind compute.\n");
    printf("  The P4200 has 1 copy engine, so H2D and D2H cannot overlap\n");
    printf("  each other, but both overlap with kernel execution.\n");
    printf("  \n");
    printf("  Theoretical best speedup with 1 copy engine:\n");
    printf("    Serial time   = T_h2d + T_kernel + T_d2h\n");
    printf("    Pipeline time = T_kernel + max(T_h2d, T_d2h) / N_streams\n");
    printf("    (approximately, if compute dominates)\n");
    printf("  ─────────────────────────────────────────────────────────\n");

    // ---- Cleanup ----
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFreeHost(h_ref));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    printf("\n═══════════════════════════════════════════════════════════════\n");
    return 0;
}
