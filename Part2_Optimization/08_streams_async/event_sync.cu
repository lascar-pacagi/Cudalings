// ============================================================================
// Chapter 08: CUDA Events for Timing and Synchronization
// ============================================================================
//
// CUDA events serve two critical roles:
//
//   1. TIMING:   Record GPU timestamps to measure how long operations take.
//                Unlike CPU-side timing (which measures wall clock time
//                including CPU overhead), event-based timing measures the
//                actual GPU execution time.
//
//   2. SYNCHRONIZATION: Create dependencies between streams using
//                cudaStreamWaitEvent(). This lets you build complex
//                dependency graphs across streams — producer/consumer
//                patterns, fork/join parallelism, etc.
//
// This file demonstrates:
//   - Creating and using events for GPU timing
//   - Timing individual operations within a stream
//   - Using cudaStreamWaitEvent for inter-stream dependencies
//   - Producer-consumer pattern: Stream A produces data, Stream B consumes it
//   - Non-blocking polling with cudaEventQuery
//
// ============================================================================
// INTER-STREAM DEPENDENCY DIAGRAM
// ============================================================================
//
//   Without events (no dependency):
//
//     Stream A: ┌──Kernel A──┐
//     Stream B: ┌──Kernel B──┐           ← B starts immediately, may read
//                                          garbage if it depends on A's output!
//
//   With cudaStreamWaitEvent:
//
//     Stream A: ┌──Kernel A──┐ ●event_done
//                                   │
//     Stream B:                     ▼ waitEvent(B, event_done)
//                                   ┌──Kernel B──┐
//                                   │ B starts AFTER A finishes!
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
// Kernel: compute-intensive work (producer)
// ============================================================================
// This kernel "produces" data by writing computed results to output[].
// In a real application, this might be a feature extraction step, a matrix
// multiply, or any computation whose results feed into a subsequent stage.
// ============================================================================
__global__ void producer_kernel(float *output, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Simulate work: compute a value through many iterations
    float val = (float)(idx % 1000) * 0.001f;
    for (int i = 0; i < iters; i++) {
        val = sinf(val) * 0.99f + 0.01f;
    }
    output[idx] = val;
}

// ============================================================================
// Kernel: consumer that reads from the producer's output
// ============================================================================
// This kernel "consumes" data produced by producer_kernel. It reads from
// input[] (which was output[] of the producer) and writes to output[].
//
// CRITICAL: This kernel must NOT start until the producer has finished
// writing to input[]. Without proper synchronization, the consumer could
// read garbage data — a classic race condition.
// ============================================================================
__global__ void consumer_kernel(const float *input, float *output,
                                int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Read the producer's output and do further processing
    float val = input[idx];
    for (int i = 0; i < iters; i++) {
        val = cosf(val) * 0.99f + 0.01f;
    }
    output[idx] = val;
}

// ============================================================================
// Kernel: independent work that can overlap with producer/consumer
// ============================================================================
// This kernel does unrelated work. It can run on the SMs while the
// producer or consumer is also running, demonstrating that events allow
// fine-grained control over what must wait and what can proceed freely.
// ============================================================================
__global__ void independent_kernel(float *data, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = data[idx];
    for (int i = 0; i < iters; i++) {
        val = val * 0.999f + 0.001f;
    }
    data[idx] = val;
}

// ============================================================================
// Demo 1: Precise GPU Timing with Events
// ============================================================================
// Shows how to time individual GPU operations using event pairs.
// Each operation is bracketed by a start event and a stop event.
//
// This is more accurate than CPU-side timing because:
//   - Events are recorded on the GPU clock
//   - No CPU-GPU synchronization latency is included
//   - You measure exactly what the GPU does, not CPU overhead
// ============================================================================
void demo_timing() {
    printf("  DEMO 1: Precise GPU Timing with Events\n");
    printf("  ─────────────────────────────────────────────────────────\n");

    const int N = 1 << 20;  // 1 million elements
    const int ITERS = 200;

    // ---- Allocate ----
    float *h_data, *d_in, *d_out;
    CUDA_CHECK(cudaMallocHost(&h_data, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    for (int i = 0; i < N; i++) h_data[i] = (float)i * 0.001f;

    // ---- Create events for timing each phase ----
    // We need pairs of events: one before and one after each operation.
    // cudaEventElapsedTime(start, stop) gives the time between them.
    cudaEvent_t ev_h2d_start, ev_h2d_stop;
    cudaEvent_t ev_kern_start, ev_kern_stop;
    cudaEvent_t ev_d2h_start, ev_d2h_stop;

    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_stop));
    CUDA_CHECK(cudaEventCreate(&ev_kern_start));
    CUDA_CHECK(cudaEventCreate(&ev_kern_stop));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // ---- Time each operation individually ----
    // By placing event pairs around each operation, we get precise per-phase
    // timing. This is invaluable for identifying bottlenecks.

    // Time H2D
    CUDA_CHECK(cudaEventRecord(ev_h2d_start, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_data, N * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(ev_h2d_stop, stream));

    // Time kernel
    CUDA_CHECK(cudaEventRecord(ev_kern_start, stream));
    producer_kernel<<<blocks, threads, 0, stream>>>(d_out, N, ITERS);
    CUDA_CHECK(cudaEventRecord(ev_kern_stop, stream));

    // Time D2H
    CUDA_CHECK(cudaEventRecord(ev_d2h_start, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_out, N * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(ev_d2h_stop, stream));

    // ---- Synchronize and read timings ----
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_stop));

    float t_h2d, t_kern, t_d2h;
    CUDA_CHECK(cudaEventElapsedTime(&t_h2d,  ev_h2d_start, ev_h2d_stop));
    CUDA_CHECK(cudaEventElapsedTime(&t_kern, ev_kern_start, ev_kern_stop));
    CUDA_CHECK(cudaEventElapsedTime(&t_d2h,  ev_d2h_start, ev_d2h_stop));

    printf("    H2D transfer:   %7.3f ms  (%.1f MB)\n",
           t_h2d, N * sizeof(float) / 1e6);
    printf("    Kernel:          %7.3f ms  (%d iters/element)\n",
           t_kern, ITERS);
    printf("    D2H transfer:   %7.3f ms  (%.1f MB)\n",
           t_d2h, N * sizeof(float) / 1e6);
    printf("    Sum:             %7.3f ms\n", t_h2d + t_kern + t_d2h);

    // Also measure total wall time for comparison
    float t_total;
    CUDA_CHECK(cudaEventElapsedTime(&t_total, ev_h2d_start, ev_d2h_stop));
    printf("    Total (event):   %7.3f ms\n", t_total);
    printf("    (Sum ≈ Total because everything is in one stream = serial)\n");

    // ---- Cleanup ----
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_start));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_stop));
    CUDA_CHECK(cudaEventDestroy(ev_kern_start));
    CUDA_CHECK(cudaEventDestroy(ev_kern_stop));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_start));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_stop));
    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

// ============================================================================
// Demo 2: Producer-Consumer with cudaStreamWaitEvent
// ============================================================================
// This is the most important pattern for real-world CUDA programming.
//
// Scenario:
//   - Stream A runs a "producer" kernel that computes some data
//   - Stream B runs a "consumer" kernel that reads that data
//   - Stream C runs an independent kernel that doesn't need the data
//
// Without events: Streams B and C would both start immediately (race!)
// With events:    Stream B waits for Stream A to finish (correct!)
//                 Stream C runs freely (it doesn't depend on A)
//
//   Stream A: ┌──Producer──┐ ●event_produced
//                                │
//   Stream B:                    ▼ waitEvent(B, event_produced)
//                                ┌──Consumer──┐
//
//   Stream C: ┌──Independent────────────────┐  ← no wait, runs freely!
//
// This gives us:
//   - Correctness: Consumer reads valid data (waits for producer)
//   - Performance: Independent work overlaps with producer
// ============================================================================
void demo_producer_consumer() {
    printf("\n  DEMO 2: Producer-Consumer with cudaStreamWaitEvent\n");
    printf("  ─────────────────────────────────────────────────────────\n");

    const int N = 1 << 20;
    const int ITERS = 300;

    // ---- Allocate ----
    float *d_produced, *d_consumed, *d_independent;
    CUDA_CHECK(cudaMalloc(&d_produced,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_consumed,    N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_independent, N * sizeof(float)));

    // Initialize independent data
    CUDA_CHECK(cudaMemset(d_independent, 0, N * sizeof(float)));

    // ---- Create 3 streams and events ----
    cudaStream_t stream_producer, stream_consumer, stream_independent;
    CUDA_CHECK(cudaStreamCreate(&stream_producer));
    CUDA_CHECK(cudaStreamCreate(&stream_consumer));
    CUDA_CHECK(cudaStreamCreate(&stream_independent));

    // This event marks when the producer has finished writing.
    // The consumer will wait on this event before reading.
    cudaEvent_t event_produced;
    CUDA_CHECK(cudaEventCreate(&event_produced));

    // Timing events
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // ---- INCORRECT version (for comparison): no synchronization ----
    // If we just launch everything without events, the consumer might
    // read garbage. We don't actually run this, just explain it.
    printf("    Without events: consumer could read uninitialized data!\n");
    printf("    With events:    consumer waits for producer to finish.\n\n");

    // ---- CORRECT version: producer → event → consumer ----
    CUDA_CHECK(cudaEventRecord(ev_start));

    // Step 1: Launch producer in its own stream
    producer_kernel<<<blocks, threads, 0, stream_producer>>>(
        d_produced, N, ITERS);

    // Step 2: Record event AFTER producer finishes
    // This event will be "completed" when all prior operations in
    // stream_producer have finished (i.e., when producer_kernel is done).
    CUDA_CHECK(cudaEventRecord(event_produced, stream_producer));

    // Step 3: Make consumer stream WAIT on the producer's event
    // cudaStreamWaitEvent(stream, event, flags):
    //   "Do not execute any more operations in 'stream' until 'event'
    //    has been recorded (completed) in whatever stream it belongs to."
    //
    // After this call, any operation issued to stream_consumer will
    // not start until event_produced is completed (i.e., after the
    // producer kernel finishes).
    //
    // The flags parameter is reserved and must be 0.
    CUDA_CHECK(cudaStreamWaitEvent(stream_consumer, event_produced, 0));

    // Step 4: Launch consumer — safe! It won't start until producer is done
    consumer_kernel<<<blocks, threads, 0, stream_consumer>>>(
        d_produced, d_consumed, N, ITERS);

    // Step 5: Launch independent work — no wait needed!
    // This kernel doesn't read d_produced, so it can run concurrently
    // with the producer. No event wait is necessary.
    // If the GPU has enough SM resources, this kernel overlaps with
    // the producer and/or consumer.
    independent_kernel<<<blocks, threads, 0, stream_independent>>>(
        d_independent, N, ITERS);

    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));

    float t_total;
    CUDA_CHECK(cudaEventElapsedTime(&t_total, ev_start, ev_end));

    // ---- Time the serial equivalent for comparison ----
    // If we ran producer, consumer, independent all in the same stream:
    cudaEvent_t ev_serial_start, ev_serial_end;
    CUDA_CHECK(cudaEventCreate(&ev_serial_start));
    CUDA_CHECK(cudaEventCreate(&ev_serial_end));

    CUDA_CHECK(cudaEventRecord(ev_serial_start));
    producer_kernel<<<blocks, threads>>>(d_produced, N, ITERS);
    consumer_kernel<<<blocks, threads>>>(d_produced, d_consumed, N, ITERS);
    independent_kernel<<<blocks, threads>>>(d_independent, N, ITERS);
    CUDA_CHECK(cudaEventRecord(ev_serial_end));
    CUDA_CHECK(cudaEventSynchronize(ev_serial_end));

    float t_serial;
    CUDA_CHECK(cudaEventElapsedTime(&t_serial, ev_serial_start, ev_serial_end));

    printf("    Serial (all default stream):  %7.3f ms\n", t_serial);
    printf("    With events (3 streams):      %7.3f ms\n", t_total);
    printf("    Speedup:                      %7.2fx\n", t_serial / t_total);
    printf("\n    The independent kernel overlaps with producer+consumer,\n");
    printf("    saving the time it would have taken serially.\n");

    // ---- Cleanup ----
    CUDA_CHECK(cudaStreamDestroy(stream_producer));
    CUDA_CHECK(cudaStreamDestroy(stream_consumer));
    CUDA_CHECK(cudaStreamDestroy(stream_independent));
    CUDA_CHECK(cudaEventDestroy(event_produced));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    CUDA_CHECK(cudaEventDestroy(ev_serial_start));
    CUDA_CHECK(cudaEventDestroy(ev_serial_end));
    CUDA_CHECK(cudaFree(d_produced));
    CUDA_CHECK(cudaFree(d_consumed));
    CUDA_CHECK(cudaFree(d_independent));
}

// ============================================================================
// Demo 3: Non-blocking Polling with cudaEventQuery
// ============================================================================
// cudaEventQuery(event) checks whether an event has completed WITHOUT
// blocking the CPU. Returns:
//   - cudaSuccess          → event has completed
//   - cudaErrorNotReady    → event has NOT completed yet (not an error!)
//
// This is useful when the CPU has other work to do while waiting for
// the GPU. Instead of blocking with cudaEventSynchronize(), the CPU can
// poll periodically and do useful work in between.
//
// Real-world uses:
//   - Polling in a game loop (render while waiting for compute)
//   - Checking if a GPU operation finished before submitting more work
//   - Implementing timeouts for GPU operations
// ============================================================================
void demo_event_query() {
    printf("\n  DEMO 3: Non-blocking Polling with cudaEventQuery\n");
    printf("  ─────────────────────────────────────────────────────────\n");

    const int N = 1 << 20;
    const int ITERS = 500;  // enough work to take some time

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Create the event we'll poll
    cudaEvent_t completion_event;
    CUDA_CHECK(cudaEventCreate(&completion_event));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    // Launch a long-running kernel
    producer_kernel<<<blocks, threads, 0, stream>>>(d_data, N, ITERS);

    // Record event after the kernel
    CUDA_CHECK(cudaEventRecord(completion_event, stream));

    // ---- Poll the event from the CPU side ----
    // Instead of blocking with cudaEventSynchronize, we check periodically.
    // In a real application, the CPU would do useful work between polls.
    int polls = 0;
    cudaError_t status;

    printf("    Polling for kernel completion...\n");

    do {
        // cudaEventQuery returns immediately (non-blocking!)
        status = cudaEventQuery(completion_event);
        polls++;

        // We could do CPU work here:
        //   - Process input
        //   - Update UI
        //   - Prepare the next batch of GPU work
        //   - etc.

        if (status == cudaErrorNotReady) {
            // Event not ready yet — this is NOT an error condition.
            // cudaErrorNotReady is a special "status code" that means
            // "still in progress." We clear the error state.
            // Note: cudaGetLastError() would return this "error" otherwise.
        } else if (status != cudaSuccess) {
            // An actual error occurred
            fprintf(stderr, "    Error during query: %s\n",
                    cudaGetErrorString(status));
            break;
        }
    } while (status == cudaErrorNotReady);

    printf("    Kernel completed after %d polls.\n", polls);
    printf("    (If polls == 1, kernel was already done before first check)\n");
    printf("    (If polls >> 1, CPU had many opportunities for other work)\n");

    // ---- Also demonstrate cudaStreamQuery ----
    // Similar to cudaEventQuery, but checks if ALL operations in a stream
    // have completed.
    printf("\n    cudaStreamQuery after completion:\n");
    status = cudaStreamQuery(stream);
    printf("    Stream status: %s\n",
           status == cudaSuccess ? "idle (all work done)" : "busy");

    // ---- Cleanup ----
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(completion_event));
    CUDA_CHECK(cudaFree(d_data));
}

// ============================================================================
// Demo 4: Multi-stage Pipeline with Events
// ============================================================================
// Demonstrates a more complex dependency graph:
//
//   Stage 1 (Stream A): Produce data set 1
//   Stage 2 (Stream B): Produce data set 2 (independent of Stage 1)
//   Stage 3 (Stream C): Consume BOTH data set 1 and 2 (depends on A AND B)
//
//   Stream A: ┌──Produce 1──┐ ●ev1_done
//                                    \
//   Stream B: ┌──Produce 2──┐ ●ev2_done
//                                    / \
//   Stream C:                        ▼  ▼ waitEvent(C, ev1) + waitEvent(C, ev2)
//                                    ┌──Consume 1+2──┐
//
// Stream C calls cudaStreamWaitEvent TWICE — once for each dependency.
// It will not start until BOTH events have completed.
// Meanwhile, Streams A and B run concurrently!
// ============================================================================
void demo_multistage_pipeline() {
    printf("\n  DEMO 4: Multi-stage Pipeline (Fork-Join)\n");
    printf("  ─────────────────────────────────────────────────────────\n");

    const int N = 1 << 20;
    const int ITERS = 200;

    float *d_data1, *d_data2, *d_result;
    CUDA_CHECK(cudaMalloc(&d_data1,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_data2,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, N * sizeof(float)));

    // Three streams for three stages
    cudaStream_t stream_a, stream_b, stream_c;
    CUDA_CHECK(cudaStreamCreate(&stream_a));
    CUDA_CHECK(cudaStreamCreate(&stream_b));
    CUDA_CHECK(cudaStreamCreate(&stream_c));

    // Two events: one for each producer
    cudaEvent_t ev1_done, ev2_done;
    CUDA_CHECK(cudaEventCreate(&ev1_done));
    CUDA_CHECK(cudaEventCreate(&ev2_done));

    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(ev_start));

    // Fork: launch two independent producers in parallel
    producer_kernel<<<blocks, threads, 0, stream_a>>>(d_data1, N, ITERS);
    CUDA_CHECK(cudaEventRecord(ev1_done, stream_a));  // mark producer 1 done

    producer_kernel<<<blocks, threads, 0, stream_b>>>(d_data2, N, ITERS);
    CUDA_CHECK(cudaEventRecord(ev2_done, stream_b));  // mark producer 2 done

    // Join: consumer waits for BOTH producers
    // Multiple cudaStreamWaitEvent calls accumulate — the stream won't
    // proceed until ALL waited-on events have completed.
    CUDA_CHECK(cudaStreamWaitEvent(stream_c, ev1_done, 0));
    CUDA_CHECK(cudaStreamWaitEvent(stream_c, ev2_done, 0));

    // Now it's safe to read from both d_data1 and d_data2
    consumer_kernel<<<blocks, threads, 0, stream_c>>>(
        d_data1, d_result, N, ITERS);

    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));

    float t_total;
    CUDA_CHECK(cudaEventElapsedTime(&t_total, ev_start, ev_end));

    // Compare with serial: all 3 kernels in sequence
    CUDA_CHECK(cudaEventRecord(ev_start));
    producer_kernel<<<blocks, threads>>>(d_data1, N, ITERS);
    producer_kernel<<<blocks, threads>>>(d_data2, N, ITERS);
    consumer_kernel<<<blocks, threads>>>(d_data1, d_result, N, ITERS);
    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));

    float t_serial;
    CUDA_CHECK(cudaEventElapsedTime(&t_serial, ev_start, ev_end));

    printf("    Serial:           %7.3f ms  (P1 → P2 → Consumer)\n", t_serial);
    printf("    Fork-Join:        %7.3f ms  (P1 || P2 → Consumer)\n", t_total);
    printf("    Speedup:          %7.2fx\n", t_serial / t_total);
    printf("\n    Producers ran in parallel (fork), consumer waited for\n");
    printf("    both to finish (join via two cudaStreamWaitEvent calls).\n");

    // ---- Cleanup ----
    CUDA_CHECK(cudaStreamDestroy(stream_a));
    CUDA_CHECK(cudaStreamDestroy(stream_b));
    CUDA_CHECK(cudaStreamDestroy(stream_c));
    CUDA_CHECK(cudaEventDestroy(ev1_done));
    CUDA_CHECK(cudaEventDestroy(ev2_done));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
    CUDA_CHECK(cudaFree(d_data1));
    CUDA_CHECK(cudaFree(d_data2));
    CUDA_CHECK(cudaFree(d_result));
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Chapter 08: CUDA Events — Timing and Synchronization\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // ---- Device info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  SMs: %d, Copy engines: %d\n",
           prop.multiProcessorCount, prop.asyncEngineCount);
    printf("  Concurrent kernels: %s\n\n",
           prop.concurrentKernels ? "yes" : "no");

    // ---- Warmup ----
    {
        float *d_tmp;
        CUDA_CHECK(cudaMalloc(&d_tmp, 1024));
        CUDA_CHECK(cudaFree(d_tmp));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("────────────────────────────────────────────────────────────\n");
    demo_timing();

    printf("\n────────────────────────────────────────────────────────────\n");
    demo_producer_consumer();

    printf("\n────────────────────────────────────────────────────────────\n");
    demo_event_query();

    printf("\n────────────────────────────────────────────────────────────\n");
    demo_multistage_pipeline();

    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("  KEY TAKEAWAYS:\n");
    printf("  • cudaEventRecord(event, stream) — insert timestamp marker\n");
    printf("  • cudaEventElapsedTime(&ms, start, stop) — GPU timing\n");
    printf("  • cudaStreamWaitEvent(stream, event, 0) — cross-stream sync\n");
    printf("  • cudaEventQuery(event) — non-blocking completion check\n");
    printf("  • Events enable producer-consumer and fork-join patterns\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}
