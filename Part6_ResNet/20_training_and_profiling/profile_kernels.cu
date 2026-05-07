/*******************************************************************************
 * profile_kernels.cu -- Kernel-Level Profiling of Our ResNet
 *
 * Chapter 20: Training, Profiling, and the Full Journey
 *
 * This program runs a single forward+backward pass on a batch and measures
 * the time spent in each individual operation. It then prints a detailed
 * profiling report showing:
 *   - Time per operation (conv2d, batchnorm, relu, linear, GAP, loss)
 *   - Bandwidth utilization estimates
 *   - Bottleneck identification
 *   - nvprof/nsys commands for deeper analysis
 *
 * This is the kind of analysis you'd do BEFORE optimizing: measure first,
 * then optimize the hottest kernel. Never optimize blind.
 *
 * Compile: nvcc -arch=sm_61 -O2 -ccbin g++-11 -std=c++14 -lcurand -lineinfo profile_kernels.cu -o profile_kernels
 ******************************************************************************/

// Include our complete ResNet implementation from Chapter 19
#include "../19_resnet_from_scratch/resnet.cuh"

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

// ============================================================================
// Profiling Configuration
// ============================================================================

static const int BATCH_SIZE   = 64;    // Same as training
static const int CHANNELS     = 32;
static const int NUM_BLOCKS   = 4;
static const int FC_SIZE      = 32;
static const int NUM_CLASSES  = 10;
static const int IN_CHANNELS  = 4;
static const int SPATIAL      = 8;
static const int WARMUP_ITERS = 5;     // Warmup iterations to stabilize timing
static const int BENCH_ITERS  = 20;    // Iterations to average over


// ============================================================================
// Utility: CUDA Event Timer
// ============================================================================
//
// A simple RAII wrapper for CUDA event-based timing.
// Records start on construction, stop on finish(), returns elapsed ms.
// CUDA events measure GPU-side time, which is more accurate than
// host-side timers because it doesn't include driver overhead.
// ============================================================================

struct CudaTimer {
    cudaEvent_t start, stop;
    float elapsed_ms;

    CudaTimer() : elapsed_ms(0.0f) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Record the start timestamp on the GPU
    void begin() {
        cudaEventRecord(start);
    }

    // Record the stop timestamp and compute elapsed time
    float finish() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        return elapsed_ms;
    }
};


// ============================================================================
// Profile Individual Operations
// ============================================================================
//
// We profile each operation in isolation by running it multiple times and
// measuring with CUDA events. This isolates the kernel cost from graph
// traversal and other overhead.
//
// For each operation, we also estimate the memory bandwidth utilization:
//   bandwidth = (bytes_read + bytes_written) / (time_seconds)
//
// The Quadro P4200 has ~192 GB/s peak memory bandwidth. Our kernels are
// mostly memory-bound (see roofline in README.md), so bandwidth utilization
// is the key metric.
// ============================================================================

// Profile Conv2d forward pass
// Conv2d is usually the hottest kernel because it has the most arithmetic
// and the most memory traffic.
//
// For Conv2d(C_in, C_out, 3, pad=1) on (B, C_in, H, W):
//   Output: (B, C_out, H, W)
//   FLOPs:  B * C_out * H * W * C_in * 3 * 3 * 2  (mul + add per MAC)
//   Reads:  input (B*C_in*H*W) + weight (C_out*C_in*3*3)
//   Writes: output (B*C_out*H*W)
static void profile_conv2d(int B, int C_in, int C_out, int H, int W,
                           int ksize, int pad, int iters) {
    // Create test tensors
    GradTensor* input  = GradTensor::randn({B, C_in, H, W});
    GradTensor* weight = GradTensor::randn({C_out, C_in, ksize, ksize}, 0.1f, true);

    int oH = H + 2 * pad - ksize + 1;
    int oW = W + 2 * pad - ksize + 1;

    // Warmup: let the GPU "warm up" its clocks and caches
    for (int i = 0; i < WARMUP_ITERS; i++) {
        GradTensor* out = new GradTensor({B, C_out, oH, oW});
        int total = B * C_out * oH * oW;
        conv2d_forward_kernel<<<ceildiv(total, BLOCK_SIZE), BLOCK_SIZE>>>(
            input->data, weight->data, out->data,
            B, C_in, H, W, C_out, ksize, ksize, oH, oW, pad);
        CUDA_CHECK(cudaDeviceSynchronize());
        delete out;
    }

    // Benchmark
    CudaTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        GradTensor* out = new GradTensor({B, C_out, oH, oW});
        int total = B * C_out * oH * oW;

        timer.begin();
        conv2d_forward_kernel<<<ceildiv(total, BLOCK_SIZE), BLOCK_SIZE>>>(
            input->data, weight->data, out->data,
            B, C_in, H, W, C_out, ksize, ksize, oH, oW, pad);
        total_ms += timer.finish();

        delete out;
    }

    float avg_ms = total_ms / iters;

    // Compute FLOPS and bandwidth
    long long flops = (long long)B * C_out * oH * oW * C_in * ksize * ksize * 2;
    long long bytes_read = ((long long)B * C_in * H * W + (long long)C_out * C_in * ksize * ksize) * sizeof(float);
    long long bytes_write = (long long)B * C_out * oH * oW * sizeof(float);
    long long total_bytes = bytes_read + bytes_write;

    float gflops = flops / (avg_ms * 1e6f);
    float bw_gb_s = total_bytes / (avg_ms * 1e6f);

    printf("  Conv2d(%d->%d, %dx%d, pad=%d):\n", C_in, C_out, ksize, ksize, pad);
    printf("    Time:      %8.3f ms\n", avg_ms);
    printf("    GFLOPS:    %8.2f\n", gflops);
    printf("    Bandwidth: %8.2f GB/s (peak ~192 GB/s)\n", bw_gb_s);
    printf("    BW util:   %8.1f%%\n", 100.0f * bw_gb_s / 192.0f);

    delete input;
    delete weight;
}

// Profile Conv2d backward passes (d_input + d_weight)
// Backward typically takes more time because we compute TWO things:
//   1. d_input:  gradient w.r.t. input (for upstream layers)
//   2. d_weight: gradient w.r.t. weight (for parameter update)
static void profile_conv2d_backward(int B, int C_in, int C_out, int H, int W,
                                     int ksize, int pad, int iters) {
    int oH = H + 2 * pad - ksize + 1;
    int oW = W + 2 * pad - ksize + 1;

    GradTensor* input    = GradTensor::randn({B, C_in, H, W});
    GradTensor* weight   = GradTensor::randn({C_out, C_in, ksize, ksize}, 0.1f, true);
    GradTensor* d_output = GradTensor::randn({B, C_out, oH, oW});

    float* d_input  = gpu_malloc(B * C_in * H * W);
    float* d_weight = gpu_malloc(C_out * C_in * ksize * ksize);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        gpu_memset_zero(d_input, B * C_in * H * W);
        gpu_memset_zero(d_weight, C_out * C_in * ksize * ksize);
        int total_in = B * C_in * H * W;
        int total_w = C_out * C_in * ksize * ksize;
        conv2d_backward_input_kernel<<<ceildiv(total_in, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_output->data, weight->data, d_input,
            B, C_in, H, W, C_out, ksize, ksize, oH, oW, pad);
        conv2d_backward_weight_kernel<<<ceildiv(total_w, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_output->data, input->data, d_weight,
            B, C_in, H, W, C_out, ksize, ksize, oH, oW, pad);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    CudaTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        gpu_memset_zero(d_input, B * C_in * H * W);
        gpu_memset_zero(d_weight, C_out * C_in * ksize * ksize);
        int total_in = B * C_in * H * W;
        int total_w = C_out * C_in * ksize * ksize;

        timer.begin();
        conv2d_backward_input_kernel<<<ceildiv(total_in, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_output->data, weight->data, d_input,
            B, C_in, H, W, C_out, ksize, ksize, oH, oW, pad);
        conv2d_backward_weight_kernel<<<ceildiv(total_w, BLOCK_SIZE), BLOCK_SIZE>>>(
            d_output->data, input->data, d_weight,
            B, C_in, H, W, C_out, ksize, ksize, oH, oW, pad);
        total_ms += timer.finish();
    }

    float avg_ms = total_ms / iters;
    printf("  Conv2d backward (d_input + d_weight):\n");
    printf("    Time:      %8.3f ms\n", avg_ms);

    gpu_free(d_input);
    gpu_free(d_weight);
    delete input;
    delete weight;
    delete d_output;
}

// Profile BatchNorm2d forward
// BatchNorm has 3 sub-kernels in forward:
//   1. Compute mean    (reduction over N*H*W per channel)
//   2. Compute variance (reduction over N*H*W per channel)
//   3. Normalize + affine (element-wise)
// Plus running stats update (small).
static void profile_batchnorm(int B, int C, int H, int W, int iters) {
    GradTensor* input = GradTensor::randn({B, C, H, W});
    GradTensor* gamma = GradTensor::ones({C}, true);
    GradTensor* beta  = GradTensor::zeros({C}, true);
    float* running_mean = gpu_malloc(C);
    float* running_var  = gpu_malloc(C);
    gpu_memset_zero(running_mean, C);
    std::vector<float> ones(C, 1.0f);
    gpu_copy_h2d(running_var, ones.data(), C);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        clear_tensor_arena();
        autograd_batchnorm2d(input, gamma, beta, running_mean, running_var,
                              true, 0.1f, 1e-5f);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    CudaTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        clear_tensor_arena();
        timer.begin();
        autograd_batchnorm2d(input, gamma, beta, running_mean, running_var,
                              true, 0.1f, 1e-5f);
        total_ms += timer.finish();
    }

    float avg_ms = total_ms / iters;
    long long total_bytes = (long long)B * C * H * W * sizeof(float) * 3; // read + write + x_hat
    float bw_gb_s = total_bytes / (avg_ms * 1e6f);

    printf("  BatchNorm2d(%d) on (%d,%d,%d,%d):\n", C, B, C, H, W);
    printf("    Time:      %8.3f ms\n", avg_ms);
    printf("    Bandwidth: %8.2f GB/s\n", bw_gb_s);

    clear_tensor_arena();
    gpu_free(running_mean);
    gpu_free(running_var);
    delete input;
    delete gamma;
    delete beta;
}

// Profile ReLU forward
// ReLU is the simplest operation: one read, one comparison, one write.
// It should be entirely memory-bound.
static void profile_relu(int B, int C, int H, int W, int iters) {
    GradTensor* input = GradTensor::randn({B, C, H, W});

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        clear_tensor_arena();
        autograd_relu(input);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    CudaTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        clear_tensor_arena();
        timer.begin();
        autograd_relu(input);
        total_ms += timer.finish();
    }

    float avg_ms = total_ms / iters;
    long long total_bytes = (long long)B * C * H * W * sizeof(float) * 2; // read + write
    float bw_gb_s = total_bytes / (avg_ms * 1e6f);

    printf("  ReLU on (%d,%d,%d,%d):\n", B, C, H, W);
    printf("    Time:      %8.3f ms\n", avg_ms);
    printf("    Bandwidth: %8.2f GB/s\n", bw_gb_s);

    clear_tensor_arena();
    delete input;
}

// Profile Linear forward
static void profile_linear(int B, int in_f, int out_f, int iters) {
    GradTensor* input  = GradTensor::randn({B, in_f});
    GradTensor* weight = GradTensor::randn({out_f, in_f}, 0.1f, true);
    GradTensor* bias   = GradTensor::zeros({out_f}, true);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        clear_tensor_arena();
        autograd_linear(input, weight, bias);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    CudaTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        clear_tensor_arena();
        timer.begin();
        autograd_linear(input, weight, bias);
        total_ms += timer.finish();
    }

    float avg_ms = total_ms / iters;
    long long flops = (long long)B * in_f * out_f * 2;
    float gflops = flops / (avg_ms * 1e6f);

    printf("  Linear(%d->%d) batch=%d:\n", in_f, out_f, B);
    printf("    Time:      %8.3f ms\n", avg_ms);
    printf("    GFLOPS:    %8.2f\n", gflops);

    clear_tensor_arena();
    delete input;
    delete weight;
    delete bias;
}

// Profile Global Average Pooling
static void profile_gap(int B, int C, int H, int W, int iters) {
    GradTensor* input = GradTensor::randn({B, C, H, W});

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        clear_tensor_arena();
        autograd_global_avg_pool(input);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    CudaTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        clear_tensor_arena();
        timer.begin();
        autograd_global_avg_pool(input);
        total_ms += timer.finish();
    }

    float avg_ms = total_ms / iters;
    printf("  GlobalAvgPool (%d,%d,%d,%d)->(%d,%d):\n", B, C, H, W, B, C);
    printf("    Time:      %8.3f ms\n", avg_ms);

    clear_tensor_arena();
    delete input;
}

// Profile CrossEntropy loss
static void profile_cross_entropy(int B, int C, int iters) {
    GradTensor* logits = GradTensor::randn({B, C});
    std::vector<int> targets(B);
    for (int i = 0; i < B; i++) targets[i] = rand() % C;

    CrossEntropyLoss criterion;

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        clear_tensor_arena();
        criterion.forward(logits, targets);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    CudaTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        clear_tensor_arena();
        timer.begin();
        criterion.forward(logits, targets);
        total_ms += timer.finish();
    }

    float avg_ms = total_ms / iters;
    printf("  CrossEntropy (B=%d, C=%d):\n", B, C);
    printf("    Time:      %8.3f ms\n", avg_ms);

    clear_tensor_arena();
    delete logits;
}


// ============================================================================
// Profile Full Forward + Backward Pass
// ============================================================================
//
// This measures the end-to-end time for one training step (minus data loading
// and optimizer). It shows the REAL cost of our ResNet.
// ============================================================================

static void profile_full_pass(int iters) {
    printf("\n--- Full Forward + Backward Pass ---\n");

    // Create model
    ResNet model(CHANNELS, NUM_BLOCKS, FC_SIZE, NUM_CLASSES, IN_CHANNELS);
    model.train();
    CrossEntropyLoss criterion;

    // Create random batch
    int img_size = IN_CHANNELS * SPATIAL * SPATIAL;
    std::vector<float> batch_data(BATCH_SIZE * img_size);
    for (int i = 0; i < BATCH_SIZE * img_size; i++) {
        batch_data[i] = host_randn() * 0.5f;
    }
    std::vector<int> targets(BATCH_SIZE);
    for (int i = 0; i < BATCH_SIZE; i++) targets[i] = rand() % NUM_CLASSES;

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        clear_tensor_arena();
        GradTensor* input = make_input(batch_data,
                                       {BATCH_SIZE, IN_CHANNELS, SPATIAL, SPATIAL});
        GradTensor* logits = model.forward(input);
        GradTensor* loss = criterion.forward(logits, targets);
        loss->backward();
        CUDA_CHECK(cudaDeviceSynchronize());
        delete input;
    }

    // Benchmark
    CudaTimer timer_fwd, timer_bwd, timer_total;
    float total_fwd = 0.0f, total_bwd = 0.0f, total_all = 0.0f;

    for (int i = 0; i < iters; i++) {
        clear_tensor_arena();
        GradTensor* input = make_input(batch_data,
                                       {BATCH_SIZE, IN_CHANNELS, SPATIAL, SPATIAL});

        // Forward
        timer_total.begin();
        timer_fwd.begin();
        GradTensor* logits = model.forward(input);
        GradTensor* loss = criterion.forward(logits, targets);
        CUDA_CHECK(cudaDeviceSynchronize());
        total_fwd += timer_fwd.finish();

        // Backward
        timer_bwd.begin();
        loss->backward();
        CUDA_CHECK(cudaDeviceSynchronize());
        total_bwd += timer_bwd.finish();

        total_all += timer_total.finish();
        delete input;
    }

    float avg_fwd = total_fwd / iters;
    float avg_bwd = total_bwd / iters;
    float avg_all = total_all / iters;

    printf("  Forward:     %8.3f ms  (%5.1f%%)\n", avg_fwd, 100.0f * avg_fwd / avg_all);
    printf("  Backward:    %8.3f ms  (%5.1f%%)\n", avg_bwd, 100.0f * avg_bwd / avg_all);
    printf("  Total:       %8.3f ms\n", avg_all);
    printf("  Throughput:  %8.0f samples/sec\n", BATCH_SIZE / (avg_all / 1000.0f));
    printf("  Fwd:Bwd ratio: 1 : %.2f\n", avg_bwd / avg_fwd);

    clear_tensor_arena();
}


// ============================================================================
// Main
// ============================================================================

int main() {
    printf("==========================================================\n");
    printf("  Chapter 20: Kernel-Level Profiling\n");
    printf("  Measuring every operation in our ResNet\n");
    printf("==========================================================\n\n");

    // GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  SMs: %d, Clock: %d MHz, Mem BW: ~192 GB/s\n",
           prop.multiProcessorCount, prop.clockRate / 1000);
    printf("  Warmup: %d iters, Benchmark: %d iters\n", WARMUP_ITERS, BENCH_ITERS);
    printf("\n");

    // ---- Profile individual operations ----
    // These are the exact dimensions used in our ResNet:
    //   Stem conv:  Conv2d(4 -> 32, 3x3, pad=1) on (64, 4, 8, 8)
    //   Body conv:  Conv2d(32 -> 32, 3x3, pad=1) on (64, 32, 8, 8)
    //   Linear1:    Linear(32 -> 32)
    //   Linear2:    Linear(32 -> 10)

    printf("=== Individual Operation Profiling ===\n");
    printf("    (batch=%d, spatial=%dx%d)\n\n", BATCH_SIZE, SPATIAL, SPATIAL);

    // Stem convolution: smaller input channels
    printf("-- Stem Conv2d --\n");
    profile_conv2d(BATCH_SIZE, IN_CHANNELS, CHANNELS, SPATIAL, SPATIAL, 3, 1, BENCH_ITERS);
    printf("\n");

    // Body convolution: this is the hottest kernel (called 8 times per forward pass)
    printf("-- Body Conv2d (called 8x per forward in 4 ResBlocks) --\n");
    profile_conv2d(BATCH_SIZE, CHANNELS, CHANNELS, SPATIAL, SPATIAL, 3, 1, BENCH_ITERS);
    printf("\n");

    // Conv2d backward (the most expensive single operation)
    printf("-- Body Conv2d Backward --\n");
    profile_conv2d_backward(BATCH_SIZE, CHANNELS, CHANNELS, SPATIAL, SPATIAL, 3, 1, BENCH_ITERS);
    printf("\n");

    // BatchNorm
    printf("-- BatchNorm2d --\n");
    profile_batchnorm(BATCH_SIZE, CHANNELS, SPATIAL, SPATIAL, BENCH_ITERS);
    printf("\n");

    // ReLU
    printf("-- ReLU --\n");
    profile_relu(BATCH_SIZE, CHANNELS, SPATIAL, SPATIAL, BENCH_ITERS);
    printf("\n");

    // Linear layers
    printf("-- Linear Layers --\n");
    profile_linear(BATCH_SIZE, CHANNELS, FC_SIZE, BENCH_ITERS);
    profile_linear(BATCH_SIZE, FC_SIZE, NUM_CLASSES, BENCH_ITERS);
    printf("\n");

    // Global Average Pooling
    printf("-- Global Average Pooling --\n");
    profile_gap(BATCH_SIZE, CHANNELS, SPATIAL, SPATIAL, BENCH_ITERS);
    printf("\n");

    // Cross-Entropy Loss
    printf("-- Cross-Entropy Loss --\n");
    profile_cross_entropy(BATCH_SIZE, NUM_CLASSES, BENCH_ITERS);
    printf("\n");

    // ---- Full forward + backward ----
    profile_full_pass(BENCH_ITERS);

    // ---- Bottleneck Analysis ----
    printf("\n=== BOTTLENECK ANALYSIS ===\n");
    printf("\n");
    printf("In our ResNet(channels=32, blocks=4) on (64, 4, 8, 8):\n");
    printf("\n");
    printf("  1. Conv2d dominates both forward and backward:\n");
    printf("     - 9 conv2d forward calls per forward pass\n");
    printf("       (1 stem + 2 per block * 4 blocks = 9)\n");
    printf("     - Each backward computes d_input AND d_weight\n");
    printf("     - Conv2d backward is typically 2-3x slower than forward\n");
    printf("\n");
    printf("  2. BatchNorm has high overhead relative to its compute:\n");
    printf("     - 9 forward calls (1 stem + 2/block + 1 head)\n");
    printf("     - 3 sub-kernels per call (mean, variance, normalize)\n");
    printf("     - Backward adds 3 more sub-kernels\n");
    printf("     - Many kernel launches for small work = launch overhead\n");
    printf("\n");
    printf("  3. ReLU, GAP, Linear are fast (small tensors, simple ops)\n");
    printf("\n");
    printf("  RECOMMENDATION: Optimize Conv2d first. Two approaches:\n");
    printf("    a. Shared memory tiling (load input tile once, reuse)\n");
    printf("    b. im2col + GEMM (convert conv to matrix multiply)\n");
    printf("  See optimization_demo.cu for implementations.\n");

    printf("\n=== PROFILER COMMANDS ===\n");
    printf("\n");
    printf("For deeper analysis beyond these timings, use NVIDIA's profilers:\n");
    printf("\n");
    printf("  # Basic kernel timing (what we did above, but with all kernels)\n");
    printf("  nvprof ./profile_kernels\n");
    printf("\n");
    printf("  # Detailed per-kernel metrics\n");
    printf("  nvprof --print-gpu-trace ./profile_kernels\n");
    printf("\n");
    printf("  # Memory bandwidth analysis\n");
    printf("  nvprof --metrics gld_throughput,gst_throughput ./profile_kernels\n");
    printf("\n");
    printf("  # Occupancy and SM utilization\n");
    printf("  nvprof --metrics achieved_occupancy,sm_efficiency ./profile_kernels\n");
    printf("\n");
    printf("  # Nsight Systems timeline (visual)\n");
    printf("  nsys profile --stats=true -o profile_report ./profile_kernels\n");
    printf("  nsys-ui profile_report.nsys-rep\n");
    printf("\n");
    printf("  # Nsight Compute (detailed single-kernel analysis)\n");
    printf("  ncu --set full ./profile_kernels\n");

    // Cleanup
    clear_tensor_arena();

    printf("\nDone.\n");
    return 0;
}
