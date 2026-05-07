/*******************************************************************************
 * course_summary.cu -- The Grand Finale
 *
 * Chapter 20: Training, Profiling, and the Full Journey
 *
 * This program:
 *   1. Prints a summary of all 20 chapters and what was learned
 *   2. Runs a quick forward+backward benchmark on our ResNet
 *   3. Reports GPU stats and performance metrics
 *   4. Shows a side-by-side comparison with PyTorch
 *   5. Celebrates what we built
 *
 * This is the LAST program in the course. If you're reading this, you
 * made it through 20 chapters of CUDA programming. That's an achievement
 * worth celebrating.
 *
 * Compile: nvcc -arch=sm_61 -O2 -ccbin g++-11 -std=c++14 -lcurand course_summary.cu -o course_summary
 ******************************************************************************/

// Include our complete ResNet -- the culmination of 19 chapters of work
#include "../19_resnet_from_scratch/resnet.cuh"

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>


// ============================================================================
// Chapter Summary
// ============================================================================
//
// Each chapter built on the last. Here is what you learned, chapter by chapter.
// ============================================================================

static void print_course_summary() {
    printf("==========================================================\n");
    printf("  THE COMPLETE CUDA DEEP LEARNING COURSE\n");
    printf("  20 Chapters | 6 Parts | From Zero to ResNet\n");
    printf("==========================================================\n\n");

    printf("PART 1: CUDA FOUNDATIONS (Chapters 01-04)\n");
    printf("------------------------------------------\n");
    printf("  Ch 01: Hello GPU\n");
    printf("         Your first kernel. cudaMalloc, kernel<<<>>>, cudaMemcpy.\n");
    printf("         The moment you first ran code on the GPU.\n\n");
    printf("  Ch 02: Threads and Blocks\n");
    printf("         threadIdx, blockIdx, blockDim. Grid geometry.\n");
    printf("         How thousands of threads map to your data.\n\n");
    printf("  Ch 03: Memory Model\n");
    printf("         Global, shared, local memory. Coalescing.\n");
    printf("         Why memory access patterns matter more than compute.\n\n");
    printf("  Ch 04: Synchronization and Atomics\n");
    printf("         __syncthreads(), atomicAdd. Race conditions.\n");
    printf("         Coordinating parallel threads safely.\n\n");

    printf("PART 2: OPTIMIZATION (Chapters 05-08)\n");
    printf("--------------------------------------\n");
    printf("  Ch 05: Tiling and Shared Memory\n");
    printf("         Tiled matrix multiply. Block-level data reuse.\n");
    printf("         The single most important GPU optimization.\n\n");
    printf("  Ch 06: Occupancy and Performance\n");
    printf("         Warps, SMs, registers. Launch configuration.\n");
    printf("         Understanding the hardware to use it fully.\n\n");
    printf("  Ch 07: Streams and Events\n");
    printf("         Async execution. Overlap compute and transfer.\n");
    printf("         CUDA events for precise GPU timing.\n\n");
    printf("  Ch 08: Parallel Reduction\n");
    printf("         Tree reduction, warp shuffle, multi-level.\n");
    printf("         Reducing millions of values to one, fast.\n\n");

    printf("PART 3: GPU ALGORITHMS (Chapters 09-11)\n");
    printf("----------------------------------------\n");
    printf("  Ch 09: Scan (Prefix Sum)\n");
    printf("         Blelloch scan. Work-efficient parallel prefix.\n");
    printf("         The hidden building block of GPU computing.\n\n");
    printf("  Ch 10: Histogram and Sorting\n");
    printf("         Atomic histograms. Counting sort. Radix sort.\n");
    printf("         Organizing data in parallel.\n\n");
    printf("  Ch 11: Stencil and Convolution\n");
    printf("         1D/2D convolution. Halo regions. Separable filters.\n");
    printf("         The bridge from GPU algorithms to deep learning.\n\n");

    printf("PART 4: DL PRIMITIVES (Chapters 12-15)\n");
    printf("---------------------------------------\n");
    printf("  Ch 12: Raw CUDA Neural Net Operations\n");
    printf("         MatMul, bias add, activation kernels.\n");
    printf("         Your first neural network ops on GPU.\n\n");
    printf("  Ch 13: GPU Tensors with Gradient Tracking\n");
    printf("         GradTensor class. Data + gradient on GPU.\n");
    printf("         The foundation of automatic differentiation.\n\n");
    printf("  Ch 14: Forward Operations\n");
    printf("         Conv2d, BatchNorm, ReLU, Linear, GAP forward.\n");
    printf("         Every forward operation a ResNet needs.\n\n");
    printf("  Ch 15: Backward Operations\n");
    printf("         Gradient computation for every forward op.\n");
    printf("         The chain rule implemented in CUDA kernels.\n\n");

    printf("PART 5: DL LIBRARY (Chapters 16-17)\n");
    printf("-------------------------------------\n");
    printf("  Ch 16: Computational Graph\n");
    printf("         Topological sort. Reverse-mode autodiff.\n");
    printf("         backward() traverses the graph automatically.\n\n");
    printf("  Ch 17: Module, Optimizer, Learning Rate Schedule\n");
    printf("         Module base class. Adam optimizer. Cosine LR.\n");
    printf("         A PyTorch-like training API in C++.\n\n");

    printf("PART 6: RESNET CAPSTONE (Chapters 18-20)\n");
    printf("-----------------------------------------\n");
    printf("  Ch 18: Python ResNet Reference\n");
    printf("         Study the PyTorch ResNet architecture.\n");
    printf("         Understand pre-activation, skip connections.\n\n");
    printf("  Ch 19: ResNet from Scratch in CUDA\n");
    printf("         2000+ lines. Every kernel hand-written.\n");
    printf("         ResBlock, ResNet, CrossEntropyLoss, Adam.\n\n");
    printf("  Ch 20: Training, Profiling, and Optimization\n");
    printf("         Train on MNIST. Profile every kernel.\n");
    printf("         Optimize the hottest operation. Ship it.\n\n");
}


// ============================================================================
// Quick Benchmark
// ============================================================================
//
// Run forward + backward on random data and measure throughput.
// This gives a concrete performance number for our implementation.
// ============================================================================

static void run_benchmark() {
    printf("=== QUICK BENCHMARK ===\n\n");

    // Model configuration -- same as train_mnist.cu
    int channels   = 32;
    int num_blocks = 4;
    int fc_size    = 32;
    int num_classes = 10;
    int in_channels = 4;
    int batch_size  = 64;
    int spatial     = 8;

    // Create model
    ResNet model(channels, num_blocks, fc_size, num_classes, in_channels);
    model.train();

    int param_count = model.parameter_count();
    printf("  Model: ResNet(ch=%d, blocks=%d, fc=%d, classes=%d)\n",
           channels, num_blocks, fc_size, num_classes);
    printf("  Parameters: %d (%.1f KB)\n", param_count,
           param_count * 4.0f / 1024.0f);
    printf("  Input: (%d, %d, %d, %d)\n", batch_size, in_channels, spatial, spatial);
    printf("\n");

    // Create random data
    int img_size = in_channels * spatial * spatial;
    std::vector<float> data(batch_size * img_size);
    for (int i = 0; i < batch_size * img_size; i++) {
        data[i] = host_randn() * 0.5f;
    }
    std::vector<int> targets(batch_size);
    for (int i = 0; i < batch_size; i++) targets[i] = rand() % num_classes;

    CrossEntropyLoss criterion;

    // Warmup (5 iterations)
    printf("  Warming up...\n");
    for (int i = 0; i < 5; i++) {
        clear_tensor_arena();
        GradTensor* input = make_input(data, {batch_size, in_channels, spatial, spatial});
        GradTensor* logits = model.forward(input);
        GradTensor* loss = criterion.forward(logits, targets);
        loss->backward();
        CUDA_CHECK(cudaDeviceSynchronize());
        delete input;
    }

    // Benchmark (20 iterations)
    int iters = 20;
    printf("  Benchmarking (%d iterations)...\n\n", iters);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_fwd = 0.0f, total_bwd = 0.0f;

    for (int i = 0; i < iters; i++) {
        clear_tensor_arena();
        GradTensor* input = make_input(data, {batch_size, in_channels, spatial, spatial});

        // Forward
        cudaEventRecord(start);
        GradTensor* logits = model.forward(input);
        GradTensor* loss = criterion.forward(logits, targets);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float fwd_ms;
        cudaEventElapsedTime(&fwd_ms, start, stop);
        total_fwd += fwd_ms;

        // Backward
        cudaEventRecord(start);
        loss->backward();
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float bwd_ms;
        cudaEventElapsedTime(&bwd_ms, start, stop);
        total_bwd += bwd_ms;

        delete input;
    }

    float avg_fwd = total_fwd / iters;
    float avg_bwd = total_bwd / iters;
    float avg_total = avg_fwd + avg_bwd;
    float throughput = batch_size / (avg_total / 1000.0f);

    printf("  Results:\n");
    printf("    Forward:   %8.3f ms\n", avg_fwd);
    printf("    Backward:  %8.3f ms\n", avg_bwd);
    printf("    Total:     %8.3f ms per step\n", avg_total);
    printf("    Throughput: %.0f samples/sec\n", throughput);
    printf("    Fwd:Bwd ratio: 1 : %.2f\n", avg_bwd / avg_fwd);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    clear_tensor_arena();
    printf("\n");
}


// ============================================================================
// GPU Stats
// ============================================================================

static void print_gpu_stats() {
    printf("=== GPU INFORMATION ===\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("  Device:          %s\n", prop.name);
    printf("  Compute:         %d.%d\n", prop.major, prop.minor);
    printf("  SMs:             %d\n", prop.multiProcessorCount);
    printf("  Clock:           %d MHz\n", prop.clockRate / 1000);
    printf("  Memory clock:    %d MHz\n", prop.memoryClockRate / 1000);
    printf("  Memory bus:      %d bits\n", prop.memoryBusWidth);
    printf("  Global memory:   %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  Shared/block:    %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Registers/block: %d\n", prop.regsPerBlock);
    printf("  Warp size:       %d\n", prop.warpSize);
    printf("  Max threads/blk: %d\n", prop.maxThreadsPerBlock);
    printf("  Max block dim:   (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max grid dim:    (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // Estimate peak performance
    // Pascal GP104: 128 CUDA cores per SM
    int cores_per_sm = 128;  // GP104 (CC 6.1)
    float peak_gflops = (float)prop.multiProcessorCount * cores_per_sm *
                        2.0f * (prop.clockRate / 1e6f);
    float peak_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8.0f) / 1e6f;

    printf("  Est. peak FP32:  %.0f GFLOPS\n", peak_gflops);
    printf("  Est. peak BW:    %.0f GB/s\n", peak_bw);
    printf("  Ridge point:     %.1f FLOPS/byte\n", peak_gflops / peak_bw);

    printf("\n");
}


// ============================================================================
// Comparison Table: Our Code vs PyTorch
// ============================================================================

static void print_comparison() {
    printf("=== OUR CUDA RESNET vs PYTORCH ===\n\n");

    printf("  +----------------------------+---------------------------+---------------------------+\n");
    printf("  | Feature                     | Our CUDA ResNet           | PyTorch                   |\n");
    printf("  +----------------------------+---------------------------+---------------------------+\n");
    printf("  | Conv2d kernel               | Direct (7 nested loops)   | cuDNN (Winograd/GEMM/FFT) |\n");
    printf("  | BatchNorm                   | 4 separate kernels        | cuDNN (fused, 1 kernel)   |\n");
    printf("  | Linear (GEMM)               | Simple 1-thread-per-elem  | cuBLAS (tiled, optimized) |\n");
    printf("  | Memory management           | Manual arena + new/delete | Cached allocator + refcnt |\n");
    printf("  | Autograd graph              | Closure-based, DFS topo   | C++ engine, multi-thread  |\n");
    printf("  | Data loading                | Single-thread CPU         | Multi-process DataLoader  |\n");
    printf("  | Precision                   | FP32 only                 | AMP (FP16/BF16 + FP32)    |\n");
    printf("  | Kernel fusion               | None                      | torch.compile / Triton    |\n");
    printf("  | Multi-GPU                   | Not supported             | DDP / FSDP / NCCL         |\n");
    printf("  | Lines of code               | ~2000 (resnet.cuh)        | Millions (framework)      |\n");
    printf("  | Understanding               | COMPLETE                  | Black box to most users   |\n");
    printf("  +----------------------------+---------------------------+---------------------------+\n");
    printf("\n");
    printf("  Expected performance difference: 10-50x (depending on model size)\n");
    printf("  Our code is educational. PyTorch is production. Both are valid.\n");
    printf("  The difference: after this course, PyTorch is no longer a black box.\n");
    printf("\n");
}


// ============================================================================
// The Grand Finale
// ============================================================================

static void print_congratulations() {
    printf("==========================================================\n");
    printf("  CONGRATULATIONS!\n");
    printf("==========================================================\n\n");

    printf("  You completed the CUDA Deep Learning Course.\n\n");

    printf("  Here is what you built, from nothing:\n\n");

    printf("    - CUDA kernels for every neural network operation\n");
    printf("      conv2d, batchnorm, relu, linear, pooling, softmax\n\n");

    printf("    - A GradTensor class with automatic differentiation\n");
    printf("      Forward ops build a graph. backward() traverses it.\n\n");

    printf("    - A Module system with parameter collection\n");
    printf("      Conv2dLayer, BatchNorm2dLayer, LinearLayer, ResBlock\n\n");

    printf("    - An Adam optimizer with cosine learning rate scheduling\n");
    printf("      Bias correction, weight decay, smooth LR decay\n\n");

    printf("    - A complete ResNet (pre-activation v2)\n");
    printf("      Stem -> ResBlocks -> Head -> Classification\n\n");

    printf("    - Cross-entropy loss with numerically stable softmax\n");
    printf("      Log-sum-exp trick, clean backward gradient\n\n");

    printf("    - Training pipeline with profiling\n");
    printf("      MNIST loading, batch processing, CUDA event timing\n\n");

    printf("    - Kernel optimization techniques\n");
    printf("      Shared memory tiling, im2col + GEMM\n\n");

    printf("  Total: 20 chapters, 6 parts, thousands of lines of CUDA.\n");
    printf("  Every multiply, every gradient, every atomicAdd -- yours.\n\n");

    printf("  This is not a toy. This is a working deep learning framework\n");
    printf("  that trains a real neural network on real data.\n\n");

    printf("  You don't just USE GPU computing. You UNDERSTAND it.\n\n");

    printf("  Now go build something amazing.\n\n");

    printf("==========================================================\n");
}


// ============================================================================
// Main
// ============================================================================

int main() {
    // Print the complete course summary
    print_course_summary();

    // GPU info
    print_gpu_stats();

    // Run a quick benchmark
    run_benchmark();

    // Show the comparison table
    print_comparison();

    // The grand finale
    print_congratulations();

    return 0;
}
