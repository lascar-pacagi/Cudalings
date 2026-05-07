# Chapter 20: Training, Profiling, and the Full Journey

## The Final Chapter

This is it. Twenty chapters. From `printf("Hello from GPU!\n")` to a working
ResNet with autograd, batch normalization, residual connections, Adam optimizer,
and cosine learning rate scheduling -- all in raw CUDA C++.

This chapter ties everything together:
1. Train our ResNet on real data (MNIST)
2. Profile the training pipeline kernel by kernel
3. Apply optimizations to the hottest kernels
4. Reflect on what we built and where to go next


## The Full Journey: Chapter 01 to Chapter 20

```
THE CUDA DEEP LEARNING COURSE -- ALL 20 CHAPTERS
=================================================

PART 1: FOUNDATIONS (Chapters 01-04)
------------------------------------
Ch01: Hello GPU         Ch02: Threads/Blocks     Ch03: Memory Model       Ch04: Sync & Atomics
  |                       |                        |                        |
  | cudaMalloc            | threadIdx/blockIdx     | Global/Shared/Local    | __syncthreads()
  | kernel<<<>>>          | Grid geometry          | Coalescing             | atomicAdd
  | cudaMemcpy            | 1D/2D/3D launch        | Bank conflicts         | Race conditions
  v                       v                        v                        v
  "I can run code    "I understand how        "I know WHERE data     "I can coordinate
   on the GPU"        threads are organized"    lives and WHY"         parallel work"

                                    |
                                    v

PART 2: OPTIMIZATION (Chapters 05-08)
--------------------------------------
Ch05: Tiling             Ch06: Occupancy          Ch07: Streams/Events     Ch08: Reduction
  |                       |                        |                        |
  | Shared mem tiles      | Warps, SMs             | Async execution        | Tree reduction
  | Matrix multiply       | Registers vs shared    | Overlap compute+copy   | Warp shuffle
  | Block-level reuse     | Launch config tuning   | CUDA events timing     | Multi-level
  v                       v                        v                        v
  "I can make kernels "I understand the       "I can overlap work    "I can reduce
   memory-efficient"   hardware limits"        and measure time"      millions to one"

                                    |
                                    v

PART 3: GPU ALGORITHMS (Chapters 09-11)
----------------------------------------
Ch09: Scan (Prefix Sum)  Ch10: Histogram/Sort     Ch11: Stencil/Conv
  |                       |                        |
  | Blelloch scan         | Atomic histogram       | 1D/2D convolution
  | Work-efficient        | Radix sort             | Halo regions
  | Inclusive/exclusive   | Counting sort          | Separable filters
  v                       v                        v
  "I can do parallel   "I can organize and    "I can apply local
   prefix operations"   sort GPU data"          neighborhood ops"

                                    |
                                    v

PART 4: DL PRIMITIVES (Chapters 12-15)
---------------------------------------
Ch12: Raw CUDA NN Ops    Ch13: GPU Tensors+Grad   Ch14: Forward Ops        Ch15: Backward Ops
  |                       |                        |                        |
  | MatMul kernel         | GradTensor class       | Conv2d forward         | Conv2d backward
  | Bias add              | GPU alloc/free         | BatchNorm forward      | BatchNorm backward
  | Activation kernels    | Shape tracking         | ReLU, Linear, GAP      | Chain rule on GPU
  v                       v                        v                        v
  "I wrote my first    "Data + gradient       "I implemented every   "I can compute
   neural net ops"      live on the GPU"        forward operation"     gradients for all"

                                    |
                                    v

PART 5: DL LIBRARY (Chapters 16-17)
-------------------------------------
Ch16: Computational Graph   Ch17: Module/Optim/LR
  |                           |
  | Topological sort          | Module base class
  | Reverse-mode AD           | Adam optimizer
  | backward() traversal      | Cosine LR schedule
  | Closure-based autograd    | Parameter collection
  v                           v
  "Gradients flow          "I have a PyTorch-like
   automatically"            training API"

                                    |
                                    v

PART 6: RESNET CAPSTONE (Chapters 18-20)
-----------------------------------------
Ch18: Python ResNet       Ch19: CUDA ResNet         Ch20: TRAIN + PROFILE
  |                        |                          |
  | Architecture study     | All-in-one resnet.cuh    | MNIST training
  | Pre-activation v2      | ResBlock, ResNet class   | Kernel profiling
  | PyTorch reference      | CrossEntropyLoss         | Optimization demo
  v                        v                          v
  "I understand the     "I built it all          "IT WORKS. I trained
   target architecture"   from scratch in CUDA"     a real neural net
                                                     on real data with
                                                     code I wrote myself."
```


## Where Time Is Spent in a Training Step

A single training step has five phases. Here is where the wall-clock time
goes in a typical DL training pipeline:

```
ONE TRAINING STEP
=================

  +------------------+     +------------------+     +------------------+
  |  DATA LOADING    |     |    FORWARD       |     |      LOSS        |
  |                  |     |                  |     |                  |
  | - Read from disk | --> | - Conv2d         | --> | - Softmax        |
  | - Preprocess     |     | - BatchNorm      |     | - Log-likelihood |
  | - H2D transfer   |     | - ReLU           |     | - Reduce mean    |
  |                  |     | - Linear         |     |                  |
  | ~5-20% of time   |     | - GAP            |     | ~1-2% of time    |
  +------------------+     | ~30-40% of time  |     +------------------+
                           +------------------+              |
                                                             v
  +------------------+     +------------------+     +------------------+
  |  OPTIMIZER       |     |    BACKWARD      |     |   LOSS BACKWARD  |
  |                  |     |                  |     |                  |
  | - Adam moments   | <-- | - d_Conv2d       | <-- | - d_logits =     |
  | - Bias correction|     | - d_BatchNorm    |     |   softmax-onehot |
  | - Param update   |     | - d_ReLU         |     |                  |
  | - Weight decay   |     | - d_Linear       |     | ~1-2% of time    |
  |                  |     | - d_GAP          |     +------------------+
  | ~5-10% of time   |     |                  |
  +------------------+     | ~40-50% of time  |
                           +------------------+

  KEY INSIGHT: Backward is typically MORE expensive than forward because:
    1. Conv2d backward computes BOTH d_input and d_weight (two kernels)
    2. BatchNorm backward has 3 sub-kernels (params, sums, input)
    3. All gradients must be ACCUMULATED (atomicAdd overhead)
```


## Profiling Methodology

### Step 1: High-Level Timing with CUDA Events

CUDA events measure GPU-side time without CPU synchronization overhead.
This is what `train_mnist.cu` uses to time each phase.

```
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... GPU work ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
```

### Step 2: Kernel-Level Profiling with nvprof/nsys

Commands for the Quadro P4200:

```bash
# nvprof: legacy profiler (CUDA 11.7 supports it)
# Shows kernel execution times, memory transfers, occupancy
nvprof ./profile_kernels

# Detailed kernel metrics
nvprof --print-gpu-trace ./profile_kernels

# Memory bandwidth utilization
nvprof --metrics gld_throughput,gst_throughput,dram_read_throughput,dram_write_throughput ./profile_kernels

# Occupancy analysis
nvprof --metrics achieved_occupancy,sm_efficiency ./profile_kernels

# nsys (Nsight Systems): timeline view
nsys profile --stats=true ./profile_kernels

# nsys with CUDA API trace
nsys profile -t cuda,nvtx --stats=true -o profile_report ./profile_kernels
# Then view with: nsys-ui profile_report.qdrep
```

### What to Look For

| Metric | Good | Bad | Fix |
|--------|------|-----|-----|
| Occupancy | > 50% | < 25% | Reduce registers/shared mem |
| SM efficiency | > 80% | < 50% | More parallelism |
| DRAM throughput | Near peak | < 30% peak | Coalesce accesses |
| Kernel duration | Steady | Varies wildly | Check branch divergence |
| H2D/D2H overlap | Overlapped | Sequential | Use streams (Ch07) |


## Optimization Checklist (Referencing Earlier Chapters)

```
OPTIMIZATION CHECKLIST FOR OUR RESNET
======================================

From Ch03 (Memory):
  [ ] Memory accesses are coalesced (NCHW layout helps)
  [ ] No unnecessary global memory reads (reuse in registers)
  [ ] Shared memory used for frequently-accessed data

From Ch04 (Synchronization):
  [ ] atomicAdd minimized (only where truly needed)
  [ ] No redundant __syncthreads()

From Ch05 (Tiling):
  [ ] Conv2d could use shared memory tiles for input reuse
  [ ] Linear layer could use tiled matrix multiply

From Ch06 (Occupancy):
  [ ] Block size 256 is reasonable for SM_61
  [ ] Register usage checked (nvcc --ptxas-options=-v)

From Ch07 (Streams):
  [ ] Data loading can overlap with forward pass
  [ ] Multiple batches can pipeline

From Ch08 (Reduction):
  [ ] BatchNorm reductions use simple loops (OK for small spatial)
  [ ] GAP reduction is simple loop (OK for 8x8)
  [ ] Cross-entropy reduction is single-thread (OK for batch<=256)
```


## Roofline Model for Our Kernels

The Quadro P4200 (GP104GL, CC 6.1):
- Peak FP32: ~2400 GFLOPS (boost clock dependent)
- Memory bandwidth: ~192 GB/s (GDDR5)
- L2 cache: 2 MB

```
ROOFLINE MODEL
==============

Performance (GFLOPS)
    ^
    |
2400|...................................... Peak Compute
    |                               .'
    |                          .'
    |                     .'
    |                .'    <-- Memory bound region
    |           .'              (most of our kernels)
    |      .'
    | .'
    +-----------------------------------------> Arithmetic Intensity
    0    1    2    4    8   12.5  (FLOPS/byte)
              ^
              |
         Ridge point = 2400 / 192 = 12.5 FLOPS/byte

OUR KERNELS:
  Conv2d (3x3):  ~4.5 FLOPS/byte  --> MEMORY BOUND
  BatchNorm:     ~1-2 FLOPS/byte  --> MEMORY BOUND
  ReLU:          ~0.25 FLOPS/byte --> MEMORY BOUND
  Linear:        ~varies (small N --> memory bound)
  GAP:           ~0.5 FLOPS/byte  --> MEMORY BOUND
  Adam:          ~3 FLOPS/byte    --> MEMORY BOUND

INSIGHT: Almost ALL our kernels are memory-bound. This means:
  1. Coalesced access patterns matter more than compute tricks
  2. Reducing global memory traffic (tiling, fusion) is key
  3. cuDNN wins by fusing operations and using tensor cores (on newer GPUs)
```


## Our Framework vs PyTorch

| Aspect | Our CUDA ResNet | PyTorch |
|--------|----------------|---------|
| Conv2d | Direct (7-loop) | cuDNN (Winograd/FFT/implicit GEMM) |
| BatchNorm | 4 separate kernels | cuDNN (fused, optimized reductions) |
| Linear | Simple kernel | cuBLAS GEMM (highly tuned) |
| Memory | Manual arena | Reference-counted autograd graph |
| Data loading | Single-threaded CPU | Multi-process DataLoader |
| Mixed precision | FP32 only | AMP (FP16/BF16 + FP32 master) |
| Kernel fusion | None | torch.compile / Triton |
| Expected speed | 1x (baseline) | 10-50x faster |

**Why the gap?** Our kernels are educational -- each operation is one kernel
launch with a simple thread-per-element strategy. Production frameworks:

1. **cuDNN**: Uses Winograd (2.25x fewer multiplies for 3x3), implicit GEMM,
   or FFT-based convolution depending on sizes. Pre-tuned for each GPU arch.

2. **cuBLAS**: GEMM kernels hand-tuned in SASS (assembly), with block-level
   tiling, warp-level matrix fragments, and double-buffered loads.

3. **Kernel fusion**: PyTorch/TensorRT fuse BN+ReLU, Conv+BN+ReLU into single
   kernels. This eliminates intermediate global memory reads/writes.

4. **Mixed precision**: FP16 has 2x bandwidth, and tensor cores provide
   8x throughput on Volta+ (our P4200 lacks tensor cores, but FP16 still
   helps bandwidth).

5. **Memory management**: CUDA memory pools (cudaMallocAsync) eliminate
   malloc overhead. PyTorch caches allocations.


## What Production Frameworks Do Differently

```
PRODUCTION DL STACK
===================

Application Layer:    PyTorch / TensorFlow / JAX
                          |
Graph Optimization:   torch.compile / XLA / TensorRT
                          |
                     +----+----+
                     |         |
Math Libraries:   cuDNN     cuBLAS    (hand-tuned, arch-specific)
                     |         |
                     +----+----+
                          |
Runtime:             CUDA Driver / Runtime API
                          |
Hardware:            GPU (SMs, Tensor Cores, Memory Controllers)

What we built:
  - We go directly from Application -> CUDA Runtime -> GPU
  - No cuDNN, no cuBLAS, no graph optimization
  - But we understand EVERY layer of the stack
```


## Where to Go Next

After completing this course, you have the foundation to:

1. **Use cuDNN/cuBLAS** -- Now you understand WHY they exist and what they
   optimize. Read the cuDNN developer guide with fresh eyes.

2. **Write custom CUDA kernels** -- When PyTorch doesn't have what you need,
   you can write it. Custom attention, novel activations, specialized losses.

3. **Profile real workloads** -- You know what to measure, what metrics matter,
   and how to interpret roofline analysis.

4. **Understand papers** -- When a paper says "we fuse the backward pass of
   LayerNorm with the residual connection," you know exactly what that means.

5. **Explore advanced topics**:
   - Tensor cores and mixed precision (Volta+)
   - Multi-GPU with NCCL (collective communications)
   - Flash Attention (tiled softmax + matmul fusion)
   - Quantization (INT8/INT4 inference)
   - CUTLASS (templated GEMM library)
   - Triton (Python -> GPU compiler)

6. **Contribute to open source** -- PyTorch, vLLM, llama.cpp, GGML all
   need people who understand GPU programming at this level.

You didn't just learn CUDA. You built a deep learning framework from scratch.
That understanding is permanent.
