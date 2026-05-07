# Chapter 01 -- Hello GPU: Your First CUDA Programs

## Table of Contents
1. [CPU vs GPU Architecture](#cpu-vs-gpu-architecture)
2. [The CUDA Execution Model](#the-cuda-execution-model)
3. [Threads, Blocks, and Grids](#threads-blocks-and-grids)
4. [CUDA Function Qualifiers](#cuda-function-qualifiers)
5. [The Kernel Launch Syntax](#the-kernel-launch-syntax)
6. [Thread Indexing](#thread-indexing)
7. [SIMT: How the GPU Actually Runs Your Code](#simt-how-the-gpu-actually-runs-your-code)
8. [Programs in This Chapter](#programs-in-this-chapter)

---

## CPU vs GPU Architecture

The fundamental difference between a CPU and a GPU is a design tradeoff:
CPUs have a **few powerful cores** optimized for sequential tasks, while
GPUs have **many simple cores** optimized for parallel throughput.

```
  CPU (e.g., Intel i7 -- 8 cores)            GPU (e.g., Quadro P4000 -- 1792 cores)
  ================================            ========================================

  +--------+  +--------+                     +--+--+--+--+--+--+--+--+--+--+--+--+--+
  | Core 0 |  | Core 1 |                     |  |  |  |  |  |  |  |  |  |  |  |  |  |
  | [ALU ] |  | [ALU ] |                     +--+--+--+--+--+--+--+--+--+--+--+--+--+
  | [ALU ] |  | [ALU ] |                     |  |  |  |  |  |  |  |  |  |  |  |  |  |
  | [FPU ] |  | [FPU ] |                     +--+--+--+--+--+--+--+--+--+--+--+--+--+
  | [BPU ] |  | [BPU ] |  BPU = Branch       |  |  |  |  |  |  |  |  |  |  |  |  |  |
  | [OoO ] |  | [OoO ] |  Prediction Unit    +--+--+--+--+--+--+--+--+--+--+--+--+--+
  | [Cache] |  | [Cache] |  OoO = Out of      |  |  |  |  |  |  |  |  |  |  |  |  |  |
  +--------+  +--------+  Order Execution    +--+--+--+--+--+--+--+--+--+--+--+--+--+
  +--------+  +--------+                     |  |  |  |  |  |  |  |  |  |  |  |  |  |
  | Core 2 |  | Core 3 |                     +--+--+--+--+--+--+--+--+--+--+--+--+--+
  | [ALU ] |  | [ALU ] |                     |  |  |  |  |  |  |  |  |  |  |  |  |  |
  | [ALU ] |  | [ALU ] |                     +--+--+--+--+--+--+--+--+--+--+--+--+--+
  | [FPU ] |  | [FPU ] |                     |  |  |  |  |  |  |  |  |  |  |  |  |  |
  | [BPU ] |  | [BPU ] |                     +--+--+--+--+--+--+--+--+--+--+--+--+--+
  | [OoO ] |  | [OoO ] |                     |  |  |  |  |  |  |  |  |  |  |  |  |  |
  | [Cache] |  | [Cache] |                     +--+--+--+--+--+--+--+--+--+--+--+--+--+
  +--------+  +--------+                       ... hundreds more rows of tiny cores ...

  Each CPU core is a                          Each GPU core is a simple
  sophisticated machine                       ALU -- just does math.
  with deep pipelines,                        No branch prediction,
  branch prediction,                          no out-of-order execution.
  speculative execution,                      But there are THOUSANDS
  large caches.                               of them.

  LATENCY optimized                           THROUGHPUT optimized
  (do one thing FAST)                         (do many things AT ONCE)
```

### Why does this matter for computing?

Many problems in science, ML, and graphics involve doing the **same operation**
on millions of data points. Adding two vectors of 10 million floats? Each
addition is independent. A CPU does them one (or a few) at a time. A GPU does
thousands at a time.

```
  CPU approach:  Sequential (or slightly parallel with SIMD)

  Time -->
  +----+----+----+----+----+----+----+----+----+----+----+----+----+---
  | a0 | a1 | a2 | a3 | a4 | a5 | a6 | a7 | a8 | a9 | .. | .. | ..
  +----+----+----+----+----+----+----+----+----+----+----+----+----+---


  GPU approach:  Massively parallel

  Time -->
  +----+
  | a0 |   All done in one step (conceptually).
  | a1 |   In practice, a few steps if N > num_cores,
  | a2 |   but still orders of magnitude faster.
  | a3 |
  | a4 |
  | .. |
  |a999|
  +----+
```

---

## The CUDA Execution Model

CUDA programs run on two processors: the **host** (CPU) and the **device** (GPU).
The host code is regular C/C++. The device code runs on the GPU in parallel.

```
  +---------------------+          +---------------------------+
  |       HOST           |          |         DEVICE             |
  |       (CPU)          |          |         (GPU)              |
  |                     |          |                           |
  |  int main() {       |          |                           |
  |    // allocate mem   | -------> |  Global Memory (VRAM)     |
  |    // copy data to   | -------> |  [d_a] [d_b] [d_c]       |
  |    //   GPU          |          |                           |
  |                     |          |                           |
  |    // launch kernel  | =======> |  kernel<<<...>>>()        |
  |    //   (async!)     |          |    thread 0: c[0]=a[0]+.. |
  |                     |          |    thread 1: c[1]=a[1]+.. |
  |    // copy results   | <------- |    thread 2: c[2]=a[2]+.. |
  |    //   back         |          |    ...                    |
  |    // free GPU mem   |          |                           |
  |  }                  |          |                           |
  +---------------------+          +---------------------------+

  Legend:
    -------> = cudaMemcpy (Host to Device)
    <------- = cudaMemcpy (Device to Host)
    =======> = kernel launch (asynchronous!)
```

### The typical CUDA workflow:

1. **Allocate** memory on the GPU with `cudaMalloc()`
2. **Copy** input data from host to device with `cudaMemcpy()`
3. **Launch** a kernel (GPU function) with `<<<blocks, threads>>>`
4. **Copy** results back from device to host with `cudaMemcpy()`
5. **Free** GPU memory with `cudaFree()`

This is sometimes called the "copy-compute-copy" pattern.

---

## Threads, Blocks, and Grids

CUDA organizes parallel execution into a three-level hierarchy:

```
                              GRID
         (the entire collection of threads for one kernel launch)
  +------------------------------------------------------------------+
  |                                                                  |
  |   Block(0,0)        Block(1,0)        Block(2,0)                |
  |  +--------------+  +--------------+  +--------------+           |
  |  | T0  T1  T2   |  | T0  T1  T2   |  | T0  T1  T2   |           |
  |  | T3  T4  T5   |  | T3  T4  T5   |  | T3  T4  T5   |           |
  |  | T6  T7  ...  |  | T6  T7  ...  |  | T6  T7  ...  |           |
  |  +--------------+  +--------------+  +--------------+           |
  |                                                                  |
  |   Block(0,1)        Block(1,1)        Block(2,1)                |
  |  +--------------+  +--------------+  +--------------+           |
  |  | T0  T1  T2   |  | T0  T1  T2   |  | T0  T1  T2   |           |
  |  | T3  T4  T5   |  | T3  T4  T5   |  | T3  T4  T5   |           |
  |  | T6  T7  ...  |  | T6  T7  ...  |  | T6  T7  ...  |           |
  |  +--------------+  +--------------+  +--------------+           |
  |                                                                  |
  +------------------------------------------------------------------+

  Grid dimensions:  gridDim.x = 3, gridDim.y = 2
  Block dimensions: blockDim.x = 3, blockDim.y = 3 (9 threads per block)
```

### Key concepts:

- **Thread**: The smallest unit of execution. Each thread runs the kernel code.
- **Block**: A group of threads that can cooperate via shared memory and
  synchronize with `__syncthreads()`. Max 1024 threads per block.
- **Grid**: The collection of all blocks for a kernel launch.

### Why blocks?

Blocks map to **Streaming Multiprocessors (SMs)** on the GPU. Your Quadro P4000
has 14 SMs. Each SM can run multiple blocks concurrently. Blocks are independent --
they can execute in any order, which is how the GPU scales across different
hardware.

```
  GPU Hardware Mapping (Quadro P4000 example)
  ============================================

  +-------+  +-------+  +-------+  +-------+     +-------+
  | SM  0 |  | SM  1 |  | SM  2 |  | SM  3 | ... | SM 13 |
  +-------+  +-------+  +-------+  +-------+     +-------+
  |Block 0|  |Block 1|  |Block 2|  |Block 3|     |Block13|
  |Block14|  |Block15|  |Block16|  |Block17|     |Block27|
  |  ...  |  |  ...  |  |  ...  |  |  ...  |     |  ...  |
  +-------+  +-------+  +-------+  +-------+     +-------+

  Blocks are assigned to SMs by the hardware scheduler.
  When one block finishes, the SM picks up another.
  This is how the GPU keeps all SMs busy.
```

---

## CUDA Function Qualifiers

CUDA extends C/C++ with three function qualifiers:

```
  +------------------+-------------------+-------------------+
  | Qualifier        | Runs on           | Called from       |
  +------------------+-------------------+-------------------+
  | __global__       | Device (GPU)      | Host (CPU)        |
  |                  |                   | or Device (GPU)   |
  +------------------+-------------------+-------------------+
  | __device__       | Device (GPU)      | Device (GPU)      |
  +------------------+-------------------+-------------------+
  | __host__         | Host (CPU)        | Host (CPU)        |
  +------------------+-------------------+-------------------+

  __global__ = "kernel" -- this is the entry point launched from the CPU.
               Must return void. Called with <<<>>> syntax.

  __device__ = helper function that runs on the GPU.
               Called by kernels or other __device__ functions.
               Cannot be called from the CPU.

  __host__   = normal CPU function (this is the default, so it is
               usually omitted). Can be combined with __device__
               to compile for both CPU and GPU:

               __host__ __device__ float square(float x) { return x*x; }
               // This function works on both CPU and GPU!
```

---

## The Kernel Launch Syntax

```cpp
kernel_name<<<numBlocks, threadsPerBlock>>>(arg1, arg2, ...);
```

```
                    <<<numBlocks, threadsPerBlock>>>
                         |              |
                         v              v
                   How many blocks   How many threads
                   in the grid?      in each block?

  Example: kernel<<<4, 256>>>(data, N);

  This launches:
    4 blocks x 256 threads/block = 1024 threads total

  Visualized:
  +----------+ +----------+ +----------+ +----------+
  | Block 0  | | Block 1  | | Block 2  | | Block 3  |
  | 256 thds | | 256 thds | | 256 thds | | 256 thds |
  +----------+ +----------+ +----------+ +----------+
```

### Common launch configurations:

```
  <<<1, 1>>>        1 block, 1 thread     (sequential -- defeats the purpose!)
  <<<1, 256>>>      1 block, 256 threads  (256 threads total)
  <<<10, 256>>>     10 blocks, 256 thds   (2560 threads total)
  <<<N/256, 256>>>  Enough blocks to      (N threads total, one per element)
                    cover N elements
```

### Why 256 threads per block?

256 (or 128, or 512) are common choices because:
- Must be a multiple of 32 (warp size -- see SIMT section below)
- Max is 1024 threads per block
- 256 is a good default that gives the GPU enough threads to hide latency

---

## Thread Indexing

Every thread knows where it is in the grid via built-in variables:

```
  threadIdx.x  -- thread's index WITHIN its block (0 to blockDim.x - 1)
  blockIdx.x   -- block's index WITHIN the grid   (0 to gridDim.x - 1)
  blockDim.x   -- number of threads per block
  gridDim.x    -- number of blocks in the grid
```

### Computing a global thread index:

```
  globalIndex = blockIdx.x * blockDim.x + threadIdx.x

  Example: <<<4, 8>>> (4 blocks of 8 threads)

  Block 0          Block 1          Block 2          Block 3
  threadIdx: 0-7   threadIdx: 0-7   threadIdx: 0-7   threadIdx: 0-7
  blockIdx:  0     blockIdx:  1     blockIdx:  2     blockIdx:  3

  Global index calculation:
  +----+----+----+----+----+----+----+----+----+----+----+----+---
  |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |  9 | 10 | 11 | ...
  +----+----+----+----+----+----+----+----+----+----+----+----+---
  |<--------- Block 0 -------->|<--------- Block 1 -------->| ...

  Thread at blockIdx=2, threadIdx=3:
    global = 2 * 8 + 3 = 19
```

### The stride for grid-stride loops:

```
  stride = blockDim.x * gridDim.x   (total number of threads in the grid)

  This is used in the grid-stride loop pattern:

  for (int i = globalIndex; i < N; i += stride) {
      // process element i
  }

  Why? Because often N >> total threads. Each thread processes
  multiple elements by striding through the array:

  N = 32, grid has 8 threads total:

  Thread 0: processes elements 0, 8, 16, 24
  Thread 1: processes elements 1, 9, 17, 25
  Thread 2: processes elements 2, 10, 18, 26
  ...
  Thread 7: processes elements 7, 15, 23, 31

  This pattern is flexible: it works regardless of N and grid size.
  It also has good memory access patterns (coalesced access).
```

---

## SIMT: How the GPU Actually Runs Your Code

SIMT = **Single Instruction, Multiple Threads**

The GPU doesn't execute threads completely independently. It groups threads
into **warps** of 32 threads. All 32 threads in a warp execute the **same
instruction** at the **same time** (on different data).

```
  A warp of 32 threads executing the same add instruction:
  +-----------------------------------------------------------------+
  | INSTRUCTION: add r1, r2, r3                                     |
  +-----------------------------------------------------------------+
  | T0: r1=a[0]+b[0] | T1: r1=a[1]+b[1] | ... | T31: r1=a[31]+b[31]|
  +-----------------------------------------------------------------+
                    All 32 happen simultaneously

  Block of 256 threads = 8 warps:
  +--------+--------+--------+--------+--------+--------+--------+--------+
  | Warp 0 | Warp 1 | Warp 2 | Warp 3 | Warp 4 | Warp 5 | Warp 6 | Warp 7 |
  | T0-T31 | T32-63 | T64-95 | T96-127|T128-159|T160-191|T192-223|T224-255|
  +--------+--------+--------+--------+--------+--------+--------+--------+
```

### Why warps matter:

1. **Thread count should be a multiple of 32.** If you launch 33 threads,
   you get 2 warps (64 thread slots), with 31 slots wasted.

2. **Branch divergence is expensive.** If threads in a warp take different
   paths in an if/else, both paths are executed serially:

```
   if (threadIdx.x < 16) {    // Threads 0-15 take this path
       doA();                  // Warp executes doA(), threads 16-31 idle
   } else {                   // Threads 16-31 take this path
       doB();                  // Warp executes doB(), threads 0-15 idle
   }
   // Total time: doA() + doB()   (not max(doA, doB) as you might hope)
```

3. **Memory coalescing.** When threads in a warp access consecutive memory
   addresses, the hardware combines them into fewer memory transactions:

```
   GOOD (coalesced):                  BAD (strided):
   T0 -> a[0]                        T0 -> a[0]
   T1 -> a[1]     => 1 transaction   T1 -> a[128]   => 32 transactions!
   T2 -> a[2]                        T2 -> a[256]
   ...                               ...
   T31-> a[31]                       T31-> a[31*128]
```

---

## Programs in This Chapter

### 1. hello_gpu.cu
Your absolute first CUDA program. A kernel that prints from the GPU.
Demonstrates basic launch configurations.

```bash
make hello_gpu
./hello_gpu
```

### 2. vector_add.cu
The classic first "real" CUDA program. Adds two vectors on the GPU.
Demonstrates the full copy-compute-copy workflow, timing, error checking,
and the grid-stride loop pattern.

```bash
make vector_add
./vector_add
```

### 3. vector_add_unified.cu
Same vector addition but using CUDA Unified Memory (`cudaMallocManaged`).
Shows how unified memory simplifies the code at the cost of some
performance considerations.

```bash
make vector_add_unified
./vector_add_unified
```

### Build all:
```bash
make          # builds all three
make clean    # removes binaries
```

---

## Exercises

1. Modify `hello_gpu.cu` to launch `<<<4, 64>>>`. How many threads print?
2. In `vector_add.cu`, try changing the block size from 256 to 32, 128, 512.
   Does performance change? Why or why not?
3. In `vector_add.cu`, remove the bounds check (`if (i < N)`). What happens
   when N is not a multiple of blockDim?
4. Compare the timing of `vector_add.cu` vs `vector_add_unified.cu`. Which is
   faster on first run? What about subsequent runs?
5. Add error checking to `hello_gpu.cu` using `cudaGetLastError()`.

---

## Quick Reference

```
  cudaMalloc(&ptr, size)                  -- allocate GPU memory
  cudaFree(ptr)                           -- free GPU memory
  cudaMemcpy(dst, src, size, direction)   -- copy between host/device
    directions: cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost
  cudaMallocManaged(&ptr, size)           -- allocate unified memory
  cudaDeviceSynchronize()                 -- wait for GPU to finish
  cudaGetLastError()                      -- check for kernel errors
```
