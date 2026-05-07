# Chapter 04: Error Handling and Profiling

## Overview

GPU programming introduces error handling challenges you never face on the CPU.
Kernels launch **asynchronously**, errors can be **silent**, and bugs may corrupt
memory without any immediate crash. This chapter teaches you to catch every error
and then measure your code's performance with profiling tools.

Your hardware: **Quadro P4200** (Compute Capability 6.1)
- 18 Streaming Multiprocessors (SMs)
- Memory bandwidth: ~192 GB/s (theoretical peak)
- CUDA 11.7, nvprof available

---

## 1. Why CUDA Error Handling Is Tricky

### The Asynchronous Problem

CPU code runs sequentially -- if a function fails, you know immediately.
CUDA kernels are **fire-and-forget**: the CPU launches a kernel and moves on
without waiting for it to finish.

```
CPU timeline:        GPU timeline:

cudaMalloc(...)      |
  (returns OK)       |
                     |
kernel<<<...>>>()    +-- kernel starts executing
  (returns OK!)      |   (might fail LATER)
                     |
printf("done!\n");   |   <-- CPU is here
                     |       GPU is still running!
                     |
                     +-- ERROR happens here
                         but CPU already moved on!
```

### Silent Failures

A CUDA kernel can fail silently. Without explicit error checks:
- The kernel may produce wrong results with no crash
- Errors accumulate -- later calls fail mysteriously
- Memory corruption can go undetected

```
Without error checking:

  kernel_A<<<...>>>();   // Launches with bad config (SILENT FAIL)
  kernel_B<<<...>>>();   // Launches on corrupted state (SILENT FAIL)
  cudaMemcpy(...);       // Finally fails here -- but WHY?
                         // You debug kernel_B or memcpy,
                         // but the real bug was in kernel_A!

With error checking:

  kernel_A<<<...>>>();
  CUDA_CHECK(cudaGetLastError());   // CAUGHT: "invalid configuration"
  // You immediately know kernel_A has the problem
```

---

## 2. The CUDA_CHECK Macro Pattern

Every CUDA API call returns a `cudaError_t`. The standard pattern wraps
every call in a checking macro:

```cpp
#define CUDA_CHECK(err) do {                                       \
    cudaError_t err_ = (err);                                      \
    if (err_ != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error at %s:%d -- %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err_));     \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while(0)
```

### Usage

```cpp
// API calls -- wrap directly:
CUDA_CHECK(cudaMalloc(&d_ptr, size));
CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));

// Kernel launches -- two-step:
my_kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());       // Check launch errors
CUDA_CHECK(cudaDeviceSynchronize());  // Check execution errors
```

---

## 3. cudaGetLastError vs cudaPeekAtLastError

These two functions retrieve the last error, but differ in an important way:

```
+-------------------------------+-------------------------------------------+
| cudaGetLastError()            | cudaPeekAtLastError()                     |
+-------------------------------+-------------------------------------------+
| Returns the last error        | Returns the last error                    |
| RESETS the error to cudaSuccess| Does NOT reset the error                 |
| Use for: checking & clearing  | Use for: inspecting without clearing     |
+-------------------------------+-------------------------------------------+

Example:

  bad_kernel<<<1, 99999>>>();   // Too many threads

  cudaPeekAtLastError();        // Returns "invalid configuration"
                                // Error is STILL set

  cudaPeekAtLastError();        // Returns "invalid configuration" again
                                // Error is STILL set

  cudaGetLastError();           // Returns "invalid configuration"
                                // Error is now CLEARED

  cudaGetLastError();           // Returns cudaSuccess
                                // (was cleared by previous call)
```

### When to use which?

- **cudaGetLastError()** -- Default choice. Check and clear after every operation.
- **cudaPeekAtLastError()** -- Debugging. Check error state without disturbing it
  (useful when multiple layers of code might check the same error).

---

## 4. cudaDeviceSynchronize for Debugging

Since kernels run asynchronously, `cudaGetLastError()` only catches **launch**
errors (bad grid/block dimensions, too much shared memory, etc.).

To catch **execution** errors (out-of-bounds access, illegal instructions),
you need `cudaDeviceSynchronize()` -- it blocks the CPU until all GPU work
finishes, then reports any error that occurred.

```
                        Launch errors          Execution errors
                        (caught immediately)   (caught after sync)

  kernel<<<...>>>();    - Invalid config       - Out-of-bounds access
  cudaGetLastError();   - Too many threads     - Unaligned access
                        - Too much shmem       - Illegal instruction
                                               - Stack overflow
  cudaDeviceSynchronize();                     <-- catches these
```

### Debug Mode Pattern

During development, synchronize after every kernel:

```cpp
#ifdef DEBUG
  #define CUDA_CHECK_KERNEL() do {                               \
      CUDA_CHECK(cudaGetLastError());                            \
      CUDA_CHECK(cudaDeviceSynchronize());                       \
  } while(0)
#else
  #define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
#endif
```

Compile with `-DDEBUG` during development, remove for production.
The synchronization kills performance but catches every error precisely.

---

## 5. Profiling Tools Overview

Once your code is correct, you need to make it **fast**. CUDA provides
a hierarchy of profiling tools:

```
+------------------------------------------------------------------+
|                     PROFILING TOOL HIERARCHY                      |
+------------------------------------------------------------------+
|                                                                  |
|  nvprof (command-line)           <-- Quick & easy, CUDA <= 11.x  |
|    |                                                             |
|    +-- Summary of kernel times, memory transfers                 |
|    +-- GPU activity timeline                                     |
|    +-- Hardware counter metrics                                  |
|                                                                  |
|  Nsight Systems (nsys)           <-- System-level timeline       |
|    |                                                             |
|    +-- CPU + GPU timeline together                               |
|    +-- Shows gaps, overlaps, dependencies                        |
|    +-- Best for "am I keeping the GPU busy?"                     |
|                                                                  |
|  Nsight Compute (ncu)            <-- Deep kernel analysis        |
|    |                                                             |
|    +-- Detailed per-kernel metrics                               |
|    +-- Roofline model analysis                                   |
|    +-- Memory access pattern analysis                            |
|    +-- Best for "why is this kernel slow?"                       |
|                                                                  |
+------------------------------------------------------------------+
```

### nvprof (available with CUDA 11.7)

```bash
# Basic summary -- see where time is spent:
nvprof ./profiling_basics

# Detailed GPU trace -- see every kernel launch and memcpy:
nvprof --print-gpu-trace ./profiling_basics

# Collect specific metrics:
nvprof --metrics achieved_occupancy,gld_throughput ./profiling_basics

# Save to file for later analysis:
nvprof -o profile.nvvp ./profiling_basics
# Open profile.nvvp in NVIDIA Visual Profiler (nvvp) for GUI view
```

### Nsight Systems

```bash
# Collect a system-level profile:
nsys profile --stats=true ./profiling_basics

# Generate a .qdrep file for GUI:
nsys profile -o my_profile ./profiling_basics
# Open in Nsight Systems GUI
```

### Nsight Compute

```bash
# Profile all kernels:
ncu ./profiling_basics

# Profile a specific kernel:
ncu --kernel-name compute_bound_kernel ./profiling_basics

# Full metrics collection:
ncu --set full ./profiling_basics
```

---

## 6. Key Profiling Metrics

### What to Look At

```
+---------------------+---------------------------------------------------+
| Metric              | What it tells you                                 |
+---------------------+---------------------------------------------------+
| Kernel exec time    | How long the kernel runs. Compare different       |
|                     | implementations directly.                         |
+---------------------+---------------------------------------------------+
| Memory throughput   | GB/s achieved vs. theoretical peak (192 GB/s      |
|                     | for P4200). Memory-bound kernels should approach   |
|                     | the peak.                                         |
+---------------------+---------------------------------------------------+
| Occupancy           | % of max warps active on each SM. Low occupancy   |
|                     | (< 50%) often means suboptimal block size or too   |
|                     | many registers per thread.                         |
+---------------------+---------------------------------------------------+
| SM utilization      | Are all 18 SMs busy? Low utilization means not     |
|                     | enough blocks to fill the GPU.                     |
+---------------------+---------------------------------------------------+
| Compute throughput  | GFLOPS achieved. Compare to theoretical peak      |
|                     | for your GPU.                                     |
+---------------------+---------------------------------------------------+
```

### Is my kernel memory-bound or compute-bound?

```
                 Compute-Bound                Memory-Bound

  Symptom:       High FLOPS, low BW           Low FLOPS, high BW
  Bottleneck:    ALU/FPU units                 Memory subsystem
  Fix:           Reduce computation,           Improve access patterns,
                 use faster math               use shared memory/cache

  Example:       Matrix multiply               Vector copy
                 (many FLOPs per byte)          (1 read + 1 write, no math)
```

---

## 7. The Profiling Workflow

```
  +-------------------+
  | 1. Write code     |
  |    (correct first)|
  +--------+----------+
           |
           v
  +-------------------+
  | 2. Run nvprof     |
  |    (quick summary)|
  +--------+----------+
           |
           v
  +-------------------+     Is the GPU busy?
  | 3. Nsight Systems |---> No --> Fix CPU-GPU overlap,
  |    (timeline)     |          reduce transfers
  +--------+----------+
           | Yes
           v
  +-------------------+     Which kernel is slow?
  | 4. Identify hot   |---> Focus on the kernel that
  |    kernel         |     takes the most time
  +--------+----------+
           |
           v
  +-------------------+     Memory-bound or compute-bound?
  | 5. Nsight Compute |---> Memory: fix access patterns
  |    (deep dive)    |     Compute: reduce work, use intrinsics
  +--------+----------+
           |
           v
  +-------------------+
  | 6. Optimize and   |
  |    re-measure     |----> Go back to step 2
  +-------------------+
```

---

## 8. How to Read nvprof Output

Running `nvprof ./profiling_basics` produces output like this:

```
==12345== NVPROF is profiling process 12345, command: ./profiling_basics
==12345== Profiling application: ./profiling_basics

            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   45.2%  3.245ms         1  3.245ms  3.245ms  3.245ms  memory_bound_kernel(...)
                   30.1%  2.160ms         1  2.160ms  2.160ms  2.160ms  uncoalesced_kernel(...)
                   15.3%  1.098ms         1  1.098ms  1.098ms  1.098ms  compute_bound_kernel(...)
                    9.4%  0.675ms         1  0.675ms  0.675ms  0.675ms  coalesced_kernel(...)
      API calls:   85.2%  120.5ms         4  30.12ms  15.3ms   45.2ms  cudaMalloc
                   10.3%   14.6ms         8   1.82ms  0.98ms   3.24ms  cudaMemcpy
                    4.5%    6.4ms         4   1.60ms  0.67ms   3.24ms  cudaDeviceSynchronize

Reading this output:

  Type         -- "GPU activities" = actual GPU work
                  "API calls" = CPU-side CUDA API calls

  Time(%)      -- What fraction of total GPU time
                  Focus on the LARGEST percentages first

  Time         -- Absolute wall-clock time

  Calls        -- How many times this kernel/API was called
                  High call count + high time = optimization
                  target

  Avg/Min/Max  -- Variation across calls. Large spread
                  suggests data-dependent behavior

  Name         -- Kernel or API function name
```

### Key things to notice:

1. **GPU activities vs API calls**: If API calls dominate, your bottleneck
   is data transfer or synchronization, not kernel computation.

2. **cudaMalloc taking 85%**: This is normal for short-running programs.
   cudaMalloc initializes the CUDA context on first call (~100ms).
   In real applications, this is amortized over many operations.

3. **Kernel time ratios**: Compare uncoalesced vs coalesced kernels
   to see the effect of memory access patterns.

---

## 9. CUDA Events for Timing

CUDA events provide GPU-accurate timing without host-device synchronization
overhead (beyond what you explicitly request).

```
CPU timeline           GPU timeline

cudaEventRecord(start) --> [start marker placed in GPU stream]
                           |
kernel<<<...>>>();         +-- kernel executes --+
                           |                     |
cudaEventRecord(stop)  --> [stop marker placed]  |
                                                 |
cudaEventSynchronize(stop) <-- CPU waits here ---+

cudaEventElapsedTime(&ms, start, stop)
  --> Returns GPU-measured time between markers
      (not affected by CPU delays)
```

### Usage Pattern

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);        // Place start marker
my_kernel<<<grid, block>>>();
cudaEventRecord(stop);         // Place stop marker

cudaEventSynchronize(stop);    // Wait for stop marker

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel took %.3f ms\n", ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### Why not just use clock() or gettimeofday()?

- CPU timers measure **wall clock time**, which includes CPU overhead,
  scheduling delays, and other processes.
- CUDA events measure **GPU time only**, giving you the true kernel duration.
- CPU timers also require `cudaDeviceSynchronize()` before stopping the timer,
  which adds synchronization overhead to every measurement.

---

## 10. Common Performance Pitfalls Checklist

Before optimizing, check these common issues:

```
+---+--------------------------------------------+---------------------------+
| # | Pitfall                                    | How to detect             |
+---+--------------------------------------------+---------------------------+
| 1 | Uncoalesced memory access                  | nvprof: low gld_efficiency|
|   | (threads access scattered addresses)       | or gst_efficiency         |
+---+--------------------------------------------+---------------------------+
| 2 | Too few blocks (GPU underutilized)         | nvprof: low sm_efficiency |
|   |                                            | or achieved_occupancy     |
+---+--------------------------------------------+---------------------------+
| 3 | Excessive host-device transfers             | nvprof: API calls dominate|
|   | (copying data back and forth)              | GPU activities            |
+---+--------------------------------------------+---------------------------+
| 4 | No overlap of compute and transfer          | nsys: gaps in timeline    |
|   | (sequential copy-compute-copy)             |                           |
+---+--------------------------------------------+---------------------------+
| 5 | Warp divergence                            | nvprof: branch_efficiency |
|   | (threads in same warp take different paths)|                           |
+---+--------------------------------------------+---------------------------+
| 6 | Using cudaDeviceSynchronize unnecessarily   | nsys: GPU idle periods    |
|   | (forces GPU to drain work queue)           |                           |
+---+--------------------------------------------+---------------------------+
| 7 | Small kernel launches                       | nvprof: kernel time < 10us|
|   | (launch overhead dominates)                | with many calls           |
+---+--------------------------------------------+---------------------------+
| 8 | Register spilling                          | ncu: high local memory    |
|   | (too many variables per thread)            | usage                     |
+---+--------------------------------------------+---------------------------+
```

---

## Files in This Chapter

| File | Description |
|------|-------------|
| `error_handling.cu` | CUDA_CHECK macro, intentional errors, debugging patterns |
| `profiling_basics.cu` | Four kernels with different profiles, CUDA event timing |
| `benchmark_harness.cu` | Reusable template for timing any kernel (used in later chapters) |
| `Makefile` | Build with `-lineinfo` for profiling support |

## Build and Run

```bash
make                           # Build all programs
make run                       # Build and run all
make profile                   # Run profiling_basics with nvprof

# Profiling commands:
nvprof ./profiling_basics                                  # Quick summary
nvprof --print-gpu-trace ./profiling_basics                # Detailed trace
nvprof --metrics achieved_occupancy ./profiling_basics     # Occupancy
nvprof --metrics gld_throughput,gst_throughput ./profiling_basics  # Bandwidth
```

---

## Next Chapter

Chapter 05 will cover **Shared Memory and Synchronization** -- using fast
on-chip memory to dramatically reduce global memory traffic.
