# Chapter 02 -- The CUDA Memory Model

## Table of Contents
1. [Why Memory Matters More Than Compute](#why-memory-matters-more-than-compute)
2. [The CUDA Memory Hierarchy](#the-cuda-memory-hierarchy)
3. [Registers](#registers)
4. [Shared Memory](#shared-memory)
5. [Global Memory](#global-memory)
6. [Constant Memory](#constant-memory)
7. [Local Memory (Register Spill)](#local-memory-register-spill)
8. [Texture Memory](#texture-memory)
9. [Memory Visibility by Scope](#memory-visibility-by-scope)
10. [Memory Bandwidth -- The Key Metric](#memory-bandwidth----the-key-metric)
11. [Arithmetic Intensity and the Roofline Model](#arithmetic-intensity-and-the-roofline-model)
12. [Programs in This Chapter](#programs-in-this-chapter)

---

## Why Memory Matters More Than Compute

Here is the single most important idea in GPU programming:

**Most GPU kernels are bottlenecked by memory, not by math.**

Your Quadro P4000 can do ~5.3 TFLOPS of single-precision math.
But its memory can only deliver ~192 GB/s of data.

A simple `c[i] = a[i] + b[i]` does 1 FLOP but moves 12 bytes (read 2 floats,
write 1 float). That is an arithmetic intensity of 1/12 = 0.083 FLOP/byte.
At 192 GB/s, the memory system can feed only 192e9 / 12 = 16 billion elements
per second. That is 16 GFLOPS -- a tiny fraction of the 5300 GFLOPS peak.

The memory hierarchy exists to bridge this gap. Use it wisely and your code
runs fast. Ignore it and your GPU sits idle waiting for data.

---

## The CUDA Memory Hierarchy

```
  CUDA Memory Hierarchy (from fastest/smallest to slowest/largest)
  ================================================================

                         Per-Thread
                        +-----------+
                        | REGISTERS |  <-- ~1 cycle latency
                        |  64K per  |      ~8 TB/s bandwidth (on-chip)
                        |    SM     |      Fastest. Each thread gets its own.
                        +-----------+
                              |
                         Per-Block
                    +------------------+
                    |  SHARED MEMORY   |  <-- ~5-30 cycles latency
                    | 48 KB per SM     |      ~1.5 TB/s bandwidth (on-chip)
                    | (configurable)   |      User-managed scratchpad.
                    +------------------+      Like a programmable L1 cache.
                              |
                         Per-SM
                    +------------------+
                    | L1 CACHE / TEX   |  <-- ~30-70 cycles latency
                    | 48 KB per SM     |      Hardware-managed.
                    | (unified w/      |      Shares physical SRAM with
                    |  shared mem)     |      shared memory on Pascal.
                    +------------------+
                              |
                         Per-Device
                    +------------------+
                    |     L2 CACHE     |  <-- ~200 cycles latency
                    |    2 MB total    |      Sits between SMs and DRAM.
                    |  (all SMs share) |      Hardware-managed.
                    +------------------+
                              |
                         Per-Device
          +--------------------------------------+
          |          GLOBAL MEMORY (VRAM)         |  <-- 400-800 cycles latency
          |           8 GB GDDR5                  |      ~192 GB/s bandwidth
          |    (also: constant, texture caches    |      Off-chip DRAM.
          |     are cached views into this)       |      Large but SLOW.
          +--------------------------------------+
                              |
                         PCIe Bus
          +--------------------------------------+
          |           HOST MEMORY (RAM)           |  <-- ~10,000+ cycles
          |          System RAM (CPU)             |      ~12 GB/s (PCIe 3.0 x16)
          +--------------------------------------+      Slowest. Avoid accessing
                                                        from GPU if possible.

  BANDWIDTH COMPARISON:
  =====================

  Registers      ||||||||||||||||||||||||||||||||||||||||||||  ~8,000 GB/s
  Shared Mem     |||||||||||||||||||||                         ~1,500 GB/s
  L2 Cache       ||||||                                       ~500 GB/s (est.)
  Global (DRAM)  |||                                          ~192 GB/s
  PCIe (Host)    |                                            ~12 GB/s

  Each level is roughly 5-10x slower than the one above!
```

---

## Registers

Registers are the fastest storage on the GPU. Each thread has its own
private set of registers. On CC 6.1 (your GPU), each SM has 65,536
32-bit registers shared among all active threads on that SM.

```
  Registers: per-thread, private
  ===============================

  +-- SM 0 ------------------------------------------------+
  |                                                         |
  |  Block A (256 threads)    Block B (256 threads)         |
  |  +----+----+----+---+    +----+----+----+---+          |
  |  |T0  |T1  |T2  |...|    |T0  |T1  |T2  |...|          |
  |  |r0  |r0  |r0  |   |    |r0  |r0  |r0  |   |          |
  |  |r1  |r1  |r1  |   |    |r1  |r1  |r1  |   |          |
  |  |r2  |r2  |r2  |   |    |r2  |r2  |r2  |   |          |
  |  |... |... |... |   |    |... |... |... |   |          |
  |  +----+----+----+---+    +----+----+----+---+          |
  |                                                         |
  |  Total: 65,536 registers shared across all threads      |
  +---------------------------------------------------------+

  If your kernel uses 32 registers per thread:
    65536 / 32 = 2048 threads max per SM

  If your kernel uses 64 registers per thread:
    65536 / 64 = 1024 threads max per SM  (reduced occupancy!)

  If your kernel uses 128 registers per thread:
    65536 / 128 = 512 threads max per SM  (even less occupancy!)
```

**Key facts about registers:**
- Local variables in your kernel are placed in registers (by the compiler)
- ~1 cycle access latency -- as fast as it gets
- Private to each thread -- no other thread can see them
- Limited supply: use too many and occupancy drops (fewer concurrent threads)
- When you run out, variables "spill" to local memory (slow! see below)

In CUDA code, registers are just local variables:
```cpp
__global__ void kernel(float *data) {
    int idx = threadIdx.x;      // register
    float val = data[idx];      // register (after loading from global)
    float result = val * 2.0f;  // register
    data[idx] = result;         // write register back to global
}
```

---

## Shared Memory

Shared memory is a fast, on-chip scratchpad that is shared by all threads
in a block. Think of it as a **user-controlled L1 cache**. Unlike hardware
caches, you decide explicitly what goes in and out.

```
  Shared Memory: per-block, on-chip
  ===================================

  +-- SM 0 -----------------------------------------------+
  |                                                        |
  |  Block A:                     Block B:                 |
  |  +--------------------------+ +-----------------------+|
  |  | Shared Memory (48 KB)    | | Shared Memory (48 KB) ||
  |  | +----------------------+ | | +-------------------+ ||
  |  | | user-controlled data | | | | user-controlled   | ||
  |  | | (declared __shared__)| | | | data              | ||
  |  | +----------------------+ | | +-------------------+ ||
  |  |                          | |                       ||
  |  | Thread 0 --+             | | Thread 0 --+          ||
  |  | Thread 1 --+--> can all  | | Thread 1 --+-> can all||
  |  | Thread 2 --+    read/    | | Thread 2 --+   read/  ||
  |  |   ...      |    write    | |   ...      |   write  ||
  |  | Thread 255-+    shared[] | | Thread 255-+   shared||
  |  +--------------------------+ +-----------------------+|
  |                                                        |
  |  Block A's shared memory is INVISIBLE to Block B.      |
  |  Each block gets its own private copy.                 |
  +--------------------------------------------------------+
```

**Why shared memory matters -- data reuse:**

Consider a 1D stencil: `out[i] = in[i-1] + in[i] + in[i+1]`

Without shared memory (global only):
```
  Thread 0 reads: in[0], in[1]           (also needs in[-1], ignore boundary)
  Thread 1 reads: in[0], in[1], in[2]    <-- in[0] and in[1] read AGAIN!
  Thread 2 reads: in[1], in[2], in[3]    <-- in[1] and in[2] read AGAIN!
  ...
  Each value is read from SLOW global memory 3 times!
```

With shared memory:
```
  1. All threads cooperatively load a tile into shared memory (1 global read each)
  2. __syncthreads()  <-- barrier: wait until all threads have loaded
  3. Each thread reads from FAST shared memory (3 reads, all on-chip)

  Result: 3x fewer global memory accesses!
```

Declaring shared memory:
```cpp
__global__ void kernel() {
    __shared__ float tile[256];         // static: size known at compile time
    // OR
    extern __shared__ float tile[];     // dynamic: size set at launch
}
// Dynamic launch: kernel<<<blocks, threads, shared_bytes>>>()
```

**Important: `__syncthreads()`** -- You MUST synchronize after loading shared
memory and before reading it. Without the barrier, some threads may read before
other threads have finished writing. This is a race condition.

---

## Global Memory

Global memory is the main GPU memory -- the large DRAM (VRAM) visible to all
threads across all blocks. It is the only way to communicate between blocks.

```
  Global Memory: per-device, off-chip DRAM
  ==========================================

  +---GPU Chip (on-die)---+          +---GDDR5 DRAM (off-chip)---+
  |  SM0  SM1  SM2 ...    |          |                            |
  |  SM3  SM4  SM5 ...    | <------> |  8 GB Global Memory        |
  |  ...  SM17            | 256-bit  |  (all threads can access)  |
  |                       |   bus    |                            |
  |  L2 cache (2 MB)      |          |  d_a[0..N-1]              |
  +-----------------------+          |  d_b[0..N-1]              |
                                     |  d_c[0..N-1]              |
                                     +----------------------------+

  Latency:    400-800 cycles (!!!)
  Bandwidth:  ~192 GB/s (theoretical peak for Quadro P4000)
  Size:       8 GB
```

**Memory coalescing is critical:**

When threads in a warp access consecutive 4-byte addresses, the hardware
combines (coalesces) them into a single 128-byte memory transaction.

```
  COALESCED (good):                    STRIDED (bad):
  Warp of 32 threads                   Warp of 32 threads

  T0  -> addr 0                        T0  -> addr 0
  T1  -> addr 4                        T1  -> addr 512
  T2  -> addr 8                        T2  -> addr 1024
  T3  -> addr 12                       T3  -> addr 1536
  ...                                  ...
  T31 -> addr 124                      T31 -> addr 15872

  = 1 transaction (128 bytes)          = 32 transactions!
  = Full bandwidth utilization         = 1/32 bandwidth utilization
```

---

## Constant Memory

Constant memory is a 64 KB read-only region that is cached aggressively.
When all threads in a warp read the **same** address, the value is broadcast
to all 32 threads in a single cycle.

```
  Constant Memory: read-only, cached, broadcast
  ================================================

  +-- Constant Cache (per SM) --+       +-- Constant Memory --+
  |  Very fast when all threads |       |   64 KB total        |
  |  in a warp read SAME addr  | <---- |   Read-only from GPU |
  |                             |       |   Written by CPU     |
  |  T0 reads addr X ----+     |       |   (cudaMemcpyTo-     |
  |  T1 reads addr X ----+---> |       |    Symbol)            |
  |  T2 reads addr X ----+ ONE |       +----------------------+
  |  ...                  | READ|
  |  T31 reads addr X ---+     |
  +-----------------------------+

  GOOD use case: filter coefficients, lookup tables, config values
  BAD use case:  per-thread data (serialized reads!)
```

Usage:
```cpp
__constant__ float filter[256];     // declared at file scope

// Host code:
cudaMemcpyToSymbol(filter, h_filter, 256 * sizeof(float));
```

---

## Local Memory (Register Spill)

"Local memory" is a confusing name. It is NOT fast local storage. It is
actually global memory (DRAM) used to store per-thread data that does not
fit in registers. The compiler places variables here when you use too many
registers.

```
  Register spill to local memory:
  ================================

  Your kernel has too many variables:
    float a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, ...
    float big_array[64];    // <-- definitely won't fit in registers!

  Compiler decision:
    Registers: a, b, c, d  (most frequently used)
    Local mem: big_array    (spilled to DRAM -- 400-800 cycle penalty!)

  Local memory has the SAME latency as global memory.
  It just has a per-thread address space.
```

**How to avoid register spill:**
- Keep kernels simple (fewer local variables)
- Avoid large local arrays
- Check register usage: `nvcc --ptxas-options=-v` reports registers per thread
- Use `__launch_bounds__` to hint the compiler

---

## Texture Memory

Texture memory is a read-only memory accessed through a special cache
optimized for 2D spatial locality. It was originally designed for graphics
but can be useful for certain compute patterns (e.g., image processing).

In modern CUDA (CC 3.0+), the texture cache is unified with the L1 cache.
For most compute workloads, you can get similar benefits using `__ldg()`
(load through read-only data cache) without the complexity of texture objects.

We will not use texture memory in this course, but know that it exists.

---

## Memory Visibility by Scope

Different memory types are visible at different levels of the thread hierarchy:

```
  Memory Visibility Diagram
  ==========================

  +--Thread---------+    +--Block-----------+    +--Grid (Device)--------+
  |                  |    |                   |    |                       |
  |  Registers       |    |  Shared Memory    |    |  Global Memory        |
  |  (private)       |    |  (all threads     |    |  (all threads in      |
  |                  |    |   in this block)  |    |   all blocks)         |
  |  Local Memory    |    |                   |    |                       |
  |  (private,       |    |                   |    |  Constant Memory      |
  |   register spill)|    |                   |    |  (read-only)          |
  |                  |    |                   |    |                       |
  +------------------+    +-------------------+    |  Texture Memory       |
                                                   |  (read-only)          |
                                                   +-----------------------+

  +--Host (CPU)---------------------------------------------------+
  |                                                                |
  |  System RAM (malloc, new, stack variables)                     |
  |                                                                |
  |  Can access GPU global memory via:                             |
  |    - cudaMemcpy (explicit copy)                                |
  |    - cudaMallocManaged (unified memory -- driver handles it)   |
  |    - Pinned memory + zero-copy (mapped host memory)            |
  +----------------------------------------------------------------+


  SUMMARY TABLE:
  +------------------+----------+---------+----------+---------+---------+
  | Memory           | Location | Scope   | Lifetime | Latency | Cached? |
  +------------------+----------+---------+----------+---------+---------+
  | Registers        | On-chip  | Thread  | Thread   | ~1 cyc  | N/A     |
  | Local (spill)    | DRAM     | Thread  | Thread   | ~500    | L1/L2   |
  | Shared           | On-chip  | Block   | Block    | ~5-30   | N/A     |
  | Global           | DRAM     | Grid    | App      | ~500    | L1/L2   |
  | Constant         | DRAM     | Grid    | App      | ~5*     | Special |
  | Texture          | DRAM     | Grid    | App      | ~5*     | Special |
  +------------------+----------+---------+----------+---------+---------+
  * When cached; cache miss falls back to DRAM latency (~500 cycles)
```

---

## Memory Bandwidth -- The Key Metric

For most GPU kernels, performance = how fast you can move data.

```
  Bandwidth Utilization
  =====================

  Theoretical peak bandwidth of Quadro P4000:
    Memory clock: 3003 MHz (effective, GDDR5)
    Bus width:    256 bits = 32 bytes
    Peak BW:      3003 MHz * 32 bytes * 2 (DDR) / 1000 = ~192 GB/s

  Achievable bandwidth (typically 80-90% of peak):
    Practical max: ~160-170 GB/s

  How to calculate achieved bandwidth:

    Achieved BW = (bytes_read + bytes_written) / kernel_time

  Example (vector add: c[i] = a[i] + b[i], N = 10M floats):
    Reads:   2 * 10M * 4 bytes = 80 MB
    Writes:  1 * 10M * 4 bytes = 40 MB
    Total:   120 MB
    If kernel takes 0.75 ms:
      BW = 120 MB / 0.00075 s = 160 GB/s  (83% of peak -- good!)

  If your kernel achieves >80% of peak bandwidth, it is well-optimized
  for a memory-bound kernel. You can only go faster by:
    1. Doing less memory traffic (algorithmic change)
    2. Using faster memory (shared mem, registers)
    3. Getting a GPU with more bandwidth
```

---

## Arithmetic Intensity and the Roofline Model

Arithmetic intensity (AI) = FLOPs performed / bytes moved

This single number tells you whether your kernel is compute-bound or
memory-bound.

```
  Roofline Model (simplified)
  ============================

  Performance (GFLOPS)
  ^
  |                          Compute ceiling (5300 GFLOPS)
  |                    ........................................
  |                   /:
  |                  / :
  |                 /  :
  |                /   :         <-- Compute-bound region
  |               /    :             (matrix multiply, etc.)
  |              /     :
  |             /      :
  |            /   <-- Memory-bound region
  |           /        :   (vector add, copy, reduction)
  |          /         :
  |         /          :
  |        /           :
  |       /            :
  |      /             :
  +-----+--------------+------> Arithmetic Intensity (FLOP/byte)
        ^              ^
        |              |
     vector add     sweet spot
     AI = 0.083     AI ~ 27.6
     (1 FLOP /      (peak compute / peak BW)
      12 bytes)

  Below the ridge point:  memory-bound -> optimize memory access
  Above the ridge point:  compute-bound -> optimize arithmetic

  Ridge point for Quadro P4000:
    5300 GFLOPS / 192 GB/s = ~27.6 FLOP/byte

  Common operations:
    Vector add:       AI = 0.08   -> heavily memory-bound
    Matrix-vector:    AI = 0.25   -> memory-bound
    Matrix multiply:  AI ~ 100+   -> compute-bound (at large N)
    Convolution:      AI ~ 1-10   -> depends on filter size
```

---

## Programs in This Chapter

### 1. memory_types.cu
Demonstrates all CUDA memory types (registers, shared, constant, global)
with timing comparisons and hardware info queries.

```bash
make memory_types
./memory_types
```

### 2. stencil_1d.cu
A practical 1D stencil (convolution/blur) showing the power of shared memory.
Compares a naive global-memory version against an optimized shared-memory
version with halo cells.

```bash
make stencil_1d
./stencil_1d
```

### 3. bandwidth_test.cu
Measures the actual achieved memory bandwidth of your GPU and compares
it against the theoretical peak. Teaches you to think about performance
in terms of bandwidth utilization.

```bash
make bandwidth_test
./bandwidth_test
```

### Build all:
```bash
make          # builds all three
make clean    # removes binaries
make run      # build and run all programs
```

---

## Exercises

1. In `memory_types.cu`, increase the shared memory array size until you
   hit the limit. What error do you get?
2. In `stencil_1d.cu`, try changing the stencil radius from 3 to 7. How does
   the shared-memory speedup change? (More reuse = more benefit.)
3. In `bandwidth_test.cu`, try different block sizes (64, 128, 256, 512).
   Does bandwidth change?
4. Calculate the arithmetic intensity of the stencil kernel. Is it
   memory-bound or compute-bound?
5. Use `nvcc --ptxas-options=-v` to compile `memory_types.cu` and check
   the register count per kernel. How many threads can run per SM?

---

## Quick Reference

```
  __shared__ float s[256];              -- shared memory (per block)
  __constant__ float c[64];             -- constant memory (64 KB max)
  extern __shared__ float dyn[];        -- dynamic shared memory
  kernel<<<B, T, sharedBytes>>>()       -- launch with dynamic shared mem
  __syncthreads()                       -- block-level barrier
  cudaMemcpyToSymbol(dst, src, size)    -- copy to constant memory
  cudaDeviceGetAttribute(&val, attr, dev)  -- query device properties
```
