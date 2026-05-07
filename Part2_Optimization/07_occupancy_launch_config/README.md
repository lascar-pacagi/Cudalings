# Chapter 07: Occupancy and Launch Configuration

## What Is Occupancy?

Occupancy is the ratio of **active warps** on an SM to the **maximum number of warps**
the SM can support simultaneously.

```
                          active warps on SM
    Occupancy  =  ─────────────────────────────────
                   max warps SM can hold (hardware)
```

For Compute Capability 6.1 (Pascal, e.g., Quadro P4200):
- Maximum threads per SM: **2048**
- A warp = 32 threads
- Maximum warps per SM: **2048 / 32 = 64 warps**

If your kernel launch results in 32 active warps on an SM, your occupancy is:

```
    32 / 64 = 50% occupancy
```

---

## Why Does Occupancy Matter?

The GPU hides memory latency by **switching between warps**. When one warp stalls
waiting for data from global memory (hundreds of cycles), the warp scheduler
instantly switches to another ready warp — no cost, no context switch overhead.

```
    Timeline on ONE SM (simplified)
    ══════════════════════════════════════════════════════════════

    Warp A: ████████░░░░░░░░░░░░░░░░████████░░░░░░░░████████
                     ↑ stalled on      ↑ data arrived,
                       memory load       resume compute

    Warp B: ░░░░░░░░████████░░░░░░░░░░░░░░░░████████░░░░░░░░
                     ↑ scheduled         ↑ scheduled
                       while A stalls      while A stalls

    Warp C: ░░░░░░░░░░░░░░░░████████░░░░░░░░░░░░░░░░████████
                             ↑ scheduled
                               while A,B stall

    ──────────────────────────────────────────────────────────
    Time →

    Key:  ████ = executing instructions
          ░░░░ = idle / waiting

    MORE active warps  →  MORE candidates to schedule when stalls happen
                       →  SM stays busy  →  higher throughput
```

With **high occupancy**, there are many warps ready to run, so the SM rarely sits idle.
With **low occupancy**, the SM may run out of ready warps and stall ("occupancy-limited").

---

## The Three Occupancy Limiters

Three resources on each SM limit how many warps (threads) can be active:

```
    ┌─────────────────────────────────────────────────────────┐
    │                    SM Resources (CC 6.1)                │
    │                                                         │
    │  ┌─────────────────┐  ┌──────────────┐  ┌───────────┐  │
    │  │   REGISTERS     │  │ SHARED MEM   │  │  THREAD   │  │
    │  │                 │  │              │  │  SLOTS    │  │
    │  │  65,536 regs    │  │  48 KB       │  │  2048     │  │
    │  │  (per SM)       │  │  (per SM)    │  │  threads  │  │
    │  │                 │  │              │  │  (per SM) │  │
    │  └────────┬────────┘  └──────┬───────┘  └─────┬─────┘  │
    │           │                  │                │         │
    │           └──────────────────┼────────────────┘         │
    │                              │                          │
    │                     ┌────────▼────────┐                 │
    │                     │   OCCUPANCY =   │                 │
    │                     │   min of all    │                 │
    │                     │   three limits  │                 │
    │                     └─────────────────┘                 │
    └─────────────────────────────────────────────────────────┘
```

### Limiter 1: Registers Per Thread

Each SM has a fixed register file. If your kernel uses many registers per thread,
fewer threads can fit on the SM.

```
    CC 6.1: 65,536 registers per SM

    If kernel uses 32 regs/thread:
        65,536 / 32 = 2048 threads can fit  →  full occupancy possible

    If kernel uses 64 regs/thread:
        65,536 / 64 = 1024 threads can fit  →  50% max occupancy

    If kernel uses 128 regs/thread:
        65,536 / 128 = 512 threads can fit  →  25% max occupancy
```

### Limiter 2: Shared Memory Per Block

Each SM has 48 KB of shared memory. If each block uses a lot of shared memory,
fewer blocks can fit concurrently on the SM.

```
    CC 6.1: 49,152 bytes (48 KB) shared memory per SM

    If block uses 16 KB shared:
        49,152 / 16,384 = 3 blocks fit  (if threads allow more)

    If block uses 24 KB shared:
        49,152 / 24,576 = 2 blocks fit

    If block uses 48 KB shared:
        49,152 / 49,152 = 1 block fits  →  severely limited
```

### Limiter 3: Block Size (Threads Per Block)

The SM also has a maximum number of **resident blocks** (32 on CC 6.1). If your
blocks are very small, you can waste thread slots:

```
    If blockDim = 32 (1 warp per block):
        Max 32 blocks/SM → 32 × 32 = 1024 threads = 50% occupancy
        (The SM can hold 2048 threads, but the 32-block limit caps us)

    If blockDim = 64 (2 warps per block):
        Max 32 blocks/SM → 32 × 64 = 2048 threads = 100% (fully utilized)

    If blockDim = 256 (8 warps per block):
        2048 / 256 = 8 blocks → 8 blocks × 256 = 2048 = 100%

    If blockDim = 1024 (32 warps per block):
        2048 / 1024 = 2 blocks → 2 × 1024 = 2048 = 100%
        But only 2 blocks — less scheduling flexibility
```

---

## CC 6.1 Resource Limits — Complete Table

```
    ┌──────────────────────────────────┬────────────┐
    │          Resource                │  CC 6.1    │
    ├──────────────────────────────────┼────────────┤
    │  Max threads per SM             │   2048     │
    │  Max warps per SM               │   64       │
    │  Max blocks per SM              │   32       │
    │  Max threads per block          │   1024     │
    │  Registers per SM               │   65,536   │
    │  Max registers per thread       │   255      │
    │  Shared memory per SM           │   48 KB    │
    │  Max shared memory per block    │   48 KB    │
    │  Warp size                      │   32       │
    │  Number of SMs (Quadro P4200)   │   18       │
    └──────────────────────────────────┴────────────┘
```

---

## Worked Example: Register-Limited Occupancy

**Given:** Kernel uses 32 registers/thread, block size = 256 threads (8 warps/block)

**Step 1: Register limit**
```
    65,536 regs / (32 regs × 256 threads) = 65,536 / 8,192 = 8 blocks
    8 blocks × 256 threads = 2048 threads → 64 warps → 100%
```

**Step 2: Thread slot limit**
```
    2048 max threads / 256 threads per block = 8 blocks
    8 blocks × 256 = 2048 → 100%
```

**Step 3: Block slot limit**
```
    32 max blocks ≥ 8 needed → not limiting
```

**Step 4: Shared memory limit** (assume 0 bytes dynamic shared)
```
    Not limiting
```

**Result:** Occupancy = min(100%, 100%, OK, OK) = **100%**

Now change to 64 registers/thread:
```
    65,536 / (64 × 256) = 65,536 / 16,384 = 4 blocks
    4 blocks × 8 warps = 32 warps → 32/64 = 50% occupancy
```

Registers halved the occupancy!

---

## The cudaOccupancyMaxActiveBlocksPerMultiprocessor API

Instead of doing the math by hand, CUDA provides runtime occupancy calculation:

```cpp
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,          // output: blocks that fit per SM
        myKernel,            // the kernel function
        blockSize,           // threads per block you plan to use
        dynamicSharedMem     // bytes of dynamic shared memory per block
    );

    float occupancy = (numBlocks * blockSize) / (float)maxThreadsPerSM;
```

There is also an auto-tuning API that finds the optimal block size:

```cpp
    int blockSize, minGridSize;
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,        // output: minimum grid size for full device occupancy
        &blockSize,          // output: optimal block size
        myKernel,            // the kernel function
        0,                   // dynamic shared memory per block
        0                    // block size limit (0 = no limit)
    );
```

---

## When High Occupancy Does NOT Help

Occupancy is **not** always the bottleneck. A kernel can be:

1. **Memory-bound:** Needs high occupancy to hide memory latency (occupancy helps)
2. **Compute-bound:** ALU is the bottleneck, not waiting for memory (occupancy
   beyond ~50% often provides diminishing returns)
3. **Latency-bound:** Occupancy helps a lot

```
    Performance vs. Occupancy (typical curves)
    ▲
    │  Memory-bound              Compute-bound
    │  kernel:                   kernel:
    │
    │        ╱───────────          ╱──────────────
    │      ╱                     ╱
    │    ╱                      ╱
    │  ╱                      ╱
    │╱                      ╱─
    └──────────────────  └──────────────────
     0%    50%   100%    0%    50%   100%
         Occupancy            Occupancy

    Memory-bound: strong     Compute-bound: steep rise
    gains all the way up     to ~50%, then flat
```

---

## The Volkov Insight: ILP vs. TLP

Vasily Volkov demonstrated that **instruction-level parallelism (ILP)** can
substitute for **thread-level parallelism (TLP)**:

```
    Traditional approach:      Volkov approach:
    Many threads (high occ.)   Fewer threads (lower occ.)
    Few regs per thread        More regs per thread
    Data in shared mem/cache   Data in registers (fastest!)
    Low ILP per thread         High ILP per thread

    Sometimes FASTER because registers are the fastest storage
    on the GPU, and ILP keeps the pipeline full without needing
    as many warps to hide latency.
```

The takeaway: **Occupancy is a tool, not a goal.** Measure actual performance.
The occupancy calculator tells you the theoretical maximum, but benchmarking
tells you the truth.

---

## Files in This Chapter

| File                   | What It Demonstrates                                    |
|------------------------|---------------------------------------------------------|
| `occupancy_demo.cu`    | Query occupancy with CUDA APIs, register/shared limits  |
| `launch_config.cu`     | Auto-tune block size, compare performance across configs|
| `register_pressure.cu` | Register spilling, -maxrregcount, the sweet spot        |
| `Makefile`             | Build all demos with sm_61                              |

---

## Building and Running

```bash
make all          # Build everything
make run          # Run all demos
make clean        # Remove binaries

# Individual targets
make occupancy_demo
make launch_config
make register_pressure
```

## Key Takeaways

1. Occupancy = active warps / max warps per SM
2. Limited by registers, shared memory, and block/thread slot counts
3. Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to query occupancy
4. Use `cudaOccupancyMaxPotentialBlockSize` to auto-tune block size
5. Higher occupancy helps memory-bound kernels hide latency
6. Compute-bound kernels may not benefit from occupancy beyond ~50%
7. Sometimes using MORE registers per thread (lower occupancy) is faster
8. Always benchmark — occupancy is a guideline, not a guarantee
