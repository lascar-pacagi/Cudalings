# Chapter 09: Warp-Level Primitives

## Warps Revisited: The True Unit of Execution

On NVIDIA GPUs, threads are not truly independent. The hardware groups every
**32 consecutive threads** within a block into a **warp**. All 32 threads in a
warp execute the **same instruction at the same time** -- this is called
**SIMT** (Single Instruction, Multiple Thread) execution.

```
    A CUDA block with 256 threads contains 8 warps
    ═══════════════════════════════════════════════════════════════

    Block (256 threads)
    ┌─────────────────────────────────────────────────────────────┐
    │  Warp 0:  threads  0-31   (lane 0 = threadIdx.x  0)       │
    │  Warp 1:  threads 32-63   (lane 0 = threadIdx.x 32)       │
    │  Warp 2:  threads 64-95   (lane 0 = threadIdx.x 64)       │
    │  Warp 3:  threads 96-127  (lane 0 = threadIdx.x 96)       │
    │  Warp 4:  threads 128-159                                   │
    │  Warp 5:  threads 160-191                                   │
    │  Warp 6:  threads 192-223                                   │
    │  Warp 7:  threads 224-255                                   │
    └─────────────────────────────────────────────────────────────┘

    Each warp runs in lockstep:
    ┌────────────────────────────────────────────┐
    │  Clock cycle N:   all 32 lanes execute ADD │
    │  Clock cycle N+1: all 32 lanes execute MUL │
    │  Clock cycle N+2: all 32 lanes execute LD  │
    └────────────────────────────────────────────┘

    "lane" = a thread's position within its warp (0-31)
    lane = threadIdx.x % 32
```

Because a warp executes in lockstep, the 32 threads can communicate **directly
through registers** without touching shared memory or calling `__syncthreads()`.
This is what makes warp-level primitives so powerful.

---

## Why Warp-Level Operations Are Fast

Consider two ways to share data between threads:

```
    Method 1: Shared Memory (traditional)
    ═══════════════════════════════════════════════════════════════

    Thread A             Shared Memory             Thread B
    ┌──────┐     write   ┌──────────┐    read     ┌──────┐
    │ reg  │ ──────────→ │  smem[]  │ ──────────→ │ reg  │
    └──────┘             └──────────┘             └──────┘
                               │
                    __syncthreads() barrier required!
                    (expensive for cross-warp, but even
                     within a warp you pay memory latency)

    Cost: ~20-30 cycles (shared memory load/store + sync)


    Method 2: Warp Shuffle (this chapter!)
    ═══════════════════════════════════════════════════════════════

    Thread A                                      Thread B
    ┌──────┐          direct register             ┌──────┐
    │ reg  │ ─────────── move ──────────────────→ │ reg  │
    └──────┘         (no memory at all!)          └──────┘

    Cost: ~1-2 cycles (register-to-register within warp)
    No shared memory needed. No synchronization needed.
```

Warp shuffles are **10-20x faster** than shared memory for intra-warp
communication. They also save shared memory for other uses (like tiling).

---

## Warp Lanes and Shuffle Data Movement

Each thread in a warp has a **lane ID** from 0 to 31. Shuffle instructions
move data between lanes:

```
    Warp: 32 lanes
    ═══════════════════════════════════════════════════════════════

    Lane:   0   1   2   3   4   5   6   7  ...  30  31
          ┌───┬───┬───┬───┬───┬───┬───┬───┐   ┌───┬───┐
    Data: │ A │ B │ C │ D │ E │ F │ G │ H │...│ ? │ ? │
          └───┴───┴───┴───┴───┴───┴───┴───┘   └───┴───┘
            │   │   │   │   │   │   │   │         │   │
            └───┴───┴───┴───┴───┴───┴───┴─────────┴───┘
                   All connected by shuffle network!

    Any lane can read any other lane's register value
    in a SINGLE instruction.
```

---

## Shuffle Operations

CUDA provides four shuffle variants (all require Compute Capability 3.0+):

### 1. `__shfl_sync(mask, value, srcLane)` -- Broadcast / Direct Read

Every participating lane reads the value from lane `srcLane`.

```
    __shfl_sync(0xFFFFFFFF, val, 2)   -- everyone reads from lane 2
    ═══════════════════════════════════════════════════════════════

    Source lane: 2
                        ↓
    Lane:   0   1   2   3   4   5   6   7
          ┌───┬───┬───┬───┬───┬───┬───┬───┐
    Before│ A │ B │ C │ D │ E │ F │ G │ H │
          └───┴───┴───┴───┴───┴───┴───┴───┘
                    │
          ┌─────┬──┼──┬─────┬─────┬─────┬─────┬─────┐
          ↓     ↓  ↓  ↓     ↓     ↓     ↓     ↓     ↓
          ┌───┬───┬───┬───┬───┬───┬───┬───┐
    After │ C │ C │ C │ C │ C │ C │ C │ C │
          └───┴───┴───┴───┴───┴───┴───┴───┘

    Use case: broadcast a value from one lane to all others
```

### 2. `__shfl_up_sync(mask, value, delta)` -- Shift Up

Each lane reads from lane `(laneID - delta)`. Lanes where `(laneID - delta) < 0`
keep their own value (no wrap-around).

```
    __shfl_up_sync(0xFFFFFFFF, val, 1)   -- shift up by 1
    ═══════════════════════════════════════════════════════════════

    Lane:   0   1   2   3   4   5   6   7
          ┌───┬───┬───┬───┬───┬───┬───┬───┐
    Before│ A │ B │ C │ D │ E │ F │ G │ H │
          └───┴───┴───┴───┴───┴───┴───┴───┘
            │   │ ↗ │ ↗ │ ↗ │ ↗ │ ↗ │ ↗
            │   ↗   ↗   ↗   ↗   ↗   ↗
          ┌───┬───┬───┬───┬───┬───┬───┬───┐
    After │ A │ A │ B │ C │ D │ E │ F │ G │
          └───┴───┴───┴───┴───┴───┴───┴───┘
            ↑
          unchanged (lane 0 has no lane -1 to read from)

    Use case: prefix sums (inclusive/exclusive scan)
```

### 3. `__shfl_down_sync(mask, value, delta)` -- Shift Down

Each lane reads from lane `(laneID + delta)`. Lanes where `(laneID + delta) >= 32`
keep their own value.

```
    __shfl_down_sync(0xFFFFFFFF, val, 1)   -- shift down by 1
    ═══════════════════════════════════════════════════════════════

    Lane:   0   1   2   3   4   5   6   7
          ┌───┬───┬───┬───┬───┬───┬───┬───┐
    Before│ A │ B │ C │ D │ E │ F │ G │ H │
          └───┴───┴───┴───┴───┴───┴───┴───┘
            ↘   ↘   ↘   ↘   ↘   ↘   ↘   │
              ↘   ↘   ↘   ↘   ↘   ↘      │
          ┌───┬───┬───┬───┬───┬───┬───┬───┐
    After │ B │ C │ D │ E │ F │ G │ H │ H │
          └───┴───┴───┴───┴───┴───┴───┴───┘
                                        ↑
                          unchanged (lane 7 has no lane 8)

    Use case: REDUCTIONS (sum, max, etc.) -- the star of this chapter
```

### 4. `__shfl_xor_sync(mask, value, laneMask)` -- Butterfly Exchange

Each lane reads from lane `(laneID ^ laneMask)`. The XOR creates a symmetric
"butterfly" exchange pattern.

```
    __shfl_xor_sync(0xFFFFFFFF, val, 1)   -- XOR with 1
    ═══════════════════════════════════════════════════════════════

    Lane:   0   1   2   3   4   5   6   7
          ┌───┬───┬───┬───┬───┬───┬───┬───┐
    Before│ A │ B │ C │ D │ E │ F │ G │ H │
          └───┴───┴───┴───┴───┴───┴───┴───┘
            ↕       ↕       ↕       ↕
          ┌───┬───┬───┬───┬───┬───┬───┬───┐
    After │ B │ A │ D │ C │ F │ E │ H │ G │
          └───┴───┴───┴───┴───┴───┴───┴───┘

    0^1=1, 1^1=0  →  lanes 0,1 swap
    2^1=3, 3^1=2  →  lanes 2,3 swap
    4^1=5, 5^1=4  →  lanes 4,5 swap  ... etc.


    __shfl_xor_sync(0xFFFFFFFF, val, 2)   -- XOR with 2
    ═══════════════════════════════════════════════════════════════

    Lane:   0   1   2   3   4   5   6   7
          ┌───┬───┬───┬───┬───┬───┬───┬───┐
    Before│ A │ B │ C │ D │ E │ F │ G │ H │
          └───┴───┴───┴───┴───┴───┴───┴───┘
            ↕   ↕               ↕   ↕
            └───┼───↔───┘       └───┼───↔───┘
                └───────↔───────────┘
          ┌───┬───┬───┬───┬───┬───┬───┬───┐
    After │ C │ D │ A │ B │ G │ H │ E │ F │
          └───┴───┴───┴───┴───┴───┴───┴───┘

    0^2=2, 2^2=0  →  lanes 0,2 swap
    1^2=3, 3^2=1  →  lanes 1,3 swap  ... etc.

    Use case: butterfly reductions, all-to-all communication
```

---

## The Mask Parameter: `0xFFFFFFFF`

Every `_sync` shuffle and vote function takes a **mask** as its first argument.
The mask is a 32-bit integer where each bit represents a lane:

```
    mask = 0xFFFFFFFF  (all 32 bits set = all lanes participate)
    ═══════════════════════════════════════════════════════════════

    Bit:   31 30 29 28 ... 3  2  1  0
           1  1  1  1      1  1  1  1   ← all lanes active

    mask = 0x0000000F  (only lanes 0-3 participate)

    Bit:   31 30 29 28 ... 3  2  1  0
           0  0  0  0      1  1  1  1   ← only first 4 lanes

    RULE: Every lane that executes the shuffle instruction MUST
    have its bit set in the mask, or you get UNDEFINED BEHAVIOR.

    In practice: almost always use 0xFFFFFFFF (full warp).
    If you know some lanes have diverged away, use __activemask()
    or construct a precise mask.
```

---

## Warp Vote Functions

Vote functions let all lanes in a warp collaboratively evaluate a condition:

```
    Warp vote functions
    ═══════════════════════════════════════════════════════════════

    __all_sync(mask, predicate)
    ───────────────────────────
    Returns non-zero if predicate is true for ALL lanes in mask.

    Lane:    0   1   2   3   4   5   6   7
    Pred:    T   T   T   T   T   T   T   T  →  __all_sync = 1
    Pred:    T   T   F   T   T   T   T   T  →  __all_sync = 0


    __any_sync(mask, predicate)
    ───────────────────────────
    Returns non-zero if predicate is true for ANY lane in mask.

    Lane:    0   1   2   3   4   5   6   7
    Pred:    F   F   F   F   F   F   F   F  →  __any_sync = 0
    Pred:    F   F   T   F   F   F   F   F  →  __any_sync = 1


    __ballot_sync(mask, predicate)
    ──────────────────────────────
    Returns a 32-bit bitmask where bit i is set if lane i's predicate is true.

    Lane:    0   1   2   3   4   5   6   7  ...  31
    Pred:    T   F   T   T   F   T   F   F  ...  F
                                                  │
    Result:  0b...00101101 = 0x2D                  │
                  ||||||||                         │
                  |||||||└─ lane 0: T (bit set)    │
                  ||||||└── lane 1: F              │
                  |||||└─── lane 2: T (bit set)    │
                  ||||└──── lane 3: T (bit set)    │
                  |||└───── lane 4: F              │
                  ||└────── lane 5: T (bit set)    │
                  |└─────── lane 6: F              │
                  └──────── lane 7: F              │
```

---

## `__activemask()` and Convergence

`__activemask()` returns a bitmask of which lanes are currently active
(not diverged away). It does NOT synchronize -- it just observes.

```
    Divergent code:
    ═══════════════════════════════════════════════════════════════

    if (threadIdx.x % 2 == 0) {
        // Only even lanes execute here
        unsigned mask = __activemask();
        // mask = 0x55555555 (bits 0,2,4,6,8,...,30 set)
    } else {
        // Only odd lanes execute here
        unsigned mask = __activemask();
        // mask = 0xAAAAAAAA (bits 1,3,5,7,9,...,31 set)
    }

    WARNING: __activemask() does not guarantee convergence.
    Always prefer explicit masks or __syncwarp(mask) when
    you need threads to reconverge.
```

---

## Why This Matters: Warp Reduction Without Shared Memory

The single most important application of warp shuffles is **reduction**:
summing 32 values into 1, using only 5 shuffle steps (log2(32) = 5).

```
    Warp reduction using __shfl_down_sync (5 steps for 32 values)
    ═══════════════════════════════════════════════════════════════

    Step 1: delta=16   lanes 0-15 add values from lanes 16-31
    Step 2: delta=8    lanes 0-7  add values from lanes 8-15
    Step 3: delta=4    lanes 0-3  add values from lanes 4-7
    Step 4: delta=2    lanes 0-1  add values from lanes 2-3
    Step 5: delta=1    lane  0    adds value from lane 1

    Result: lane 0 holds the sum of all 32 values

    Total cost: 5 shuffle instructions (~5-10 cycles)
    vs. shared memory: 5 loads + 5 stores + barriers (~100+ cycles)
```

This technique is the foundation for high-performance reductions, which we
will use extensively in Part 3 when building real kernels.

---

## Block-Level Reduction Using Warps

For blocks larger than 32 threads, we combine warp-level and shared memory
reduction:

```
    Block-level reduction (256 threads = 8 warps)
    ═══════════════════════════════════════════════════════════════

    Phase 1: Each warp reduces 32 values to 1 (warp shuffle)
    ─────────────────────────────────────────────────────────
    Warp 0: 32 values ──→ 1 partial sum (lane 0 of warp 0)
    Warp 1: 32 values ──→ 1 partial sum (lane 0 of warp 1)
    Warp 2: 32 values ──→ 1 partial sum (lane 0 of warp 2)
    ...
    Warp 7: 32 values ──→ 1 partial sum (lane 0 of warp 7)

    Phase 2: Lane 0 of each warp writes its partial sum to shared memory
    ─────────────────────────────────────────────────────────
    shared[0] = warp 0 sum
    shared[1] = warp 1 sum
    ...
    shared[7] = warp 7 sum

    Phase 3: First warp reduces the 8 partial sums (warp shuffle again)
    ─────────────────────────────────────────────────────────
    Warp 0 loads shared[0..7] and reduces → final sum in lane 0!

    Total shared memory used: just 8 floats (vs. 256 for naive approach)
```

---

## Summary

| Primitive | Purpose | Cost |
|-----------|---------|------|
| `__shfl_sync` | Broadcast / read from specific lane | ~1-2 cycles |
| `__shfl_up_sync` | Shift values toward lower lanes | ~1-2 cycles |
| `__shfl_down_sync` | Shift values toward higher lanes | ~1-2 cycles |
| `__shfl_xor_sync` | Butterfly / XOR exchange | ~1-2 cycles |
| `__all_sync` | All-true vote | ~1-2 cycles |
| `__any_sync` | Any-true vote | ~1-2 cycles |
| `__ballot_sync` | Bitmask of true lanes | ~1-2 cycles |

Warp-level primitives are the **secret weapon** of high-performance CUDA.
They bypass shared memory entirely, require no synchronization, and operate
at register speed. Every fast reduction, scan, and communication pattern
in production CUDA code uses them.

---

## Files in This Chapter

| File | Description |
|------|-------------|
| `shuffle_demo.cu` | Demonstrates all four shuffle variants with before/after output |
| `warp_reduction.cu` | Builds warp-level and block-level reductions, benchmarks vs shared memory |
| `vote_functions.cu` | Warp vote functions with practical examples |
| `Makefile` | Build with `make all`, run with `make run` |
