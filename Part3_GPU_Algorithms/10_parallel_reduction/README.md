# Chapter 10: Parallel Reduction

## The Reduction Problem

**Reduction** is one of the most fundamental operations in parallel computing:
take N input values and produce a single output value using an associative
binary operator (sum, max, min, product, bitwise OR, etc.).

```
Input:  [3, 1, 7, 0, 4, 1, 6, 3]
                  |
            reduce(+)
                  |
Output:          25
```

Every GPU program that computes a total, finds a maximum, checks a condition
across an array, or computes a dot product uses reduction at its core.

- **Sequential complexity:** O(N) -- one thread walks the array.
- **Parallel complexity:**   O(log N) with N/2 processors -- a binary tree.

The challenge is mapping the tree onto GPU hardware efficiently. This chapter
follows **Mark Harris's classic NVIDIA presentation** "Optimizing Parallel
Reduction in CUDA" step by step, showing 7 optimization levels that improve
performance by 10-30x.

---

## Binary Tree Reduction (Conceptual)

For 8 elements, a sum reduction forms a tree with log2(8) = 3 steps:

```
Step 0 (input):    3     1     7     0     4     1     6     3
                    \   /       \   /       \   /       \   /
Step 1:             4           7           5           9
                     \         /             \         /
Step 2:              11                      14
                       \                    /
Step 3:                        25
```

Each step halves the number of active values. With enough processors,
each step runs in O(1), giving O(log N) total parallel time.

---

## The Seven Optimization Levels

### Level 1: Interleaved Addressing with Divergent Branching

The simplest GPU reduction: each thread checks `if (tid % (2*stride) == 0)`
to decide whether it participates in this step.

```
Thread IDs:     0    1    2    3    4    5    6    7
Shared mem:    [3]  [1]  [7]  [0]  [4]  [1]  [6]  [3]

Step 1 (stride=1):
  Thread 0: s[0] += s[1]  ->  4
  Thread 2: s[2] += s[3]  ->  7
  Thread 4: s[4] += s[5]  ->  5
  Thread 6: s[6] += s[7]  ->  9
  (threads 1,3,5,7 idle -- DIVERGENT branch!)

Shared mem:    [4]  [1]  [7]  [0]  [5]  [1]  [9]  [3]

Step 2 (stride=2):
  Thread 0: s[0] += s[2]  -> 11
  Thread 4: s[4] += s[6]  -> 14
  (threads 1,2,3,5,6,7 idle)

Shared mem:   [11]  [1]  [7]  [0] [14]  [1]  [9]  [3]

Step 3 (stride=4):
  Thread 0: s[0] += s[4]  -> 25

Result: s[0] = 25
```

**Problem:** The `tid % (2*stride) == 0` test causes **warp divergence**.
Within a single warp of 32 threads, some take the branch and some don't.
Both paths execute serially, wasting half the cycles.

---

### Level 2: Interleaved Addressing, Bank-Conflict-Free

Replace the modulo test with a computed index to avoid divergent branching:

```
int index = 2 * stride * tid;
if (index < blockDim.x) {
    sdata[index] += sdata[index + stride];
}
```

```
Step 1 (stride=1):
  tid=0 -> index=0:  s[0] += s[1]
  tid=1 -> index=2:  s[2] += s[3]
  tid=2 -> index=4:  s[4] += s[5]
  tid=3 -> index=6:  s[6] += s[7]
  (tids 0-3 all active, 4-7 do nothing -- still some divergence)
```

**Improvement:** Threads within each warp are more likely to follow the same
path. But there is a NEW problem: the stride-based indexing causes **shared
memory bank conflicts**. Addresses 0 and 2 may fall in the same bank.

---

### Level 3: Sequential Addressing (Stride Shrinks)

Reverse the loop direction -- start with a large stride and halve it:

```
Thread IDs:     0    1    2    3    4    5    6    7
Shared mem:    [3]  [1]  [7]  [0]  [4]  [1]  [6]  [3]

Step 1 (stride=4):  -- half of N
  tid=0: s[0] += s[4]  -> 7
  tid=1: s[1] += s[5]  -> 2
  tid=2: s[2] += s[7]  -> 13   (wait -- s[2]+s[6], not s[7])
  tid=2: s[2] += s[6]  -> 13
  tid=3: s[3] += s[7]  -> 3

Shared mem:    [7]  [2] [13]  [3]  [4]  [1]  [6]  [3]

Step 2 (stride=2):
  tid=0: s[0] += s[2]  -> 20
  tid=1: s[1] += s[3]  -> 5

Shared mem:   [20]  [5] [13]  [3]  [4]  [1]  [6]  [3]

Step 3 (stride=1):
  tid=0: s[0] += s[1]  -> 25

Result: s[0] = 25
```

**Key insight:** Threads 0..stride-1 are active at each step. These are
*contiguous* threads, so no warp divergence (all threads in a warp take the
same branch). Sequential addressing also avoids bank conflicts.

---

### Level 4: First Add During Load

Observation: in level 3, the first step activates only N/2 threads (half the
block). We launch N threads but immediately idle half of them. Wasteful!

**Solution:** Each thread loads and adds TWO elements from global memory:

```
// Each thread loads two elements and adds them
sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
```

Now we launch half as many blocks. The first reduction step is "free" --
it happens during the load. Same work, half the blocks, better occupancy.

```
Global data:  [3  1  7  0 | 4  1  6  3]
               block loads:
  tid=0: s[0] = g[0] + g[4] = 3+4 = 7
  tid=1: s[1] = g[1] + g[5] = 1+1 = 2
  tid=2: s[2] = g[2] + g[6] = 7+6 = 13
  tid=3: s[3] = g[3] + g[7] = 0+3 = 3

Shared mem:    [7]  [2] [13]  [3]
               (only 4 elements -- half the block size)

Step 1 (stride=2):
  tid=0: s[0] += s[2]  -> 20
  tid=1: s[1] += s[3]  -> 5

Step 2 (stride=1):
  tid=0: s[0] += s[1]  -> 25
```

---

### Level 5: Unroll the Last Warp

When fewer than 32 elements remain (one warp), all active threads execute
in lockstep (SIMD). We do NOT need `__syncthreads()` for the last 5 steps
(stride 16, 8, 4, 2, 1). We can unroll them into straight-line code:

```
// When 32 or fewer threads remain, no sync needed
if (tid < 32) {
    volatile float* smem = sdata;  // volatile prevents caching
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
}
```

The `volatile` keyword is critical -- it forces the compiler to write back
to shared memory after each operation, ensuring other threads in the warp
see the updated value. (On CC >= 7.0, you would use `__syncwarp()` instead.)

**Note on CC 6.1 (our Quadro P4200):** Warps are still fully synchronous
(no independent thread scheduling), so the volatile trick works correctly.

---

### Level 6: Completely Unrolled (Template on Block Size)

If we know the block size at compile time, we can unroll the ENTIRE loop
using C++ templates:

```cpp
template <unsigned int blockSize>
__global__ void reduce6(float *g_idata, float *g_odata, int n) {
    // ... load ...
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid+256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid+128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  sdata[tid] += sdata[tid+64];  __syncthreads(); }
    // warp unroll for last 32 ...
}
```

The `if (blockSize >= X)` conditions are evaluated at **compile time** (the
template parameter is a constant). Dead code is eliminated entirely. The
result is a fully unrolled reduction with zero loop overhead.

---

### Level 7: Multiple Elements per Thread (Grid-Stride Loop)

The ultimate optimization: instead of one (or two) elements per thread,
each thread processes MANY elements using a grid-stride loop before the
tree reduction begins:

```
// Each thread accumulates many elements
float mySum = 0;
for (int i = blockIdx.x * blockDim.x + threadIdx.x;
     i < n;
     i += blockDim.x * gridDim.x)    // grid stride
{
    mySum += g_idata[i];
}
sdata[tid] = mySum;
__syncthreads();
// ... then do the tree reduction in shared memory ...
```

**Why this works so well:**
- Maximizes **arithmetic intensity** -- each thread does useful work before
  the tree reduction even starts.
- We launch far fewer blocks (e.g., just 2x the SM count), improving
  occupancy and reducing the final block-level reduction.
- The grid-stride loop achieves **memory coalescing** naturally.
- Can finish with **warp shuffle** (`__shfl_down_sync`) instead of shared
  memory for the final 32 elements.

```
Grid-stride pattern for 32 elements, 4 threads, 1 block:

  tid=0 processes: g[0], g[4], g[8],  g[12], g[16], g[20], g[24], g[28]
  tid=1 processes: g[1], g[5], g[9],  g[13], g[17], g[21], g[25], g[29]
  tid=2 processes: g[2], g[6], g[10], g[14], g[18], g[22], g[26], g[30]
  tid=3 processes: g[3], g[7], g[11], g[15], g[19], g[23], g[27], g[31]

  Each thread has a partial sum -> shared memory tree reduction on 4 values
```

---

## Multi-Block Reduction Strategy

A single block can hold at most 1024 threads. For millions of elements,
we need multiple blocks. But blocks cannot communicate during a kernel launch.

**Solution: two-pass reduction.**

```
Pass 1: N elements -> B partial sums (one per block)
  Block 0: reduce elements [0..chunk)       -> partial[0]
  Block 1: reduce elements [chunk..2*chunk) -> partial[1]
  ...
  Block B-1: reduce last chunk              -> partial[B-1]

Pass 2: B partial sums -> 1 final result
  Single block (or a few blocks) reduces the partial array.

Two kernel launches suffice for ANY array size.
```

```
                    Global Array (N elements)
  [----chunk 0----][----chunk 1----]...[----chunk B-1----]
         |                |                    |
      Block 0          Block 1             Block B-1
         |                |                    |
    partial[0]       partial[1]          partial[B-1]
         \                |                   /
          \               |                  /
           --------  Pass 2 kernel  --------
                          |
                     final result
```

---

## Performance Summary (Typical on Pascal/Maxwell GPUs)

| Level | Optimization                    | Speedup vs L1 |
|-------|---------------------------------|---------------|
| 1     | Divergent branching             | 1.0x          |
| 2     | Non-divergent indexing           | ~1.5x         |
| 3     | Sequential addressing           | ~2.0x         |
| 4     | First add during load           | ~3.0x         |
| 5     | Unroll last warp                | ~4.0x         |
| 6     | Completely unrolled             | ~4.5x         |
| 7     | Grid-stride + warp shuffle      | ~10-30x       |

The jump from Level 1 to Level 7 is dramatic. Level 7 typically achieves
close to peak memory bandwidth of the GPU, which is the theoretical limit
for a memory-bound operation like reduction.

---

## Key Takeaways

1. **Reduction is memory-bound.** The theoretical limit is determined by
   global memory bandwidth, not compute throughput.
2. **Warp divergence kills performance.** Ensure contiguous threads take
   the same branch.
3. **Bank conflicts matter.** Sequential addressing avoids them naturally.
4. **Do useful work during loads.** The first add during load is "free."
5. **Unrolling the last warp** removes synchronization overhead for the
   final 32 threads.
6. **Grid-stride loops** let each thread process many elements, maximizing
   the ratio of useful work to overhead.
7. **Warp shuffle** (`__shfl_down_sync`) replaces shared memory for the
   final intra-warp reduction -- no shared memory needed, no bank conflicts.

---

## Files in This Chapter

| File                       | Description                                    |
|----------------------------|------------------------------------------------|
| `reduction.cu`             | All 7 optimization levels, benchmarked          |
| `reduce_max.cu`            | Reduction for max/argmax, generic template       |
| `multi_block_reduction.cu` | Full two-pass reduction for very large arrays    |
| `Makefile`                 | Build all three programs                        |

---

## References

- Mark Harris, "Optimizing Parallel Reduction in CUDA," NVIDIA Developer
  Technology, 2007.
- CUDA C++ Programming Guide, Chapter on Warp Shuffle Functions.
- NVIDIA CUB library (production-quality reduction implementation).
