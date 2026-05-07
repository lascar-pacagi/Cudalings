# Chapter 11: Scan (Prefix Sum)

## What Is Scan?

**Scan** (also called **prefix sum**) is one of the most important parallel
primitives in GPU computing. Given an array and an associative binary operator,
scan produces an output array where each element is the "running total" of all
previous elements.

There are two variants:

### Inclusive Scan

Each output element includes the current input element:

```
Input:      [3,  1,  7,  0,  4]
Output:     [3,  4, 11, 11, 15]
             |   |   |   |   |
             3  3+1 3+1+7 ... 3+1+7+0+4
```

### Exclusive Scan

Each output element excludes the current input element (starts with identity):

```
Input:      [3,  1,  7,  0,  4]
Output:     [0,  3,  4, 11, 11]
             |   |   |   |   |
            id   3  3+1 3+1+7 3+1+7+0
```

The last element of an exclusive scan equals the total sum minus the last input.
Equivalently, exclusive scan is inclusive scan shifted right by one, with the
identity (0 for addition) inserted at the front.

---

## Why Scan Is So Important

Scan converts serial dependencies into parallel-friendly form. It appears as a
building block in a huge number of GPU algorithms:

| Application              | How Scan Is Used                                    |
|--------------------------|-----------------------------------------------------|
| **Stream compaction**    | Compute scatter addresses from a predicate array    |
| **Radix sort**           | Prefix sum of digit histograms gives write offsets  |
| **Histogram**            | Prefix sum turns counts into bin boundaries         |
| **Sparse matrix (SpMV)**| Row pointer array is a prefix sum of row lengths    |
| **Memory allocation**    | Exclusive scan of sizes gives offsets into buffer   |
| **Tree traversal (BVH)**| Scan to assign contiguous indices to leaves         |
| **Run-length encoding**  | Scan of flags marks segment boundaries              |
| **Polynomial evaluation**| Horner's rule is a scan with multiply-add           |

If reduction is the "hello world" of parallel algorithms, **scan is the Swiss
Army knife**.

---

## Algorithm 1: Hillis-Steele (Inclusive Scan)

This algorithm is **step-efficient** (fewest parallel steps = log2 N) but
**work-inefficient** (total operations = O(N log N) instead of O(N)).

### Idea

In each step d = 0, 1, ..., log2(N)-1:
- Each element i reads the value at position (i - 2^d) and adds it to itself
- If (i - 2^d) < 0, the element keeps its current value

### Hillis-Steele Step-by-Step for 8 Elements

```
Input:        [3]   [1]   [7]   [0]   [4]   [1]   [6]   [3]
Index:         0     1     2     3     4     5     6     7

Step d=0 (offset=1):
  Each element i adds element i-1 (if it exists).
  Read from a SECOND buffer to avoid race conditions.

  i=0: no i-1          -> 3
  i=1: a[1]+a[0] = 1+3 -> 4
  i=2: a[2]+a[1] = 7+1 -> 8
  i=3: a[3]+a[2] = 0+7 -> 7
  i=4: a[4]+a[3] = 4+0 -> 4
  i=5: a[5]+a[4] = 1+4 -> 5
  i=6: a[6]+a[5] = 6+1 -> 7
  i=7: a[7]+a[6] = 3+6 -> 9

After d=0:    [3]   [4]   [8]   [7]   [4]   [5]   [7]   [9]
               |     |     |     |     |     |     |     |
             sum    sum   sum   sum   sum   sum   sum   sum
             of 1  of 2  of 2  of 2  of 2  of 2  of 2  of 2

Step d=1 (offset=2):
  Each element i adds element i-2 (if it exists).

  i=0: no i-2                -> 3
  i=1: no i-2 (would be -1) -> 4
  i=2: b[2]+b[0] = 8+3      -> 11
  i=3: b[3]+b[1] = 7+4      -> 11
  i=4: b[4]+b[2] = 4+8      -> 12
  i=5: b[5]+b[3] = 5+7      -> 12
  i=6: b[6]+b[4] = 7+4      -> 11
  i=7: b[7]+b[5] = 9+5      -> 14

After d=1:    [3]   [4]  [11]  [11]  [12]  [12]  [11]  [14]
               |     |     |     |     |     |     |     |
             sum    sum   sum   sum   sum   sum   sum   sum
             of 1  of 2  of 3  of 4  of 4  of 4  of 4  of 4

Step d=2 (offset=4):
  Each element i adds element i-4 (if it exists).

  i=0: no i-4                 ->  3
  i=1: no i-4                 ->  4
  i=2: no i-4                 -> 11
  i=3: no i-4                 -> 11
  i=4: c[4]+c[0] = 12+3      -> 15
  i=5: c[5]+c[1] = 12+4      -> 16
  i=6: c[6]+c[2] = 11+11     -> 22
  i=7: c[7]+c[3] = 14+11     -> 25

After d=2:    [3]   [4]  [11]  [11]  [15]  [16]  [22]  [25]
               |     |     |     |     |     |     |     |
             sum    sum   sum   sum   sum   sum   sum   sum
             of 1  of 2  of 3  of 4  of 5  of 6  of 7  of 8
                                                         ^
                                             INCLUSIVE SCAN COMPLETE!
```

**Steps**: log2(8) = 3  (optimal number of steps)
**Total work**: 3 * 8 = 24 additions  (versus only 7 needed sequentially)
**Work efficiency**: O(N log N) -- wasteful, but the low step count means
good performance for small N.

---

## Algorithm 2: Blelloch (Exclusive Scan)

This algorithm is **work-efficient** (O(N) total operations) but uses
**more parallel steps** (2 * log2 N). It has two phases:

1. **Up-sweep (reduce)** -- build a reduction tree from leaves to root
2. **Down-sweep** -- traverse from root to leaves, distributing prefix sums

### Blelloch Up-Sweep (Reduce Phase)

Same as a parallel reduction. Build partial sums bottom-up:

```
Input:        [3]   [1]   [7]   [0]   [4]   [1]   [6]   [3]
Index:         0     1     2     3     4     5     6     7

Up-sweep step d=0 (stride=2):
  For i = 1, 3, 5, 7:   a[i] += a[i-1]

              [3]  [3+1]  [7]  [7+0]  [4]  [4+1]  [6]  [6+3]
              [3]   [4]   [7]   [7]   [4]   [5]   [6]   [9]

Up-sweep step d=1 (stride=4):
  For i = 3, 7:   a[i] += a[i-2]

              [3]   [4]   [7] [4+7]   [4]   [5]   [6]  [5+9]
              [3]   [4]   [7]  [11]   [4]   [5]   [6]  [14]

Up-sweep step d=2 (stride=8):
  For i = 7:   a[i] += a[i-4]

              [3]   [4]   [7]  [11]   [4]   [5]   [6] [11+14]
              [3]   [4]   [7]  [11]   [4]   [5]   [6]  [25]
                                                         ^
                                              Total sum at root
```

The tree structure during up-sweep:

```
                              [25]                     d=2
                            /      \
                        [11]        [14]               d=1
                       /    \      /    \
                     [4]    [7]  [5]    [9]            d=0
                    / \    / \   / \    / \
                   3   1  7   0 4   1  6   3           input
```

### Blelloch Down-Sweep (Distribute Phase)

Set the root to 0 (identity), then push prefix sums downward:

At each node, the left child gets the parent's value, and the right child
gets the parent's value plus the old left child value.

```
Start: set a[7] = 0 (identity element for addition)

              [3]   [4]   [7]  [11]   [4]   [5]   [6]   [0]

Down-sweep step d=2 (stride=8):
  Swap and add at index 7:
    temp = a[3]         = 11
    a[3] = a[7]         = 0      (left child gets parent)
    a[7] = a[7] + temp  = 0+11   (right child gets parent + old left)
           = 11

              [3]   [4]   [7]   [0]   [4]   [5]   [6]  [11]

Down-sweep step d=1 (stride=4):
  Swap and add at index 3:
    temp = a[1]         = 4
    a[1] = a[3]         = 0
    a[3] = a[3] + temp  = 0+4 = 4

  Swap and add at index 7:
    temp = a[5]         = 5
    a[5] = a[7]         = 11
    a[7] = a[7] + temp  = 11+5 = 16

              [3]   [0]   [7]   [4]   [4]  [11]   [6]  [16]

Down-sweep step d=0 (stride=2):
  Swap and add at index 1:
    temp = a[0] = 3,  a[0] = a[1] = 0,  a[1] = a[1]+temp = 0+3 = 3

  Swap and add at index 3:
    temp = a[2] = 7,  a[2] = a[3] = 4,  a[3] = a[3]+temp = 4+7 = 11

  Swap and add at index 5:
    temp = a[4] = 4,  a[4] = a[5] = 11, a[5] = a[5]+temp = 11+4 = 15

  Swap and add at index 7:
    temp = a[6] = 6,  a[6] = a[7] = 16, a[7] = a[7]+temp = 16+6 = 22

Result:       [0]   [3]   [4]  [11]  [11]  [15]  [16]  [22]
               ^     ^     ^     ^     ^     ^     ^     ^
              id   0+3  0+3+1  ...                    sum of first 7
                                      EXCLUSIVE SCAN COMPLETE!
```

**Verify**: for input [3,1,7,0,4,1,6,3]:
- Exclusive scan = [0, 3, 4, 11, 11, 15, 16, 22] (correct!)
- Total sum = 22 + 3 = 25

**Steps**: 2 * log2(8) = 6  (more steps than Hillis-Steele)
**Total work**: 2 * (N-1) = 14 additions  (versus 24 for Hillis-Steele)
**Work efficiency**: O(N) -- optimal!

---

## Comparison: Hillis-Steele vs. Blelloch

```
                 Hillis-Steele          Blelloch
                 ---------------        ---------------
Type:            Inclusive scan          Exclusive scan
Steps:           log2(N)                2 * log2(N)
Total work:      O(N log N)             O(N)
Best for:        Small arrays           Large arrays
                 (step-bound)           (work-bound)
```

For GPUs with thousands of cores, the work-efficient Blelloch algorithm
typically wins for large arrays because there is enough parallelism in
each step to keep the GPU busy, and the total work is half or less.

---

## Handling Arrays Larger Than One Block

A single thread block can scan at most ~1024-2048 elements (limited by
shared memory and max threads per block). For millions of elements, we
use a three-phase **scan-then-propagate** approach:

```
Phase 1: Block-level scan
  Each block scans its own chunk independently.
  Save each block's total sum into an auxiliary array.

  Input:   [--- block 0 ---][--- block 1 ---][--- block 2 ---][--- block 3 ---]
  Scanned: [partial scan 0 ][partial scan 1 ][partial scan 2 ][partial scan 3 ]
  Totals:  [    T0         ][    T1         ][    T2         ][    T3         ]

Phase 2: Scan the block totals
  Exclusive scan of [T0, T1, T2, T3] -> [0, T0, T0+T1, T0+T1+T2]
  (If there are many blocks, this scan itself may need to be recursive.)

  Block offsets: [0, T0, T0+T1, T0+T1+T2]

Phase 3: Propagate offsets back
  Add the block offset to every element in the corresponding block.

  Final:  [partial scan 0 + 0]
          [partial scan 1 + T0]
          [partial scan 2 + T0+T1]
          [partial scan 3 + T0+T1+T2]
                = COMPLETE SCAN
```

This is exactly the strategy used by `thrust::exclusive_scan` and CUB.

---

## Bank Conflicts in Scan

Shared memory on NVIDIA GPUs is divided into 32 banks. Two threads
accessing the same bank (but different addresses) cause a **bank conflict**,
serializing those accesses.

In both Hillis-Steele and Blelloch, the stride pattern can create
systematic bank conflicts. For Blelloch:

- In the up-sweep with stride 2, threads 0 and 16 both access bank 0
- With stride 4, threads 0, 8, 16, 24 all hit bank 0
- This creates 2-way, 4-way, ... 32-way bank conflicts!

### Padding Trick to Avoid Bank Conflicts

Add one padding element for every 32 elements in shared memory:

```
Without padding (bank conflicts):
  Logical index:  0  1  2  ... 31  32  33  ... 63
  Shared index:   0  1  2  ... 31  32  33  ... 63
  Bank:           0  1  2  ... 31   0   1  ...  31   <-- conflicts!

With padding (conflict-free):
  Logical index:  0  1  2  ... 31  32  33  ... 63
  Shared index:   0  1  2  ... 31  33  34  ... 64   (skip index 32)
  Bank:           0  1  2  ... 31   1   2  ...  0   <-- no conflicts!

  Macro: CONFLICT_FREE_OFFSET(i) = (i) / 32
  Padded index = i + CONFLICT_FREE_OFFSET(i)
```

This adds ~3% memory overhead but can improve scan performance by 2x
on conflict-heavy patterns.

---

## Files in This Chapter

| File                    | Description                                          |
|-------------------------|------------------------------------------------------|
| `scan_algorithms.cu`    | Hillis-Steele and Blelloch side-by-side, benchmarked |
| `scan_large.cu`         | Three-phase scan for arrays of millions of elements  |
| `stream_compaction.cu`  | Practical application: remove zeros using scan       |
| `Makefile`              | Build all three programs                             |

---

## Key Takeaways

1. **Scan is everywhere** -- it is the second most important parallel primitive
   after reduction. Every parallel sort, compaction, and allocation uses it.

2. **Hillis-Steele** is simple and step-efficient (log N steps) but does
   O(N log N) work -- good for small arrays.

3. **Blelloch** is work-efficient (O(N) work) with 2*log N steps -- better
   for large arrays where the GPU has enough parallelism per step.

4. **Large arrays** require a three-phase scan-then-propagate approach:
   block scans, scan of totals, then propagate offsets.

5. **Bank conflict padding** can double performance for scan kernels that
   use shared memory with power-of-two strides.

6. **Stream compaction** (filter/select) is the canonical application of scan
   and is used in physics simulation, ray tracing, and ML sparse ops.
