# Chapter 03: Thread Hierarchy

## Overview

CUDA organizes parallel threads into a two-level hierarchy: **grids** and **blocks**.
Understanding this hierarchy is essential for writing correct and efficient GPU code.

Your hardware: **Quadro P4200** (Compute Capability 6.1)
- 18 Streaming Multiprocessors (SMs)
- 128 CUDA cores per SM (2304 total)
- Max 1024 threads per block
- Max 2048 threads per SM
- Max 32 blocks per SM (CC 6.x)

---

## 1. The Grid-Block-Thread Hierarchy

When you launch a kernel, you specify a **grid** of **blocks**, and each block
contains **threads**.

```
kernel<<<gridDim, blockDim>>>(args...);
            |          |
            |          +-- threads per block (1D, 2D, or 3D)
            +------------- blocks per grid   (1D, 2D, or 3D)
```

### 1D Grid and Block (simplest case)

```
Grid (8 blocks)
+-------+-------+-------+-------+-------+-------+-------+-------+
|Block 0|Block 1|Block 2|Block 3|Block 4|Block 5|Block 6|Block 7|
+-------+-------+-------+-------+-------+-------+-------+-------+

Each block has 4 threads (blockDim.x = 4):

Block 0                    Block 1                    Block 2
+----+----+----+----+      +----+----+----+----+      +----+----+----+----+
| T0 | T1 | T2 | T3 |      | T0 | T1 | T2 | T3 |      | T0 | T1 | T2 | T3 |
+----+----+----+----+      +----+----+----+----+      +----+----+----+----+

Global thread ID = blockIdx.x * blockDim.x + threadIdx.x

Block 0: 0*4+0=0, 0*4+1=1, 0*4+2=2, 0*4+3=3
Block 1: 1*4+0=4, 1*4+1=5, 1*4+2=6, 1*4+3=7
Block 2: 2*4+0=8, 2*4+1=9, 2*4+2=10, 2*4+3=11
...
```

### 2D Grid and Block (common for image processing)

```
Grid (3x2 blocks):
            blockIdx.x -->
           0           1           2
      +----------+----------+----------+
   0  | Block    | Block    | Block    |
      | (0,0)    | (1,0)    | (2,0)    |  blockIdx.y
      +----------+----------+----------+      |
   1  | Block    | Block    | Block    |      v
      | (0,1)    | (1,1)    | (2,1)    |
      +----------+----------+----------+

Each block has 4x4 threads:
      threadIdx.x -->
     0    1    2    3
  +----+----+----+----+
0 |T0,0|T1,0|T2,0|T3,0|   threadIdx.y
  +----+----+----+----+      |
1 |T0,1|T1,1|T2,1|T3,1|      v
  +----+----+----+----+
2 |T0,2|T1,2|T2,2|T3,2|
  +----+----+----+----+
3 |T0,3|T1,3|T2,3|T3,3|
  +----+----+----+----+

Global 2D coordinates:
  col = blockIdx.x * blockDim.x + threadIdx.x
  row = blockIdx.y * blockDim.y + threadIdx.y

Linear index (row-major):
  idx = row * width + col
```

### 3D Grid and Block (used for volumetric data)

```
dim3 blockDim(Bx, By, Bz);
dim3 gridDim(Gx, Gy, Gz);

Global 3D coordinates:
  x = blockIdx.x * blockDim.x + threadIdx.x
  y = blockIdx.y * blockDim.y + threadIdx.y
  z = blockIdx.z * blockDim.z + threadIdx.z

Linear index:
  idx = z * (width * height) + y * width + x
```

---

## 2. Warps: The Fundamental Execution Unit

A **warp** is a group of **32 consecutive threads** within a block.
The GPU hardware fetches one instruction and executes it across all 32 threads
in the warp simultaneously. This is called **SIMT** (Single Instruction,
Multiple Threads).

### How Threads Map to Warps

Threads are grouped into warps by their linear thread index within the block:

```
Linear thread index = threadIdx.x
                    + threadIdx.y * blockDim.x
                    + threadIdx.z * blockDim.x * blockDim.y

Warp ID = linear_thread_index / 32
Lane ID = linear_thread_index % 32
```

### Example: Block of 256 Threads = 8 Warps

```
Block (256 threads, e.g. blockDim = 256)

Warp 0:  threads [  0 ..  31]   Lane 0  Lane 1  Lane 2  ... Lane 31
Warp 1:  threads [ 32 ..  63]   Lane 0  Lane 1  Lane 2  ... Lane 31
Warp 2:  threads [ 64 ..  95]   Lane 0  Lane 1  Lane 2  ... Lane 31
Warp 3:  threads [ 96 .. 127]   Lane 0  Lane 1  Lane 2  ... Lane 31
Warp 4:  threads [128 .. 159]   Lane 0  Lane 1  Lane 2  ... Lane 31
Warp 5:  threads [160 .. 191]   Lane 0  Lane 1  Lane 2  ... Lane 31
Warp 6:  threads [192 .. 223]   Lane 0  Lane 1  Lane 2  ... Lane 31
Warp 7:  threads [224 .. 255]   Lane 0  Lane 1  Lane 2  ... Lane 31

Total: 256 / 32 = 8 warps
```

### SIMT Execution

All 32 threads in a warp execute the **same instruction** at the **same time**:

```
Instruction stream:        Warp execution:

  LOAD  R1, [addr]    -->  All 32 threads load from their respective addresses
  ADD   R1, R1, R2    -->  All 32 threads add simultaneously
  STORE [addr], R1    -->  All 32 threads store simultaneously
```

---

## 3. Warp Divergence

When threads in the same warp take **different branches** of an if/else,
the warp must execute BOTH paths, disabling threads that don't belong to
each path. This is called **warp divergence**.

### No Divergence (all threads same path)

```
if (condition_true_for_all_threads) {
    // path A
}

Warp execution:
Lane:  0  1  2  3  4  5  6  7  ... 31
       A  A  A  A  A  A  A  A  ... A     <-- all active, full speed
```

### Divergence (threads split across paths)

```
if (threadIdx.x % 2 == 0) {
    // path A (even threads)
} else {
    // path B (odd threads)
}

Warp execution -- TWO passes required:

Pass 1 (path A):
Lane:  0  1  2  3  4  5  6  7  ... 31
       A  -  A  -  A  -  A  -  ... -     <-- odd lanes IDLE (wasted cycles!)

Pass 2 (path B):
Lane:  0  1  2  3  4  5  6  7  ... 31
       -  B  -  B  -  B  -  B  ... B     <-- even lanes IDLE (wasted cycles!)

Result: takes ~2x as long as non-divergent code
```

### Key Rule

Divergence only matters **within a warp**. If entire warps take one path
and other entire warps take another path, there is NO divergence penalty.

```
// This is FINE -- divergence is at the warp level, not thread level:
if (threadIdx.x / 32 < 4) {   // warps 0-3 go one way
    // path A
} else {                       // warps 4-7 go another way
    // path B
}
// Each warp is uniform -> NO divergence!
```

---

## 4. How Blocks Map to SMs (Hardware View)

```
GPU: Quadro P4200 (18 SMs)
+------+------+------+------+------+------+------+------+------+
| SM 0 | SM 1 | SM 2 | SM 3 | SM 4 | SM 5 | SM 6 | SM 7 | SM 8 |
+------+------+------+------+------+------+------+------+------+
| SM 9 |SM 10 |SM 11 |SM 12 |SM 13 |SM 14 |SM 15 |SM 16 |SM 17 |
+------+------+------+------+------+------+------+------+------+

Blocks are assigned to SMs by the hardware scheduler:

SM 0:  [Block 0] [Block 1] [Block 2] ...   (up to 32 blocks max)
SM 1:  [Block 3] [Block 4] [Block 5] ...   (up to 2048 threads max)
SM 2:  [Block 6] [Block 7] [Block 8] ...
...

Rules (CC 6.1):
- Max 32 blocks per SM
- Max 2048 threads per SM
- Max 1024 threads per block
- Max 64 warps per SM
```

### Occupancy Examples

```
Block size 256 (8 warps):
  2048 / 256 = 8 blocks per SM (limited by threads)
  8 blocks * 8 warps = 64 warps -> 100% occupancy

Block size 128 (4 warps):
  2048 / 128 = 16 blocks per SM (limited by threads)
  16 blocks * 4 warps = 64 warps -> 100% occupancy

Block size 1024 (32 warps):
  2048 / 1024 = 2 blocks per SM
  2 blocks * 32 warps = 64 warps -> 100% occupancy

Block size 512 (16 warps):
  2048 / 512 = 4 blocks per SM
  4 blocks * 16 warps = 64 warps -> 100% occupancy

Block size 32 (1 warp):
  32 blocks per SM (hit block limit before thread limit!)
  32 blocks * 1 warp = 32 warps -> 50% occupancy  <-- BAD
```

---

## 5. Choosing Block Size: Practical Rules

1. **Always use a multiple of 32** (warp size). Non-multiples waste lanes
   in the last warp.

2. **128 or 256 are the most common choices:**
   - 256: simple, gives 100% occupancy, good for most kernels
   - 128: more blocks per SM, can help hide latency with more scheduling options

3. **Never use fewer than 64 threads per block** unless you have a specific reason.

4. **Never exceed 1024 threads per block** (hardware limit).

5. **Register and shared memory pressure** can reduce occupancy below the
   theoretical maximum. Use `nvcc --ptxas-options=-v` to check.

6. **When in doubt, start with 256** and profile.

---

## 6. Files in This Chapter

| File | Description |
|------|-------------|
| `indexing_2d.cu` | 2D thread indexing demo with row-major and column-major layouts |
| `warp_divergence.cu` | Measure the performance cost of warp divergence |
| `image_blur.cu` | Practical 2D example: box blur with shared memory optimization |
| `Makefile` | Build all examples with `make` |

### Build and Run

```bash
make              # build all
make run          # build and run all
make indexing     # build just indexing_2d
make divergence   # build just warp_divergence
make blur         # build just image_blur
make clean        # remove binaries
```
