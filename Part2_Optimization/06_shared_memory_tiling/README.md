# Chapter 06: Shared Memory and Tiling

## Overview

Shared memory is **the** most important optimization tool in CUDA programming.
It is a small, fast, programmer-managed cache that lives on each SM (Streaming
Multiprocessor). Unlike global memory (hundreds of cycles latency), shared
memory has latency comparable to registers (a few cycles) -- but unlike
registers, it is **shared** across all threads in a block.

**Your GPU (Quadro P4200, CC 6.1):**
- 48 KB of shared memory per SM (configurable, can trade with L1)
- 96 KB total per SM split between shared memory and L1 cache
- 18 SMs total
- 32 banks, each 4 bytes wide

---

## Shared Memory Architecture

### Where It Lives

```
 GPU Chip (Quadro P4200)
 +-----------------------------------------------------------------+
 |                                                                 |
 |  SM 0                    SM 1                    SM 17          |
 |  +------------------+   +------------------+   +------------+  |
 |  | Warp schedulers  |   | Warp schedulers  |   |    ...     |  |
 |  | 128 CUDA cores   |   | 128 CUDA cores   |   |            |  |
 |  |                  |   |                  |   |            |  |
 |  | Register File    |   | Register File    |   |            |  |
 |  | (256 KB)         |   | (256 KB)         |   |            |  |
 |  |                  |   |                  |   |            |  |
 |  | +==============+ |   | +==============+ |   |            |  |
 |  | | SHARED MEM   | |   | | SHARED MEM   | |   |            |  |
 |  | | 48 KB        | |   | | 48 KB        | |   |            |  |
 |  | | (~5 cycles)  | |   | | (~5 cycles)  | |   |            |  |
 |  | +==============+ |   | +==============+ |   |            |  |
 |  |                  |   |                  |   |            |  |
 |  | L1 Cache (48 KB) |   | L1 Cache (48 KB) |   |            |  |
 |  +------------------+   +------------------+   +------------+  |
 |                                                                 |
 |  +-----------------------------------------------------------+  |
 |  |                    L2 Cache (2 MB)                         |  |
 |  +-----------------------------------------------------------+  |
 |                                                                 |
 |  +-----------------------------------------------------------+  |
 |  |               Global Memory (8 GB GDDR5)                  |  |
 |  |               ~400 cycles latency                         |  |
 |  +-----------------------------------------------------------+  |
 +-----------------------------------------------------------------+

 Memory Hierarchy Latency:
   Registers    ~1 cycle     per-thread     256 KB per SM
   Shared Mem   ~5 cycles    per-block      48 KB per SM
   L1 Cache     ~30 cycles   per-SM         48 KB per SM
   L2 Cache     ~200 cycles  global         2 MB total
   Global Mem   ~400 cycles  global         8 GB total
```

### Key Properties of Shared Memory

1. **Per-SM**: Each SM has its own shared memory. Blocks on different SMs
   cannot share it.
2. **Per-block lifetime**: Shared memory is allocated when a block is launched
   and freed when the block finishes. All threads in the block share it.
3. **Programmer-managed**: Unlike L1/L2, YOU decide what goes in shared memory.
4. **Bank-organized**: Divided into 32 banks for parallel access (see below).
5. **Limits occupancy**: If your kernel uses a lot of shared memory, fewer
   blocks can run simultaneously on each SM.

---

## Bank Conflicts

Shared memory is divided into **32 banks**. Each bank is 4 bytes wide and can
serve one address per cycle. When multiple threads in a warp access different
addresses in the SAME bank, those accesses are **serialized** -- this is a
**bank conflict**.

### Bank Layout

```
 Shared Memory Banks (32 banks, 4 bytes each = 128 bytes per "row")

 Address:  0    4    8   12   16   20  ...  120  124  128  132  ...
 Bank:     0    1    2    3    4    5  ...   30   31    0    1  ...

 +------+------+------+------+------+------+     +------+------+
 | B0   | B1   | B2   | B3   | B4   | B5   | ... | B30  | B31  |  Row 0
 |addr 0|addr 4|addr 8|adr12 |adr16 |adr20 |     |ad120 |ad124 |
 +------+------+------+------+------+------+     +------+------+
 | B0   | B1   | B2   | B3   | B4   | B5   | ... | B30  | B31  |  Row 1
 |ad128 |ad132 |ad136 |ad140 |ad144 |ad148 |     |ad248 |ad252 |
 +------+------+------+------+------+------+     +------+------+
 | B0   | B1   | B2   | B3   | B4   | B5   | ... | B30  | B31  |  Row 2
 |ad256 |ad260 |ad264 |ad268 |ad272 |ad276 |     |ad376 |ad380 |
 +------+------+------+------+------+------+     +------+------+
   :      :      :      :      :      :              :      :

 Bank number = (byte_address / 4) % 32

 Example: float array smem[256]
   smem[0]  -> bank 0    smem[32] -> bank 0    (same bank!)
   smem[1]  -> bank 1    smem[33] -> bank 1
   smem[2]  -> bank 2    smem[34] -> bank 2
   ...
   smem[31] -> bank 31   smem[63] -> bank 31
```

### Conflict Patterns

```
 NO CONFLICT (stride-1): Each thread accesses a different bank
 ================================================================

 Thread:    T0   T1   T2   T3   T4  ...  T30  T31
 Accesses:  |    |    |    |    |         |    |
            v    v    v    v    v         v    v
 Bank:     B0   B1   B2   B3   B4  ...  B30  B31

 smem[tid] -- thread i accesses bank i.  PERFECT!
 All 32 banks serve in parallel -> 1 cycle.


 2-WAY CONFLICT (stride-2): Two threads hit each bank
 ================================================================

 Thread:    T0   T1   T2   T3  ...  T15  T16  T17  ...  T31
 Accesses:  |    |    |    |         |    |    |         |
            v    v    v    v         v    v    v         v
 Bank:     B0   B2   B4   B6  ...  B30  B0   B2  ...   B30
            ^                             ^
            |_____ CONFLICT! _____________|

 smem[tid * 2] -- threads 0 and 16 both hit bank 0,
                  threads 1 and 17 both hit bank 2, etc.
 Must serialize: 2 cycles instead of 1.


 32-WAY CONFLICT (stride-32): ALL threads hit the same bank
 ================================================================

 Thread:    T0   T1   T2   T3  ...  T31
 Accesses:  |    |    |    |         |
            v    v    v    v         v
 Bank:     B0   B0   B0   B0  ...  B0     <-- ALL BANK 0!

 smem[tid * 32] -- every thread hits bank 0.
 Must serialize: 32 cycles instead of 1.   TERRIBLE!


 BROADCAST (special case): All threads read the SAME address
 ================================================================

 Thread:    T0   T1   T2   T3  ...  T31
 Accesses:  |    |    |    |         |
            +----+----+----+-...-+---+
            v
 Bank:     B0 (address 0)

 smem[0] for ALL threads -- same address, not a conflict!
 Hardware broadcasts the value -> 1 cycle.
 (Only works when ALL threads access the SAME address in a bank)
```

### The Padding Trick to Avoid Bank Conflicts

When you have a 2D shared memory array where column access causes conflicts,
add 1 extra element per row:

```
 WITHOUT padding: float smem[32][32]
 Row 0: smem[0][0]=B0  smem[0][1]=B1  ... smem[0][31]=B31
 Row 1: smem[1][0]=B0  smem[1][1]=B1  ... smem[1][31]=B31

 Column access: smem[0][0], smem[1][0], smem[2][0] ...
 All in bank 0!  32-way conflict!


 WITH padding: float smem[32][32 + 1]   <-- +1 padding
 Row 0: smem[0][0]=B0  smem[0][1]=B1  ... smem[0][31]=B31  [pad=B0]
 Row 1: smem[1][0]=B1  smem[1][1]=B2  ... smem[1][31]=B0   [pad=B1]
 Row 2: smem[2][0]=B2  smem[2][1]=B3  ... smem[2][31]=B1   [pad=B2]

 Column access: smem[0][0]=B0, smem[1][0]=B1, smem[2][0]=B2 ...
 Each row shifts by one bank.  NO CONFLICT!
```

---

## Tiling Strategy

The core idea: **load a chunk (tile) of data into shared memory, then reuse it
many times.** This turns expensive global memory accesses into cheap shared
memory accesses.

### The Tiling Pattern

```
 GLOBAL MEMORY (slow, ~400 cycles)           SHARED MEMORY (fast, ~5 cycles)
 +-----------------------------------+       +------------------+
 |                                   |       |                  |
 |  +-------+                       |  (1)  |  +-------+       |
 |  | Tile  |------ LOAD ------------------>|  | Tile  |       |
 |  +-------+                       |       |  | Copy  |       |
 |  |       |                       |       |  +-------+       |
 |  |       |                       |       |                  |
 |  | Rest  |                       |       |  __syncthreads() |
 |  | of    |                       |       |                  |
 |  | data  |                       |  (2)  |  COMPUTE using   |
 |  |       |                       |       |  shared memory   |
 |  |       |                       |       |  (many accesses, |
 |  |       |                       |       |   all fast!)     |
 |  +-------+                       |       |                  |
 |                                   |       |  __syncthreads() |
 +-----------------------------------+       |                  |
                                             |  (3) Move to     |
                                             |      next tile   |
                                             +------------------+

 Tiling Loop (pseudocode):

   for each tile T in data:
       1. ALL threads cooperate to load tile T into shared memory
          (coalesced global reads -> fast)

       2. __syncthreads()      <-- CRITICAL! Wait for load to finish

       3. ALL threads compute using shared memory
          (many reads from shared mem -> fast, no global traffic)

       4. __syncthreads()      <-- CRITICAL! Wait for compute to finish
                                    before overwriting tile

   Write final result to global memory
```

### Why Tiling Works: Data Reuse

```
 Example: Matrix Transpose of 1024x1024

 WITHOUT tiling:
   Each element read from global memory, written to global memory.
   Writes are non-coalesced (strided) -> SLOW.
   Total global memory transactions: very high due to non-coalesced writes.

 WITH tiling (32x32 tiles):
   1. Load 32x32 tile from global mem (coalesced reads)      -> FAST
   2. Store in shared memory
   3. Read from shared memory in transposed order             -> FAST (~5 cycles)
   4. Write to global memory (coalesced writes)               -> FAST
   Both reads AND writes are coalesced!

 Example: Matrix Multiplication C = A * B, all NxN

 WITHOUT tiling:
   Each element of C requires reading an entire row of A and column of B.
   Element C[i][j] reads A[i][0..N-1] and B[0..N-1][j] from global memory.
   N elements of A, N elements of B -> 2N global reads per output element.
   Total: N^2 * 2N = 2N^3 global memory reads.

 WITH tiling (TILE_SIZE = T):
   Load TxT tiles of A and B into shared memory.
   Each tile load: T^2 elements from global memory.
   Each tile load serves T^2 output elements, each using T values.
   Reuse factor: each loaded element is used T times!
   Total: 2N^3 / T global memory reads.
   For T=32: 32x reduction in global memory traffic!
```

---

## __syncthreads() -- Why It Is Critical

`__syncthreads()` is a **barrier**: all threads in the block must reach it
before any thread can proceed past it.

```
 WITHOUT __syncthreads() -- RACE CONDITION!

 Thread 0:  LOAD smem[0] ----+
 Thread 1:  LOAD smem[1] ----|---+
 Thread 2:  LOAD smem[2] ----|---|--- READ smem[5]  <-- OOPS! Thread 5
 Thread 3:  LOAD smem[3] ----|---|--- READ smem[7]      hasn't written
 Thread 4:  LOAD smem[4] ----|---|                       smem[5] yet!
 Thread 5:                   |   +--- (still loading!)
    ...                      v

 WITH __syncthreads() -- CORRECT

 Thread 0:  LOAD smem[0] --+
 Thread 1:  LOAD smem[1] --+
 Thread 2:  LOAD smem[2] --+-- BARRIER --+-- READ smem[5]  (safe!)
 Thread 3:  LOAD smem[3] --+             +-- READ smem[7]  (safe!)
 Thread 4:  LOAD smem[4] --+             |
 Thread 5:  LOAD smem[5] --+             +-- all data guaranteed
    ...                                       to be in shared memory
```

**Rules for __syncthreads():**
1. Call it AFTER cooperative load, BEFORE any thread reads loaded data.
2. Call it AFTER compute phase, BEFORE overwriting shared memory with next tile.
3. NEVER put it inside a branch where some threads might not reach it.
   (Deadlock! The barrier waits for ALL threads -- if some are diverged away,
   the barrier will hang forever.)

---

## Static vs Dynamic Shared Memory

### Static Allocation (size known at compile time)

```c
__global__ void kernel() {
    __shared__ float smem[256];   // 256 * 4 bytes = 1024 bytes
    // All threads in the block share this array
}
```

### Dynamic Allocation (size determined at launch time)

```c
__global__ void kernel() {
    extern __shared__ float smem[];  // Size not specified here!
    // Size is passed via the 3rd kernel launch parameter
}

// At launch:
int sharedBytes = 256 * sizeof(float);
kernel<<<blocks, threads, sharedBytes>>>();
//                        ^^^^^^^^^^^^ 3rd parameter = shared mem size
```

Dynamic is useful when tile size is a runtime parameter.

---

## Shared Memory and Occupancy

Each SM has a fixed amount of shared memory (48 KB on your P4200). If each
block uses a lot of shared memory, fewer blocks can be resident simultaneously.

```
 Example on Quadro P4200 (48 KB shared mem per SM):

 Block uses 8 KB  -> 48 / 8  = 6 blocks per SM  (limited by other factors)
 Block uses 16 KB -> 48 / 16 = 3 blocks per SM
 Block uses 24 KB -> 48 / 24 = 2 blocks per SM
 Block uses 48 KB -> 48 / 48 = 1 block per SM   (low occupancy!)

 CC 6.1 also limits to 32 resident blocks per SM and 2048 threads per SM.

 For a 32x32 tile (4 KB), you have plenty of room for multiple blocks.
 For a 64x64 tile (16 KB), only 3 blocks fit.

 Trade-off: bigger tiles = more reuse but lower occupancy.
 Usually 32x32 (4 KB) is the sweet spot.
```

---

## Files in This Chapter

| File | Description |
|------|-------------|
| `bank_conflicts.cu` | Demonstrate and measure bank conflicts with different stride patterns, plus the padding fix |
| `tiled_transpose.cu` | Classic optimized matrix transpose: naive vs tiled vs tiled+padded |
| `tiled_matmul_intro.cu` | Preview of tiled matrix multiplication (detailed treatment in Ch 12) |

---

## Key Takeaways

1. **Shared memory is per-SM, per-block, programmer-managed, and FAST (~5 cycles).**
2. **Bank conflicts serialize accesses** -- avoid stride-2, stride-32 patterns.
   Use the **+1 padding trick** for 2D arrays accessed by column.
3. **Tiling = load a chunk into shared memory, reuse it many times.**
   This is THE fundamental CUDA optimization pattern.
4. **__syncthreads() is mandatory** between cooperative load and compute phases.
   Forgetting it causes race conditions. Putting it in divergent branches causes deadlocks.
5. **Shared memory usage limits occupancy** -- balance tile size against
   the number of blocks that can run simultaneously.
