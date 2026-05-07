# Chapter 12: Matrix Multiplication Optimization

## Why Matmul is THE Operation

Matrix multiplication (GEMM -- General Matrix Multiply) is the single most
important operation in deep learning. Almost everything reduces to matmul:

- **Fully-connected layers:** `Y = X * W + b` -- literally matmul
- **Convolutions:** im2col transforms conv into matmul (cuDNN does this)
- **Attention (Transformers):** `QK^T` and `(QK^T)V` -- two matmuls per head
- **Batch normalization:** matrix-vector ops (small matmul variants)
- **Recurrent layers:** `h_t = f(W_h * h_{t-1} + W_x * x_t)` -- matmuls

NVIDIA GPUs are designed around matmul. Tensor Cores exist for matmul.
cuBLAS is the most optimized library on the planet, and it is a matmul library.
Understanding how matmul maps to GPU hardware is understanding GPU computing.

---

## The Basic Operation

```
C[M x N] = A[M x K] * B[K x N]

For each output element C[i][j]:

    C[i][j] = sum over k of A[i][k] * B[k][j]

              K
    A: i --> [a_i0  a_i1  a_i2 ... a_i(K-1)]
                |     |     |         |
    B: j       b_0j  b_1j  b_2j ... b_(K-1)j
                |     |     |         |
              multiply pairwise, then sum
                        |
                     C[i][j]
```

### Row-Major Memory Layout

```
A[M][K] in memory (row-major):

  Row 0: A[0][0] A[0][1] A[0][2] ... A[0][K-1]    <- contiguous
  Row 1: A[1][0] A[1][1] A[1][2] ... A[1][K-1]    <- contiguous
  ...
  Row M-1: ...

Address of A[i][k] = A + i*K + k

B[K][N] in memory (row-major):

  Row 0: B[0][0] B[0][1] B[0][2] ... B[0][N-1]
  Row 1: B[1][0] B[1][1] B[1][2] ... B[1][N-1]
  ...

Address of B[k][j] = B + k*N + j
```

### Arithmetic Intensity

```
FLOPs:  2*M*N*K  (one multiply + one add per (i,j,k) triple)
Data:   M*K + K*N + M*N  floats loaded (once each, ideally)

For square N x N matrices:
  FLOPs = 2*N^3
  Data  = 3*N^2 * 4 bytes = 12*N^2 bytes

  Arithmetic Intensity = 2*N^3 / (12*N^2) = N/6 FLOP/byte

For N=1024:  AI = 170 FLOP/byte  (VERY compute-bound)
For N=4096:  AI = 682 FLOP/byte

Quadro P4200:  5300 GFLOPS / 192 GB/s = 27.6 FLOP/byte to be compute-bound
=> Any matmul with N > 166 is compute-bound on this GPU
=> The goal: get as close to 5.3 TFLOPS as possible
```

---

## The Optimization Progression

This chapter follows the Simon Boehm / NVIDIA progression through 7 kernels,
each fixing a bottleneck from the previous one.

### Kernel 1: Naive (one thread per output element)

```
Grid: (N/32, M/32) blocks of (32, 32) threads

Each thread computes one C[row][col]:

  Thread (tx, ty) in block (bx, by):
    row = by * 32 + ty
    col = bx * 32 + tx

    float sum = 0;
    for (k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;

Memory access pattern for the inner loop:
  A[row][k]: threads in same row read same A element (broadcast OK)
  B[k][col]: threads in same warp read consecutive columns
             B[k][col], B[k][col+1], ... B[k][col+31]
             These ARE contiguous in memory -> coalesced!

  BUT: thread (ty=0,tx=0) and thread (ty=1,tx=0) are in same warp
       (warp = 32 consecutive threadIdx.x values for same threadIdx.y)
       Actually, 2D blocks are linearized: tid = ty * blockDim.x + tx
       So warp 0 = ty=0, tx=0..31  -> reads B[k][col..col+31] COALESCED!

  The naive kernel actually has coalesced B access if block is (32,32).
  But it reads A and B from global memory K times each -- terrible reuse.

Performance: ~200-400 GFLOPS (3-7% of peak)
```

### Kernel 2: Global Memory Coalescing (explicit)

```
What if we had the loop indices wrong? Consider a WRONG version:

  row = bx * 32 + tx    <- swapped!
  col = by * 32 + ty

  Now threads in same warp (consecutive tx) have consecutive ROW indices.
  B[k * N + col]: all threads in warp read SAME column -> no coalescing issue
  A[row * K + k]: threads read A[0*K+k], A[1*K+k], ... A[31*K+k]
                  These are STRIDED by K -- UNCOALESCED!

The coalesced version ensures threads in the same warp access
consecutive memory addresses. With row-major storage:
  - Consecutive threads should map to consecutive COLUMNS of C
  - This means consecutive threads read consecutive elements of B rows

This kernel makes the coalescing explicit and correct.

Performance: ~400-600 GFLOPS (matches naive if naive was already coalesced)
```

### Kernel 3: Shared Memory Tiling

```
Problem: In kernels 1-2, each element of A and B is loaded from global
memory O(N) times across all threads. Terrible bandwidth waste.

Solution: Tile the computation. Load a TILE_SIZE x TILE_SIZE block of A
and B into shared memory, compute partial results, move to next tile.

TILE_SIZE = 32 (one tile = one thread block)

  For tile t = 0, 1, ..., K/32 - 1:

    Step 1: Load tiles into shared memory
      tile_A[ty][tx] = A[row][t*32 + tx]
      tile_B[ty][tx] = B[t*32 + ty][col]

    Step 2: __syncthreads()

    Step 3: Accumulate partial dot product
      for (k = 0; k < 32; k++)
          sum += tile_A[ty][k] * tile_B[k][tx];

    Step 4: __syncthreads()

Diagram -- how tiling partitions the matrices:

  A [M x K]                     B [K x N]
  +-------+-------+----+        +-------+
  |       |       |    |        | tile  |
  | A_t0  | A_t1  |... |        | B_t0  |
  |       |       |    |        +-------+
  +-------+-------+----+        | tile  |
  |       |       |    |        | B_t1  |
  |  ...  |  ...  |    |        +-------+
  +-------+-------+----+        |  ...  |
                                +-------+

  C[i][j] = A_row_i * B_col_j
          = sum over tiles:  (A_tile_t's row i) dot (B_tile_t's col j)

  Each tile is 32x32 = 1024 elements = 4 KB in shared memory.
  Two tiles = 8 KB << 48 KB available on SM (CC 6.1).

Why this helps:
  Before: each thread loads K elements from A and K from B = 2K global loads
          32x32 threads doing this = 32*32*2K = 2048K global loads per block
  After:  each tile load = 32*32 = 1024 elements from A + 1024 from B
          K/32 tiles = (K/32) * 2048 = 64K global loads per block
          Reduction factor: 2048K / 64K = 32 = TILE_SIZE!

Performance: ~1000-1500 GFLOPS (19-28% of peak)
```

### Kernel 4: 1D Block Tiling (more work per thread)

```
Problem with Kernel 3: each thread does 32 multiply-adds per tile load,
but loads 2 values. Compute-to-load ratio = 32:2 = 16:1.
The GPU has way more compute than this uses.

Solution: each thread computes TM output elements (a column vector in C).
This increases work per thread without increasing shared memory loads.

  BK = 8  (tile width along K dimension)
  BM = 64 (tile height along M dimension -- block covers 64 rows)
  BN = 64 (tile width along N dimension)
  TM = 8  (each thread computes 8 elements vertically)

  Block size: (BN, BM/TM) = (64, 8) -> 512 threads? No:
  Actually: blockDim = (BN) threads, each computing TM rows.
  Or more practically: blockDim.x = BN, blockDim.y = BM/TM = 8

  Let's use a simpler 1D layout:
  blockDim = BM/TM * BN = 8 * 64 = 512 threads
  Each thread has index: threadRow = tid / BN, threadCol = tid % BN

  Each thread computes C[row + threadRow*TM + 0..TM-1][col + threadCol]

Diagram -- 1D block tiling:

  Block computes a BM x BN = 64x64 tile of C.
  Each thread computes TM = 8 elements in a column:

  C tile (64 x 64):
  +--+--+--+--+--+--+-- ... --+
  |t0|t1|t2|t3|  |  |         |  <- row group 0
  |t0|t1|t2|t3|  |  |         |     (TM=8 rows per thread)
  |t0|t1|t2|t3|  |  |         |
  |t0|t1|t2|t3|  |  |         |
  |t0|t1|t2|t3|  |  |         |
  |t0|t1|t2|t3|  |  |         |
  |t0|t1|t2|t3|  |  |         |
  |t0|t1|t2|t3|  |  |         |
  +--+--+--+--+--+--+-- ... --+

  t0 = thread 0: computes C[row0..row7][col0]
  t1 = thread 1: computes C[row0..row7][col1]
  ...

  For each BK-wide strip along K:
    - Load BM x BK = 64x8 tile of A into shared memory
    - Load BK x BN = 8x64 tile of B into shared memory
    - Each thread: load TM values from A tile column,
      1 value from B tile, accumulate TM products

  Compute per load:  TM * BK = 8 * 8 = 64 FMAs per thread per tile
  Loads per thread:  TM (from A) + BK (from B) = 16 loads
  Ratio: 64:16 = 4:1 (better than kernel 3's 1:1 per element)

Performance: ~1500-2500 GFLOPS (28-47% of peak)
```

### Kernel 5: 2D Block Tiling (the real GEMM approach)

```
Insight: extend 1D tiling to 2D. Each thread computes a TM x TN sub-tile.
This maximizes register reuse and is how real GEMM implementations work.

  BK = 8
  BM = 128, BN = 128
  TM = 8,  TN = 8
  Block: (BM/TM) * (BN/TN) = 16 * 16 = 256 threads

Diagram -- 2D register tile per thread:

  Block's C tile (128 x 128):
  +----+----+----+----+---- ... ----+
  | t00| t01| t02| t03|            |
  |8x8 |8x8 |8x8 |8x8 |           |   <- 16 thread-tiles across
  +----+----+----+----+---- ... ----+
  | t10| t11| t12| t13|            |
  |8x8 |8x8 |8x8 |8x8 |           |
  +----+----+----+----+---- ... ----+
  |    |    |    |    |             |
  ...                                   <- 16 thread-tiles down
  +----+----+----+----+---- ... ----+

  Each "t" box is one thread's 8x8 output sub-tile.

How one thread (computing TM x TN = 8 x 8 = 64 outputs) works:

  For each BK-wide strip along K:
    1. Cooperatively load A tile (BM x BK = 128x8) into smem_A
    2. Cooperatively load B tile (BK x BN = 8x128) into smem_B
    3. __syncthreads()
    4. For dotIdx = 0 to BK-1:
         Load TM values from smem_A column -> regA[0..TM-1]
         Load TN values from smem_B row    -> regB[0..TN-1]
         For i = 0 to TM-1:
           For j = 0 to TN-1:
             threadResults[i][j] += regA[i] * regB[j];  // outer product!
    5. __syncthreads()

  Register usage per thread:
    regA[8]  = 8 registers
    regB[8]  = 8 registers
    threadResults[8][8] = 64 registers
    Total: ~80 registers per thread (out of 255 max)

  Compute per tile:  TM * TN * BK = 8 * 8 * 8 = 512 FMAs
  Loads from smem:   TM * BK + TN * BK = 64 + 64 = 128 loads
  Ratio: 512:128 = 4:1 from shared memory (excellent)

  Global loads per tile: (128*8 + 8*128) / 256 threads = 8 loads/thread
  Compute per global load: 512/8 = 64 FMAs per global load (outstanding!)

Performance: ~2500-3500 GFLOPS (47-66% of peak)
```

### Kernel 6: Vectorized Loads (float4)

```
Problem: individual float loads use only 4 bytes of the 128-byte cache line.
Solution: use float4 to load 16 bytes at once -> 4x fewer load instructions.

  float4 tmp = reinterpret_cast<float4*>(&A[offset])[0];
  // Loads A[offset], A[offset+1], A[offset+2], A[offset+3] in one instruction

This reduces:
  - Number of load instructions (less instruction overhead)
  - Better utilization of memory bus width
  - Fewer shared memory bank conflicts (if carefully aligned)

The kernel is otherwise identical to Kernel 5 but with float4 loads
for both the global->shared and shared->register data movements.

Performance: ~3000-4000 GFLOPS (57-75% of peak)
```

### Kernel 7: Double Buffering

```
Problem: after loading a tile, ALL threads must __syncthreads() before
computing. Then after computing, ALL must sync again before loading next
tile. The load and compute phases are serialized:

  Timeline WITHOUT double buffering:
  |--load tile 0--|--sync--|--compute tile 0--|--sync--|--load tile 1--|--...

Solution: use TWO shared memory buffers. While computing from buffer A,
load the next tile into buffer B. Then swap.

  Timeline WITH double buffering:
  |--load tile 0 into buf A--|
                              |--compute buf A + load tile 1 into buf B--|
                              |--compute buf B + load tile 2 into buf A--|
                              |--compute buf A + load tile 3 into buf B--|
                              ...

Diagram:

  Shared Memory:
  +------------------+------------------+
  |   Buffer A       |   Buffer B       |
  | smem_A[BM][BK]   | smem_A2[BM][BK]  |
  | smem_B[BK][BN]   | smem_B2[BK][BN]  |
  +------------------+------------------+

  Iteration k:
    - Compute using buffer (k % 2)
    - Simultaneously prefetch tile (k+1) into buffer ((k+1) % 2)
    - Sync before switching

This overlaps memory latency with computation.
Shared memory usage doubles: 2 * (BM*BK + BK*BN) * 4 bytes.
For BM=BN=128, BK=8: 2 * (128*8 + 8*128) * 4 = 16 KB (fits easily).

Performance: ~3500-4500 GFLOPS (66-85% of peak)
```

---

## Performance Summary (expected on Quadro P4200)

```
Kernel                          GFLOPS    % of Peak (5.3 TFLOPS)
--------------------------------------------------------------
1. Naive                        200-400         4-8%
2. Coalesced                    400-600        8-11%
3. Shared Memory Tiling        1000-1500      19-28%
4. 1D Block Tiling             1500-2500      28-47%
5. 2D Block Tiling             2500-3500      47-66%
6. Vectorized Loads (float4)   3000-4000      57-75%
7. Double Buffering            3500-4500      66-85%
cuBLAS (reference)             4000-4800      75-90%
--------------------------------------------------------------
```

The gap between kernel 7 and cuBLAS comes from:
- Assembly-level scheduling (cuBLAS uses SASS, not PTX)
- Warp-level matrix operations (WMMA on newer architectures)
- Autotuning of tile sizes per specific GPU
- Software pipelining of register usage

---

## Register File Analysis

```
Quadro P4200 (CC 6.1):
  - 65536 registers per SM (256 KB)
  - Max 255 registers per thread
  - Max 2048 threads per SM

Kernel 5 (2D tiling) register budget per thread:
  threadResults[TM][TN] = 8 * 8 = 64 registers
  regA[TM]              = 8 registers
  regB[TN]              = 8 registers
  loop vars, pointers   = ~10 registers
  Total                 = ~90 registers

  With 90 regs/thread: 65536 / 90 = 728 threads max per SM
  That is 2-3 blocks of 256 threads -- enough to hide latency.

  If we used TM=TN=16: 256 + 16 + 16 + 10 = 298 > 255 max!
  So TM=TN=8 is the sweet spot for CC 6.1 (no register spilling).
```

---

## Files in This Chapter

| File | Description |
|------|-------------|
| `matmul_naive.cu` | Kernels 1-3: naive, coalesced, shared memory tiled |
| `matmul_optimized.cu` | Kernels 4-6: 1D tiling, 2D tiling, vectorized loads |
| `matmul_double_buffer.cu` | Kernel 7: double-buffered prefetching |
| `Makefile` | Build all three executables (links with cuBLAS) |

---

## Building and Running

```bash
make              # build all three executables
./matmul_naive    # runs kernels 1-3 + cuBLAS reference
./matmul_optimized  # runs kernels 4-6 + cuBLAS reference
./matmul_double_buffer  # runs kernel 7 + cuBLAS reference
```

Each executable benchmarks at M=N=K=2048 by default (change in code) and
reports GFLOPS plus correctness vs CPU reference.
