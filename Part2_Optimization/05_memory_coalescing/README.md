# Chapter 05: Memory Coalescing

## The Single Most Important GPU Optimization

If you learn only one thing about GPU performance, let it be this:
**how your threads access memory determines everything.**

A GPU can move hundreds of gigabytes per second -- but only if threads
cooperate to access memory in the right pattern. When they do, we call
it **coalesced access**. When they don't, performance can drop by 10-30x.

Your Quadro P4200 has a theoretical peak bandwidth of ~192 GB/s.
With coalesced access, you can achieve ~160-175 GB/s.
With the worst access patterns, you might see only 5-15 GB/s.

---

## How the Memory Controller Works

When a warp (32 threads) executes a load or store instruction, the hardware
does NOT issue 32 separate memory requests. Instead:

1. All 32 threads present the addresses they want to access.
2. The memory controller **groups** these addresses into **cache line transactions**.
3. Each transaction fetches a full cache line (32, 64, or 128 bytes on L1/L2).
4. Data is distributed from the fetched cache lines to the requesting threads.

The key insight: **fewer transactions = higher bandwidth**.

```
  HOW A MEMORY REQUEST WORKS
  ===========================

  32 threads in a warp each want one float (4 bytes):

  Step 1: Threads present addresses
  ---------------------------------
  T0:  addr 0x1000
  T1:  addr 0x1004
  T2:  addr 0x1008
  ...
  T31: addr 0x107C

  Step 2: Memory controller groups into transactions
  ---------------------------------------------------
  All 32 addresses fall within one 128-byte cache line!
  [0x1000 - 0x107F] = ONE transaction

  Step 3: Data delivered to threads
  ----------------------------------
  128 bytes fetched, 128 bytes used = 100% efficiency
```

---

## Coalesced Access (Best Case)

When consecutive threads access consecutive memory addresses, all addresses
fall within one (or very few) cache lines. This is **coalesced access**.

```
  COALESCED ACCESS: 32 threads reading consecutive floats
  =======================================================

  Thread:  T0   T1   T2   T3   T4   T5   ... T30  T31
           |    |    |    |    |    |    |     |    |
           v    v    v    v    v    v    v     v    v
  Memory: [f0 | f1 | f2 | f3 | f4 | f5 |...| f30| f31]
          |<------------- 128 bytes (one cache line) ----------->|

  Result: 1 transaction to fetch 128 bytes
          128 bytes requested / 128 bytes transferred = 100% efficiency
          Achieved bandwidth: ~170 GB/s on your P4200
```

Code pattern:
```c
// GOOD: thread i accesses element i (consecutive)
int i = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[i];   // Coalesced!
```

---

## Strided Access (Partial Waste)

When threads access every Nth element, the addresses spread across
multiple cache lines. Stride of 2 wastes 50% of fetched bytes.

```
  STRIDED ACCESS (stride = 2): 32 threads, each skipping one float
  =================================================================

  Thread:  T0        T1        T2        T3        ...
           |         |         |         |
           v         v         v         v
  Memory: [f0 | -- | f2 | -- | f4 | -- | f6 | -- | ...]
          |<---- cache line 1 ---->|<---- cache line 2 ---->|  ...

  T0-T15 addresses span 256 bytes -> need 2 cache line transactions
  T16-T31 addresses span another 256 bytes -> need 2 more transactions

  Result: 4 transactions to fetch 512 bytes, but only 128 bytes used
          128 bytes requested / 512 bytes transferred = 25% efficiency

  STRIDED ACCESS (stride = 32): worst case for a warp
  =====================================================

  Thread:  T0             T1              T2              ...
           |              |               |
           v              v               v
  Memory: [f0 |...128B...| f32 |...128B...| f64 |...128B...|...]

  Each thread's address is in a DIFFERENT cache line!

  Result: 32 transactions to fetch 32 x 128 = 4096 bytes
          128 bytes requested / 4096 bytes transferred = 3.1% efficiency
          Achieved bandwidth: ~5-8 GB/s (30x slower!)
```

Code pattern:
```c
// BAD: thread i accesses element i*stride (non-consecutive)
int i = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[i * stride];   // Strided -- wastes bandwidth!
```

---

## Random Access (Worst Case)

When each thread accesses a random address, every access likely hits
a different cache line. This is the absolute worst case.

```
  RANDOM ACCESS: 32 threads reading random locations
  ====================================================

  Thread:  T0      T1      T2      T3      T4      ...
           |       |       |       |       |
           v       v       v       v       v
  Memory: [................|...............|......]
          ^       ^    ^           ^   ^
          f[917]  f[3] f[42081]   f[7] f[55002]

  Each address is in a completely different cache line.

  Result: up to 32 transactions, each fetching 128 bytes
          128 bytes requested / up to 4096 bytes transferred
          Efficiency: as low as 3%
          Achieved bandwidth: ~5-10 GB/s
```

---

## Cache Line Sizes and Transaction Granularity

On your Quadro P4200 (CC 6.1, Pascal architecture):

```
  CACHE HIERARCHY
  ================

  GPU Cores (SMs)
       |
       v
  L1 Cache (per SM): 48 KB, cache line = 128 bytes
       |              (configurable: 16KB/48KB L1 vs shared memory)
       v
  L2 Cache (shared): ~2 MB, cache line = 32 bytes (sector)
       |              L2 operates in 32-byte sectors within 128-byte lines
       v
  GDDR5 DRAM:        8 GB, 256-bit bus
```

Key details:
- L1 cache line: **128 bytes** (32 floats, or the exact output of one coalesced warp read)
- L2 sector: **32 bytes** (8 floats) -- the L2 can fetch partial cache lines
- A warp reading 32 consecutive floats = 128 bytes = exactly 1 L1 cache line
- Misaligned access may straddle 2 cache lines, fetching 256 bytes for 128 needed

---

## AoS vs SoA: The Layout That Changes Everything

### Array of Structures (AoS)

```c
struct Particle {
    float x, y, z;           // position
    float vx, vy, vz;        // velocity
    float mass;              // mass
};  // 7 floats = 28 bytes per particle

Particle particles[N];      // AoS layout
```

Memory layout (AoS):
```
  ARRAY OF STRUCTURES (AoS) -- Bad for GPU
  ==========================================

  Memory address -->
  [x0|y0|z0|vx0|vy0|vz0|m0|x1|y1|z1|vx1|vy1|vz1|m1|x2|y2|z2|vx2|...]

  When a warp reads ALL the x-coordinates:
  Thread:  T0              T1              T2
           |               |               |
           v               v               v
          [x0|y0|z0|vx|vy|vz|m|x1|y0|z1|vx|vy|vz|m|x2|...]
           ^-- stride of 7 floats (28 bytes) between x values!

  This is strided access with stride = 7!
  Each thread's x value is 28 bytes apart.
  A warp needs 32 x values = 32 addresses spread over 32*28 = 896 bytes
  That spans 7 cache lines -> 7 transactions, but only 128 bytes used
  Efficiency: 128 / (7 * 128) = 14%
```

### Structure of Arrays (SoA)

```c
struct ParticlesSoA {
    float x[N], y[N], z[N];       // positions
    float vx[N], vy[N], vz[N];    // velocities
    float mass[N];                 // masses
};

ParticlesSoA particles;            // SoA layout
```

Memory layout (SoA):
```
  STRUCTURE OF ARRAYS (SoA) -- Good for GPU
  ===========================================

  Memory address -->
  x array:    [x0|x1|x2|x3|x4|x5|x6|x7|...|x31|x32|...]
  y array:    [y0|y1|y2|y3|y4|y5|y6|y7|...|y31|y32|...]
  z array:    [z0|z1|z2|z3|z4|z5|z6|z7|...|z31|z32|...]
  vx array:   [vx0|vx1|vx2|vx3|...]
  vy array:   [vy0|vy1|vy2|vy3|...]
  vz array:   [vz0|vz1|vz2|vz3|...]
  mass array: [m0|m1|m2|m3|...]

  When a warp reads ALL the x-coordinates:
  Thread:  T0  T1  T2  T3  ... T31
           |   |   |   |       |
           v   v   v   v       v
          [x0| x1| x2| x3|...| x31]
          |<-- 128 bytes, ONE cache line -->|

  Perfectly coalesced! 1 transaction, 100% efficiency.
```

### Why SoA Wins on GPUs

| Aspect | AoS | SoA |
|--------|-----|-----|
| Access pattern | Strided (stride = struct size) | Coalesced (consecutive) |
| Efficiency per field read | ~14% (for 7-field struct) | 100% |
| Memory transactions | 7x more | Minimal |
| Practical speedup | Baseline | 2-5x faster |

### Connection to Deep Learning: NCHW vs NHWC

The AoS vs SoA choice appears in neural networks as tensor layout:

```
  Image tensor with N=batch, C=channels, H=height, W=width

  NCHW (SoA-like): all red pixels, then all green, then all blue
  [R00 R01 R02 ... R_HW | G00 G01 ... G_HW | B00 B01 ... B_HW]
   |<-- one channel -->|  |<-- one channel -->|

  NHWC (AoS-like): each pixel has all channels together
  [R00 G00 B00 | R01 G01 B01 | R02 G02 B02 | ...]
   |<-pixel 0->| |<-pixel 1->|

  For convolution: NCHW often has better coalescing for reading input
  For modern tensor cores: NHWC is preferred (hardware-specific)
```

---

## Coalescing Rules Summary

| Access Pattern | Transactions (32 threads, float) | Efficiency | Bandwidth |
|---------------|----------------------------------|------------|-----------|
| Consecutive (stride 1) | 1 x 128B | 100% | ~170 GB/s |
| Stride 2 | 2-4 x 128B | 25-50% | ~50-85 GB/s |
| Stride 4 | 4-8 x 128B | 12-25% | ~25-40 GB/s |
| Stride 32 | up to 32 x 128B | ~3% | ~5-8 GB/s |
| Random | up to 32 x 128B | ~3% | ~5-10 GB/s |
| Misaligned by 1 | 2 x 128B | 50% | ~85 GB/s |
| Broadcast (all same addr) | 1 x 128B | 3% (but cached) | varies |

---

## Quantifying the Impact

**Effective bandwidth** = bytes_your_kernel_needs / time

**Theoretical bandwidth** = memory_clock x bus_width x 2 (DDR) / 8

For your Quadro P4200:
- Theoretical: ~192 GB/s
- Practical peak (coalesced): ~160-175 GB/s
- Strided by 32 or random: ~5-10 GB/s

That is a **20-30x difference** from the same hardware, just by changing
how threads map to memory addresses.

---

## Files in This Chapter

| File | Description |
|------|-------------|
| `coalescing_patterns.cu` | Benchmark 5 access patterns, see bandwidth impact |
| `aos_vs_soa.cu` | Practical particle simulation: AoS vs SoA comparison |
| `matrix_transpose.cu` | Classic case study: why naive transpose is slow |
| `Makefile` | Build all programs |

## Build and Run

```bash
make          # Build all
make run      # Build and run all programs
make clean    # Remove binaries
```

---

## Key Takeaways

1. **Coalescing = consecutive threads access consecutive addresses**
2. **Strided access wastes bandwidth proportional to the stride**
3. **Use SoA layout instead of AoS for GPU data structures**
4. **Measure effective bandwidth to quantify your kernel's efficiency**
5. **Memory coalescing is the #1 optimization -- do this before anything else**
