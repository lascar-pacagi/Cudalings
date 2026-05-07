# Chapter 08: CUDA Streams and Asynchronous Operations

## What Is a CUDA Stream?

A CUDA **stream** is an ordered sequence of operations (kernel launches, memory
copies, event records) that execute **in order** with respect to each other.
Operations in *different* streams, however, may execute **concurrently** -- and
this is the entire point: by organizing work into independent streams, we unlock
the GPU's ability to overlap computation with data transfers.

```
    Stream = ordered queue of GPU operations
    ═══════════════════════════════════════════════════════════════

    Stream A:   op1 ──→ op2 ──→ op3        (op1 finishes before op2 starts)

    Stream B:   op4 ──→ op5 ──→ op6        (op4 finishes before op5 starts)

    BUT: op1 and op4 may run at the SAME TIME (different streams!)
         op2 and op5 may run at the SAME TIME
         etc.

    The GPU hardware decides what can actually overlap based on
    available resources (SMs, copy engines, etc.)
```

Think of streams as **lanes on a highway**. Cars in a single lane must follow
one another, but cars in different lanes travel in parallel.

---

## The Default Stream (Stream 0)

Every CUDA call that does not specify a stream goes into the **default stream**
(also called stream 0 or the NULL stream). This stream has special blocking
behavior: operations in the default stream cannot overlap with operations in
any other stream (unless you compile with `--default-stream per-thread`).

This means that a naive CUDA program serializes everything:

```
    Default Stream (stream 0) — everything is serial
    ═══════════════════════════════════════════════════════════════

    Time ──────────────────────────────────────────────────────→

    ┌──────────┐ ┌─────────────────┐ ┌──────────┐
    │  H2D     │ │    Kernel       │ │   D2H    │
    │  copy    │ │    execution    │ │   copy   │
    └──────────┘ └─────────────────┘ └──────────┘
    ← 2 ms →     ←    5 ms        → ← 2 ms   →

    Total: 2 + 5 + 2 = 9 ms   (no overlap at all!)

    The GPU's copy engine sits idle during the kernel.
    The SMs sit idle during the copies.
    Terrible utilization!
```

The solution: use **multiple streams** so the GPU can overlap transfers with
computation.

---

## Overlapped Execution with Multiple Streams

When we split our data into chunks and assign each chunk to its own stream,
the GPU can overlap Host-to-Device transfers, kernel execution, and
Device-to-Host transfers:

```
    Multiple Streams — overlapped execution
    ═══════════════════════════════════════════════════════════════

    Time ──────────────────────────────────────────────────────→

    Stream 0:  ┌─H2D─┐ ┌──Kernel──┐ ┌─D2H─┐
               │  A0  │ │    A0    │ │  A0  │
               └──────┘ └──────────┘ └──────┘

    Stream 1:           ┌─H2D─┐ ┌──Kernel──┐ ┌─D2H─┐
                        │  A1  │ │    A1    │ │  A1  │
                        └──────┘ └──────────┘ └──────┘

    Stream 2:                    ┌─H2D─┐ ┌──Kernel──┐ ┌─D2H─┐
                                │  A2  │ │    A2    │ │  A2  │
                                └──────┘ └──────────┘ └──────┘

    Stream 3:                             ┌─H2D─┐ ┌──Kernel──┐ ┌─D2H─┐
                                          │  A3  │ │    A3    │ │  A3  │
                                          └──────┘ └──────────┘ └──────┘

    ═══════════════════════════════════════════════════════════════
    KEY INSIGHT: While Stream 0's kernel runs, Stream 1's H2D transfer
    happens simultaneously! While Stream 1's kernel runs, Stream 0's
    D2H and Stream 2's H2D overlap with it!

    The copy engine and the SMs work in parallel.
    Total time ≈ time(all kernels) + time(one H2D) + time(one D2H)
    Instead of: N * (H2D + kernel + D2H)
```

**Important caveat for the Quadro P4200:** This GPU has only **1 copy engine**,
meaning it can perform only one DMA transfer at a time. An H2D and a D2H
**cannot** overlap with each other — only transfers and compute can overlap.
GPUs with 2 copy engines (like many Tesla/A100 cards) can overlap H2D and D2H
simultaneously with compute (triple overlap).

```
    1 Copy Engine (P4200):  H2D and D2H CANNOT overlap
    ═══════════════════════════════════════════════════════════════

    Copy Engine:  ┌H2D-0┐┌H2D-1┐┌D2H-0┐┌H2D-2┐┌D2H-1┐┌D2H-2┐
    SMs:                 ┌Kern0─┐┌Kern1─┐┌Kern2─┐
                  ↑       ↑
                  │       └─ H2D-1 overlaps Kern0 (good!)
                  └─ nothing to overlap yet

    2 Copy Engines (Tesla/A100): H2D and D2H CAN overlap
    ═══════════════════════════════════════════════════════════════

    Copy Eng 1 (H2D):  ┌H2D-0┐┌H2D-1┐┌H2D-2┐
    Copy Eng 2 (D2H):               ┌D2H-0┐┌D2H-1┐┌D2H-2┐
    SMs:                      ┌Kern0─┐┌Kern1─┐┌Kern2─┐
```

---

## Pinned (Page-Locked) Memory

Asynchronous memory copies (`cudaMemcpyAsync`) **require pinned (page-locked)
memory** on the host side. Without it, the copy falls back to synchronous
behavior even if you call the Async variant.

### Why Pinned Memory?

Normal (pageable) memory can be swapped to disk by the OS at any time. The GPU
cannot safely DMA from pageable memory because the physical address might change.
So the CUDA runtime must first copy pageable data to an internal pinned staging
buffer, then DMA from there — this is an extra copy and forces synchronization.

With pinned memory, the GPU can DMA directly to/from the host buffer:

```
    Pageable Memory (default malloc/new)
    ═══════════════════════════════════════════════════════════════

      CPU (Host)                              GPU (Device)
    ┌──────────────────┐                   ┌────────────────┐
    │                  │     CANNOT DMA     │                │
    │  Pageable Buffer │──── directly ────X │  Device Mem    │
    │  (can be swapped │                   │                │
    │   to disk!)      │                   │                │
    └────────┬─────────┘                   └────────────────┘
             │  extra copy                        ↑
             ▼  (CPU does this)                   │
    ┌──────────────────┐          DMA             │
    │  Internal Pinned │──────────────────────────┘
    │  Staging Buffer  │   (now safe, address
    │  (CUDA runtime   │    won't change)
    │   allocates this) │
    └──────────────────┘

    → Two copies: pageable → pinned staging → device
    → The first copy blocks the CPU
    → cudaMemcpyAsync acts synchronously (bad!)


    Pinned Memory (cudaMallocHost / cudaHostAlloc)
    ═══════════════════════════════════════════════════════════════

      CPU (Host)                              GPU (Device)
    ┌──────────────────┐                   ┌────────────────┐
    │                  │    DIRECT DMA      │                │
    │  Pinned Buffer   │──────────────────→ │  Device Mem    │
    │  (page-locked,   │   (safe! address   │                │
    │   never swapped) │    is stable)      │                │
    └──────────────────┘                   └────────────────┘

    → One copy: pinned → device (via DMA)
    → CPU is free to do other work
    → cudaMemcpyAsync is truly asynchronous!
```

### Allocation Functions

```cpp
// Method 1: cudaMallocHost — simplest
float *h_pinned;
cudaMallocHost(&h_pinned, size);   // allocate pinned memory
// ... use it ...
cudaFreeHost(h_pinned);            // free pinned memory

// Method 2: cudaHostAlloc — more options
float *h_pinned2;
cudaHostAlloc(&h_pinned2, size, cudaHostAllocDefault);
// Flags: cudaHostAllocDefault    — same as cudaMallocHost
//        cudaHostAllocPortable   — pinned across all CUDA contexts
//        cudaHostAllocMapped     — also mapped into device address space
//        cudaHostAllocWriteCombined — WC memory, fast for CPU writes
cudaFreeHost(h_pinned2);
```

**Warning:** Pinned memory is a limited resource. Allocating too much can
degrade system performance because it reduces the OS's ability to manage
virtual memory. Use it for transfer buffers, not for all host allocations.

---

## CUDA Events

CUDA events are markers that can be inserted into streams. They serve two
main purposes:

1. **Timing**: measure elapsed GPU time between two events
2. **Synchronization**: create dependencies between streams

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);   // record timestamp in stream
myKernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);    // record timestamp in stream

cudaEventSynchronize(stop);       // wait for stop event to complete

float ms;
cudaEventElapsedTime(&ms, start, stop);  // compute elapsed time
printf("Kernel took %.3f ms\n", ms);
```

### Inter-Stream Synchronization with Events

`cudaStreamWaitEvent(stream, event)` makes `stream` wait until `event` has
been recorded (completed) in whatever stream it was recorded in:

```
    Inter-Stream Dependency with cudaStreamWaitEvent
    ═══════════════════════════════════════════════════════════════

    Stream A:  ┌──Kernel A──┐  ●event_done
                                     │
    Stream B:                        ▼ cudaStreamWaitEvent(B, event_done)
                                     ┌──Kernel B──┐
                                     │ waits until │
                                     │ event_done  │
                                     └─────────────┘

    Kernel B will not start until Kernel A finishes and event_done
    is recorded. This creates a cross-stream dependency.
```

---

## Overlap Strategies

### Strategy 1: Breadth-First Issue Order

Issue all H2D copies first, then all kernels, then all D2H copies. This works
but can delay the start of overlap:

```
    Breadth-First (less ideal on 1 copy engine):
    ═══════════════════════════════════════════════════════════════
    Issue order: H2D(0), H2D(1), H2D(2), Kern(0), Kern(1), Kern(2), D2H(0)...

    Copy Engine: ┌H2D0┐┌H2D1┐┌H2D2┐        ┌D2H0┐┌D2H1┐┌D2H2┐
    SMs:                            ┌K0┐┌K1┐┌K2┐
                 ← all H2D first → ← all kernels → ← all D2H →

    Not much overlap possible!
```

### Strategy 2: Depth-First Issue Order (Preferred)

Issue all operations for stream 0, then all for stream 1, etc. The GPU
hardware reorders to maximize overlap:

```
    Depth-First (preferred):
    ═══════════════════════════════════════════════════════════════
    Issue order: H2D(0), K(0), D2H(0), H2D(1), K(1), D2H(1), ...

    Copy Engine: ┌H2D0┐┌H2D1┐┌D2H0┐┌H2D2┐┌D2H1┐┌D2H2┐
    SMs:               ┌──K0──┐┌──K1──┐┌──K2──┐

    The GPU sees that H2D(1) is independent of K(0), so it overlaps them!
    Much better pipeline utilization.
```

---

## Synchronization Levels

```
    Synchronization Hierarchy
    ═══════════════════════════════════════════════════════════════

    Most specific (least blocking)
    ──────────────────────────────
    cudaEventQuery(event)          ← non-blocking check
    cudaStreamQuery(stream)        ← non-blocking check
    cudaEventSynchronize(event)    ← block until one event completes
    cudaStreamSynchronize(stream)  ← block until all ops in stream done
    cudaDeviceSynchronize()        ← block until ALL streams done
    ──────────────────────────────
    Most general (most blocking)

    Rule: always use the LEAST blocking synchronization that is correct.
    cudaDeviceSynchronize() is a sledgehammer — avoid it in production.
```

---

## Best Practices

1. **Always use pinned memory** for async transfers. Without it, `cudaMemcpyAsync`
   is secretly synchronous and you get zero overlap.

2. **Use depth-first issue order.** Issue H2D, kernel, D2H for stream 0, then
   for stream 1, etc. The GPU hardware will find the overlap opportunities.

3. **Use enough streams** to fill the pipeline (typically 4-8 streams suffice).
   Too many streams add overhead without benefit.

4. **Be aware of your copy engine count.** The P4200 has 1 copy engine, so H2D
   and D2H cannot overlap with each other — only with compute.

5. **Profile with `nsys` or `nvprof`** to visualize the actual timeline.
   `nsys profile ./my_app` then `nsys-ui` gives a beautiful timeline view.

6. **Do not over-allocate pinned memory.** It is taken from the OS's
   non-pageable pool. Too much pinned memory starves the OS.

7. **Avoid the default stream** in performance-critical code. Always create
   explicit streams.

8. **Destroy streams and events** when done to avoid resource leaks.

---

## Files in This Chapter

| File                  | Purpose                                                   |
|-----------------------|-----------------------------------------------------------|
| `streams_basics.cu`   | Serial vs multi-stream execution, timing comparison       |
| `overlap_transfer.cu` | Overlap of data transfer and compute with pinned memory   |
| `event_sync.cu`       | CUDA events for timing and inter-stream synchronization   |
| `Makefile`            | Build all examples for SM 6.1 with g++-11                 |
