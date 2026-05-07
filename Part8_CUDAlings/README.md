# Part 8 — CUDAlings

> **Learn CUDA + Python bindings by doing.**
> A rustlings-style runner with **138 progressive exercises** across 30
> chapters, mirroring every chapter of this course (Parts 1-7) plus
> cross-cutting drills (patterns, perf, CNN) and capstone milestones.
> Edit one file, save, the runner tells you whether your kernel compiles,
> runs, and produces the right answer.

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                       HOW THE LOOP FEELS                                 │
 │                                                                          │
 │   $ ./cudalings watch                                                    │
 │                                                                          │
 │   • 01_hello_gpu/02_thread_id  [cuda]                                    │
 │     ⏸  not started -- remove `// I AM NOT DONE` to attempt.              │
 │                                                                          │
 │   ── you delete the marker, save the file ───────────────                │
 │                                                                          │
 │   • 01_hello_gpu/02_thread_id  [cuda]                                    │
 │     ✗ build failed                                                       │
 │       error: identifier "blockIdx" is undefined                          │
 │                                                                          │
 │   ── you fix it, save again ─────────────────────────                    │
 │                                                                          │
 │   • 01_hello_gpu/02_thread_id  [cuda]                                    │
 │     ✓ passed  (87 ms)                                                    │
 │                                                                          │
 │   → Next exercise: 01_hello_gpu/03_grid_strides                          │
 │                                                                          │
 └──────────────────────────────────────────────────────────────────────────┘
```

---

## Why exercises and not just reading?

Knowledge of CUDA decays fast unless you keep typing it. The course
chapters (Parts 1-7) teach the *concepts*; this part **drills them**.
Every exercise targets one well-defined skill:

- *"Index a 2D thread block correctly."*
- *"Avoid the bank conflict in this transpose."*
- *"Make this kernel actually launch when N is not a multiple of blockDim.x."*
- *"Wire a custom Conv2d backward into PyTorch's autograd."*

If you can solve the exercises without looking at hints, you've internalized
the chapter. If not, the runner shows you exactly what's wrong.

---

## Quick start

```bash
cd Part8_CUDAlings
./cudalings list                  # see every exercise and its status
./cudalings next                  # try the first unfinished one
./cudalings watch                 # auto-rerun when you save a file
./cudalings hint 01_hello_gpu/01_kernel_launch
./cudalings solution 01_hello_gpu/01_kernel_launch   # spoilers!
```

`watch` is the mode you'll spend the most time in: it polls for file changes
and re-runs the next failing exercise. Open the source file in your editor
on one side, the runner on the other, and you've got a tight feedback loop.

---

## How to actually work through this

A few things that matter more than they look:

- **Just run `./cudalings watch` from chapter 00 and let it pick the next
  exercise.** Don't plan a path -- the order is the path. Each exercise
  builds on muscle memory from the previous one.
- **Use `./cudalings hint <name>` before `solution`.** The hint points at
  the missing concept without giving you the code. Reaching for the
  solution first is how the muscle memory doesn't form. Hints exist for
  most early-chapter exercises; later chapters reuse the parent course
  chapter's `README.md` as their hint -- read that first.
- **Skip the `29_milestones` capstones until you've done the regular
  chapters of that Part.** Each milestone references 5+ earlier exercises
  and assumes the skills are already wired in. Doing them early just
  means you'll Google your way through.
- **Run `27_perf` once you've done a few optimization chapters.** Its ncu
  parsing and roofline drills look small but they're the difference
  between *"this kernel is slow"* and *"this kernel is at 78% of memory
  ceiling, I should optimize tile size."*
- **If a stub's stuck and you've already deleted `I AM NOT DONE`, run
  `./cudalings reset <name>`** to restore the pristine starting point
  from `solutions/<chapter>/<name>.stub.<ext>`.
- **When something breaks** (build errors on a solution, validators
  refusing valid output) -- the runner is meant to be hackable, not magic.
  Read `runner/cudalings.py` (~400 lines, well commented) and tweak.

Realistic time budget: 138 exercises × ~10-30 minutes each ≈ **40-60
hours** of focused practice. That's the order of magnitude for "I have
CUDA in muscle memory" -- comparable to rustlings (≈100 exercises) plus
a deep-dive topic.

---

## Anatomy of an exercise

```
exercises/01_hello_gpu/01_kernel_launch.cu      ← you edit this
exercises/01_hello_gpu/01_kernel_launch.expected.txt   ← validator spec
exercises/01_hello_gpu/01_kernel_launch.hint.md ← optional hint
solutions/01_hello_gpu/01_kernel_launch.cu      ← reference answer
solutions/01_hello_gpu/01_kernel_launch.stub.cu ← pristine starting point
```

Each source file starts with a top-of-file comment block describing the
problem, followed by a TODO marker:

```cuda
// CUDAlings 01.01 -- Launch your first kernel
//
// Goal: print "hello from thread 0..7" exactly once each, in any order.
//
// You should NOT need to add or remove top-level functions. Only fill the
// TODO block and modify the launch line marked TODO.

// I AM NOT DONE     <-- delete this line when you've attempted the exercise

#include <cstdio>

__global__ void hello() {
    // TODO: print "hello from thread <tid>" using printf and threadIdx.x
}

int main() {
    // TODO: launch `hello` with 1 block of 8 threads, then synchronize.
    return 0;
}
```

The runner skips files that still contain `I AM NOT DONE`. Once you delete
that line, it tries to build and validate.

---

## Validators

Each exercise has a sibling `*.expected.txt` describing how to grade your
answer. Modes:

| `mode:`            | What it does                                                                |
|--------------------|-----------------------------------------------------------------------------|
| `rc_zero`          | (default) Build succeeds, program returns 0                                 |
| `stdout_exact`     | Stdout matches the payload literally                                        |
| `stdout_contains`  | Every payload line appears somewhere in stdout                              |
| `stdout_regex`     | Stdout matches the payload as a regex (multiline + dotall)                  |
| `numeric`          | Last numeric token in stdout is within `tol:` of `target:`                  |
| `pytest`           | Run `pytest` against a sibling `test_<name>.py` file                        |

Example for a reduction kernel:

```
mode: numeric
target: 1024.0
tol: 1e-4
```

Example for a hello-world kernel:

```
mode: stdout_contains
---
hello from thread 0
hello from thread 1
hello from thread 7
```

---

## Course-aligned roadmap

Every CUDAlings chapter mirrors the corresponding course chapter:

| CUDAlings dir          | Mirrors course chapter                          | What you'll drill                       |
|------------------------|-------------------------------------------------|-----------------------------------------|
| `00_warmup`            | (pre-chapter)                                   | Host C, pointers, malloc-only, simplest kernel |
| `01_hello_gpu`         | Part 1 / Ch 01                                  | Kernel launch, threadIdx, vector add    |
| `02_memory_model`      | Part 1 / Ch 02                                  | Global/shared/constant, host↔device     |
| `03_thread_hierarchy`  | Part 1 / Ch 03                                  | 1D/2D/3D indexing, grid strides         |
| `04_error_handling`    | Part 1 / Ch 04                                  | cudaGetLastError, profiling             |
| `05_coalescing`        | Part 2 / Ch 05                                  | AoS→SoA, stride patterns                |
| `06_shared_memory`     | Part 2 / Ch 06                                  | Tiling, bank conflicts, __syncthreads   |
| `07_occupancy`         | Part 2 / Ch 07                                  | Launch tuning, register pressure        |
| `08_streams`           | Part 2 / Ch 08                                  | Async copy/compute overlap              |
| `09_warp_prims`        | Part 2 / Ch 09                                  | Shuffle, ballot, vote                   |
| `10_reduction`         | Part 3 / Ch 10                                  | Tree reduction, warp unrolling          |
| `11_scan`              | Part 3 / Ch 11                                  | Hillis-Steele, Blelloch                 |
| `12_matmul`            | Part 3 / Ch 12                                  | Naive → tiled → register-blocked        |
| `13_tensor`            | Part 4 / Ch 13                                  | RAII Tensor, strides, broadcast         |
| `14_forward_ops`       | Part 4 / Ch 14                                  | ReLU, Linear, BatchNorm, Conv2D forward |
| `15_backward_ops`      | Part 4 / Ch 15                                  | Backward kernels                        |
| `16_autograd`          | Part 4 / Ch 16                                  | Computational graph, topological sort   |
| `17_library`           | Part 5 / Ch 17                                  | Module/Parameter/Optimizer plumbing     |
| `18_python_bindings`   | Part 5 / Ch 18                                  | pybind11, PyTorch C++ extensions        |
| `19_resnet`            | Part 6 / Ch 19                                  | ResBlock, skip, GAP                     |
| `20_training`          | Part 6 / Ch 20                                  | End-to-end training loop                |
| `21_nanogpt_py`        | Part 7 / Ch 21                                  | Attention, transformer block in PyTorch |
| `22_tokenizer`         | Part 7 / Ch 22                                  | BPE, dataloader, sampling               |
| `23_llm_c_fwd`         | Part 7 / Ch 23                                  | LLM forward in pure CUDA                |
| `24_llm_c_bwd`         | Part 7 / Ch 24                                  | LLM backward                            |
| `25_llm_c_train`       | Part 7 / Ch 25                                  | AdamW, training loop in CUDA            |
| `26_patterns`          | (cross-cutting parallel patterns)               | Histogram, stencil, conv1d, partition, segmented reduce, radix |
| `27_perf`              | (cross-cutting performance drills)              | Bandwidth measurement, arithmetic intensity, warp divergence, memcpy bench |
| `28_cnn`               | (CNN-focused drills, builds on Part 4 + 6)      | im2col kernel, conv with padding/stride, batchnorm2d, full TinyCNN forward, conv2d backward |
| `29_milestones`        | (capstone per Part 1-7)                         | SAXPY bench, async copy, max-scan, MLP train, torch extension, ResNet train, GPT train |

Each chapter has 3-7 exercises that progress from "make it compile" to
"make it match cuBLAS within 0.1%". Hint files are deliberately sparse --
the README of the parent course chapter is your real reference.

---

## Authoring your own exercises

Want to add one? Three files:

1. `exercises/<chapter>/<NN>_<name>.<ext>` — the stub the student edits.
2. `exercises/<chapter>/<NN>_<name>.expected.txt` — validator spec.
3. `solutions/<chapter>/<NN>_<name>.<ext>` — your reference answer.
4. *(optional)* `solutions/<chapter>/<NN>_<name>.stub.<ext>` — frozen copy
   of the original stub so `./cudalings reset <name>` works.
5. *(optional)* `exercises/<chapter>/<NN>_<name>.hint.md` — a hint.

Run `./cudalings list` to confirm it's discovered.

---

## Architecture override

The runner targets `sm_61` (Pascal / Quadro P4200) by default to match the
rest of the course. If you're on newer hardware:

```bash
CUDALINGS_ARCH=sm_86 ./cudalings watch       # Ampere
CUDALINGS_ARCH=sm_75 ./cudalings watch       # Turing
```

Pure ANSI exercises will compile anywhere. Some optimization exercises in
chapter 09 and beyond will note when they require a specific arch (e.g.
`__shfl_sync` is sm_30+, `wmma` is sm_70+, `cp.async` is sm_80+).
