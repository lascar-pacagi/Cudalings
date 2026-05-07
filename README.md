# CUDA Deep Learning from Scratch
## A Progressive Course: From First Kernel to ChatGPT-style Assistant

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                    COURSE ARCHITECTURE OVERVIEW                            │
 │                                                                            │
 │  Part 1: Foundations          Part 2: Optimization        Part 3: Algos    │
 │  ┌──────────────────┐        ┌──────────────────┐        ┌──────────────┐  │
 │  │ 01 Hello GPU     │        │ 05 Coalescing    │        │ 10 Reduction │  │
 │  │ 02 Memory Model  │───────►│ 06 Shared Mem    │───────►│ 11 Scan      │  │
 │  │ 03 Threads       │        │ 07 Occupancy     │        │ 12 MatMul    │  │
 │  │ 04 Profiling     │        │ 08 Streams       │        └──────┬───────┘  │
 │  └──────────────────┘        │ 09 Warp Prims    │               │          │
 │                              └──────────────────┘               │          │
 │                                                                 ▼          │
 │  Part 6: ResNet               Part 5: DL Library     Part 4: DL Prims     │
 │  ┌──────────────────┐        ┌──────────────────┐    ┌──────────────────┐  │
 │  │ 19 ResNet Arch   │◄───────│ 17 C++ Backend   │◄───│ 13 Tensor Class  │  │
 │  │ 20 Train+Profile │        │ 18 Python Front  │    │ 14 Forward Ops   │  │
 │  └─────┬────────────┘        └──────────────────┘    │ 15 Backward Ops  │  │
 │        │                                             │ 16 Comp Graph    │  │
 │        ▼                                             └──────────────────┘  │
 │  Part 7: GPT From Scratch (NEW)                                            │
 │  ┌──────────────────────────────────────────────────────────────────────┐  │
 │  │ 21 nanoGPT (PyTorch)  →  22 BPE + Training                           │  │
 │  │            ▼                                                         │  │
 │  │ 23 llm.c forward in CUDA  →  24 backward  →  25 full training        │  │
 │  └──────────────────────────────────────────────────────────────────────┘  │
 │                                                                            │
 │  Part 8: CUDAlings (NEW) -- ~100 progressive exercises across all parts    │
 │  ┌──────────────────────────────────────────────────────────────────────┐  │
 │  │  ./cudalings watch  → edit, save, see ✓ or ✗ instantly                │  │
 │  └──────────────────────────────────────────────────────────────────────┘  │
 │                                                                            │
 └─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- C/C++ fundamentals (pointers, structs, templates)
- Basic linear algebra (matrix multiply, dot product)
- Some neural network intuition (what a layer does, what backprop is)
- A machine with an NVIDIA GPU and CUDA toolkit installed

## Your Setup

- **GPU**: Quadro P4200 (Compute Capability 6.1, Pascal, 18 SMs)
- **CUDA Toolkit**: 11.7
- **Driver**: 535.x (supports up to CUDA 12.2)

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                     YOUR GPU: Quadro P4200                      │
 │                                                                 │
 │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐             │
 │  │ SM0 │ │ SM1 │ │ SM2 │ │ SM3 │ │ SM4 │ │ SM5 │    18 SMs    │
 │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘    total     │
 │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐             │
 │  │ SM6 │ │ SM7 │ │ SM8 │ │ SM9 │ │SM10 │ │SM11 │  128 CUDA    │
 │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘  cores/SM    │
 │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐             │
 │  │SM12 │ │SM13 │ │SM14 │ │SM15 │ │SM16 │ │SM17 │  = 2304      │
 │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘  cores       │
 │                                                                 │
 │  L2 Cache: 2 MB          Memory: 8 GB GDDR5                    │
 │  Memory Bandwidth: ~192 GB/s    Compute: ~5.3 TFLOPS FP32     │
 └─────────────────────────────────────────────────────────────────┘
```

## How to Use This Course

Each chapter is a self-contained directory with:
- `README.md` — theory, diagrams, explanations
- `.cu` / `.cpp` / `.py` — **heavily commented** source files you compile and run
- `Makefile` — just type `make` then `./program_name`

**Compile any example:**
```bash
cd Part1_Foundations/01_hello_gpu
make
./hello_gpu
```

**Profile any example (chapters 04+):**
```bash
nvprof ./my_program           # Quick profiling
nsys profile ./my_program     # Timeline profiling (if nsight-systems installed)
ncu ./my_program              # Kernel-level profiling (if nsight-compute installed)
```

## Course Roadmap

### Part 1: CUDA Foundations
| Ch | Topic | Key Concept | You Build |
|----|-------|-------------|-----------|
| 01 | Hello GPU | Kernels, threads, blocks | Vector add |
| 02 | Memory Model | Global, shared, constant, registers | Stencil computation |
| 03 | Thread Hierarchy | Grids, blocks, warps, indexing | 2D image processing |
| 04 | Error Handling & Profiling | cudaGetLastError, nvprof, nsight | Benchmarking harness |

### Part 2: Optimization
| Ch | Topic | Key Concept | You Build |
|----|-------|-------------|-----------|
| 05 | Memory Coalescing | Strided vs. coalesced access patterns | SoA vs AoS comparison |
| 06 | Shared Memory & Tiling | Bank conflicts, tiling strategy | Tiled matrix transpose |
| 07 | Occupancy & Launch Config | Registers, shared mem, warps | Auto-tuning launcher |
| 08 | Streams & Async | Overlap compute/transfer, events | Pipeline with streams |
| 09 | Warp-Level Primitives | Shuffle, ballot, vote | Warp-level reduction |

### Part 3: GPU Algorithms
| Ch | Topic | Key Concept | You Build |
|----|-------|-------------|-----------|
| 10 | Parallel Reduction | Tree reduction, warp unrolling | Multi-kernel sum/max |
| 11 | Scan (Prefix Sum) | Blelloch, Hillis-Steele | Inclusive/exclusive scan |
| 12 | Matrix Multiplication | Naive → tiled → register-blocked | Near-cuBLAS matmul |

### Part 4: Deep Learning Primitives
| Ch | Topic | Key Concept | You Build |
|----|-------|-------------|-----------|
| 13 | Tensor Class | RAII, device memory, ref counting | `Tensor<float>` in C++ |
| 14 | Forward Ops | Conv2D, ReLU, BatchNorm, Linear, GAP | All forward CUDA kernels |
| 15 | Backward Ops | Chain rule on GPU, gradient kernels | All backward passes |
| 16 | Computational Graph | DAG, topological sort, autograd | `autograd::Engine` |

### Part 5: Deep Learning Library
| Ch | Topic | Key Concept | You Build |
|----|-------|-------------|-----------|
| 17 | C++ Library Architecture | Module, Parameter, Optimizer | `cudalearn` C++ lib |
| 18 | Python Frontend (pybind11) | Bindings, pythonic API | `import cudalearn` |

### Part 6: ResNet from Scratch
| Ch | Topic | Key Concept | You Build |
|----|-------|-------------|-----------|
| 19 | ResNet Architecture | ResBlock, skip connections, GAP | Full ResNet in cudalearn |
| 20 | Training & Profiling | End-to-end training, optimization | Train on MNIST + profile |

### Part 7: ChatGPT-style Assistant from Scratch
| Ch | Topic | Key Concept | You Build |
|----|-------|-------------|-----------|
| 21 | nanoGPT in PyTorch | Causal attention, MLP, transformer block | Karpathy-style GPT |
| 22 | Tokenizer + Data + Training | BPE, streaming dataloader, AdamW + cosine LR | Chat REPL on Tiny Shakespeare |
| 23 | llm.c Forward in CUDA | Encoder, layernorm, attention, GELU kernels | Pure-CUDA forward pass |
| 24 | llm.c Backward in CUDA | Per-op gradients, atomic accumulation | Hand-written backward kernels |
| 25 | Full Training in CUDA | AdamW kernel, training loop | GPT trained without PyTorch |

### Part 8: CUDAlings — Progressive Exercises
| What | How |
|------|-----|
| **138 exercises across 30 chapters** | Edit a stub, save, runner shows ✓ or ✗ in <1s |
| Validation modes | `stdout_exact`, `stdout_contains`, `stdout_regex`, `numeric`, `pytest` |
| Watch loop | `./cudalings watch` — auto-rebuild on save |
| Reference solutions | `./cudalings solution <name>` (use sparingly) |
| Hints | `./cudalings hint <name>` (use these *before* the solution) |
| Reset a stub | `./cudalings reset <name>` (restore the pristine starting point) |

The chapters mirror Parts 1-7 one-to-one and add four cross-cutting groups:
**00_warmup** (pre-chapter basics), **26_patterns** (histogram, stencil,
1D conv, partition, segmented reduce, radix), **27_perf** (bandwidth,
roofline, ncu parsing), **28_cnn** (im2col, batchnorm, full TinyCNN,
conv2d backward), and **29_milestones** (one capstone per Part: SAXPY
benchmark, async copy, max-scan, MLP train, custom torch CUDA op,
ResNet train, tiny-GPT train).

```bash
cd Part8_CUDAlings
./cudalings list                 # see every exercise + status
./cudalings watch                # tight feedback loop while you work
./cudalings hint 23_llm_c_fwd/01_layernorm_kernel
./cudalings solution 23_llm_c_fwd/01_layernorm_kernel   # spoilers
```

Realistic budget: ≈40-60 hours of focused practice to finish all 138.
See `Part8_CUDAlings/README.md` for the full per-chapter map and a
"how to actually work through this" section.

## Inspirations & References

- **Karpathy** — [micrograd](https://github.com/karpathy/micrograd), [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- **George Hotz (Geo)** — [tinygrad](https://github.com/tinygrad/tinygrad)
- **Kirk & Hwu** — *Programming Massively Parallel Processors* (4th ed.)
- **NVIDIA** — [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **Your target** — `cnn_resnet.py` from the Yolah project (256×20 ResNet for board eval)

## The End Goal

By the end of this course, you will have:

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                                                                 │
 │   Python:   model = ResNet(channels=64, blocks=8)               │
 │             gpt   = GPT(GPTConfig(n_layer=6, n_embd=384, ...))   │
 │             for x, y in dataloader:                             │
 │                 logits = model.forward(x)     ←── your CUDA     │
 │                 loss = cross_entropy(logits, y)    kernels      │
 │                 loss.backward()               ←── your autograd │
 │                 optimizer.step()              ←── your optimizer │
 │                                                                 │
 │   C++/CUDA: custom Tensor, Conv2D, BatchNorm, ReLU, Linear      │
 │             attention, layernorm, GELU, AdamW                   │
 │             computational graph with automatic differentiation  │
 │             trained GPT generates Shakespeare-style text        │
 │                                                                 │
 └─────────────────────────────────────────────────────────────────┘
```

A complete, from-scratch deep learning framework — every matrix multiply,
every gradient, every memory transfer — written and understood by you.
Both a ResNet image classifier and a ChatGPT-style language model,
built up kernel by kernel, with progressive exercises drilling each step.
