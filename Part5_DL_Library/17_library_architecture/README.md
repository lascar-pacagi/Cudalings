# Chapter 17: Deep Learning Library Architecture — "cudalearn"

## Overview

This chapter unifies every CUDA kernel, tensor abstraction, and autograd mechanism
from Parts 1-4 into a cohesive, PyTorch-like C++ deep learning library called
**cudalearn**. The goal: write neural network training code that *looks* like
PyTorch but compiles and runs entirely on NVIDIA GPUs via CUDA.

---

## Library Architecture

```
 ============================================================
 |                    USER CODE (library_test.cu)            |
 |  model = Sequential(Conv2d, BN, ReLU, Linear)            |
 |  optimizer = Adam(model.parameters(), lr=0.001)           |
 |  for batch : dataloader                                   |
 |      loss = cross_entropy(model.forward(x), y)            |
 |      loss.backward()                                      |
 |      optimizer.step()                                     |
 ============================================================
          |              |              |              |
          v              v              v              v
 +-------------+ +-------------+ +-----------+ +------------+
 |   Module    | |  Optimizer  | |   Loss    | | DataLoader |
 |  (module.cuh| | (optimizer. | | (loss.cuh)| | (dataloader|
 |  layers.cuh)| |   cuh)      | |           | |   .cuh)    |
 +------+------+ +------+------+ +-----+-----+ +-----+------+
        |               |              |              |
        v               v              v              v
 ============================================================
 |            GradTensor + Autograd Engine                   |
 |   GradTensor wraps raw GPU memory + gradient tracking     |
 |   Backward pass via topological sort of computation graph  |
 |   Operations: matmul, conv2d, batchnorm, relu, softmax    |
 ============================================================
        |               |              |              |
        v               v              v              v
 ============================================================
 |               Raw CUDA Kernels                            |
 |   __global__ matmul_kernel(...)                           |
 |   __global__ conv2d_forward_kernel(...)                   |
 |   __global__ batchnorm_forward_kernel(...)                |
 |   __global__ relu_kernel(...)                             |
 |   __global__ softmax_kernel(...)                          |
 |   __global__ adam_update_kernel(...)                      |
 ============================================================
        |               |              |              |
        v               v              v              v
 ============================================================
 |           CUDA Runtime / GPU Hardware                     |
 |   cudaMalloc, cudaMemcpy, cuRAND, streams                |
 ============================================================
```

---

## The Module Pattern (PyTorch nn.Module)

In PyTorch, every neural network component inherits from `nn.Module`. Our C++
version mirrors this design:

### Key Concepts

1. **`parameters()`** returns all learnable `GradTensor` pointers.
   The base class recursively collects parameters from all registered
   sub-modules plus any directly registered parameters.

2. **`forward()`** defines the computation graph. Each Module subclass
   overrides this to specify how inputs map to outputs.

3. **Nesting**: Modules contain sub-modules. A `Sequential` module holds
   a list of child modules. A `ResBlock` holds two Conv2d layers, two
   BatchNorm2d layers, etc. The parameter collection is recursive.

```
 Sequential
   |
   +-- Conv2d          (parameters: weight, bias)
   |
   +-- BatchNorm2d     (parameters: gamma, beta)
   |                   (buffers: running_mean, running_var)
   |
   +-- ReLU            (no parameters)
   |
   +-- Linear          (parameters: weight, bias)
```

### Parameter vs Buffer

- **Parameter**: A `GradTensor` with `requires_grad = true`. Updated by
  the optimizer. Examples: convolution weights, linear weights, BN gamma/beta.

- **Buffer**: A raw `float*` (or wrapped tensor) that is *not* learnable
  but *is* part of the module's state. Updated by the layer itself during
  forward pass in training mode. Examples: BatchNorm's `running_mean` and
  `running_var`.

```
 Parameter (GradTensor)              Buffer (float*)
 +---------------------+            +---------------------+
 | data (GPU float*)   |            | data (GPU float*)   |
 | grad (GPU float*)   |            | (no gradient)       |
 | requires_grad=true  |            | updated by layer    |
 | updated by Optimizer|            | during training     |
 +---------------------+            +---------------------+
```

---

## Optimizer

The optimizer takes a list of parameter pointers (from `model.parameters()`)
and applies gradient-based updates after each backward pass.

### Adam (used in cnn_resnet.py)

Adam maintains per-parameter first moment (m) and second moment (v):

```
  m_t = beta1 * m_{t-1} + (1 - beta1) * grad
  v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2

  m_hat = m_t / (1 - beta1^t)       // bias correction
  v_hat = v_t / (1 - beta2^t)

  param -= lr * m_hat / (sqrt(v_hat) + eps)
```

### CosineAnnealing Learning Rate Scheduler

Smoothly decays the learning rate following a cosine curve:

```
  lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))

  Learning Rate
  lr_max |*
         | *
         |   *
         |     **
         |        ***
         |            ****
  lr_min |                *****
         +--------------------------> epoch
         0                      T_max
```

---

## DataLoader

The DataLoader handles batching and shuffling of training data:

```
 Full Dataset (N samples)
 +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
 | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
 +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

 After shuffle (Fisher-Yates):
 +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
 | 7| 2|14| 0|11| 5| 9| 3|15| 1| 6|13| 4|10| 8|12|
 +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

 Batches (batch_size=4):
 Batch 0: [ 7,  2, 14,  0]
 Batch 1: [11,  5,  9,  3]
 Batch 2: [15,  1,  6, 13]
 Batch 3: [ 4, 10,  8, 12]
```

Uses pinned (page-locked) memory for the batch buffer so that
`cudaMemcpyAsync` can overlap transfers with computation.

---

## Training Loop Flow

```
 +------------------+
 | DataLoader.next()|-----> batch_x, batch_y
 +------------------+
         |
         v
 +------------------+
 | model.forward(x) |-----> predictions  (builds computation graph)
 +------------------+
         |
         v
 +------------------+
 | CrossEntropyLoss |-----> scalar loss  (extends computation graph)
 +------------------+
         |
         v
 +------------------+
 | loss.backward()  |-----> gradients flow back through graph
 +------------------+       (topological sort, chain rule)
         |
         v
 +------------------+
 | optimizer.step() |-----> update all parameters using gradients
 +------------------+       (Adam: m, v moments + bias correction)
         |
         v
 +------------------+
 | optimizer        |
 |  .zero_grad()    |-----> reset all gradients to zero
 +------------------+
         |
         v
 +------------------+
 | scheduler.step() |-----> adjust learning rate (cosine annealing)
 +------------------+
         |
         +---------> repeat for next batch
```

---

## Target Architecture: ResNet (from cnn_resnet.py)

```
 Input (N, 4, H, W)
       |
 Conv2d(4 -> C, 3x3, pad=1)
       |
 BatchNorm2d(C)
       |
 ReLU
       |
 ResBlock x N_blocks
   |  BN -> ReLU -> Conv(C,C,3x3,pad=1) -> BN -> ReLU -> Conv(C,C,3x3,pad=1)
   |  + skip connection (identity)
       |
 GlobalAvgPool2d  -->  (N, C, 1, 1)  -->  reshape to (N, C)
       |
 Linear(C, num_classes)
       |
 CrossEntropyLoss
```

---

## Files in This Chapter

| File                | Purpose                                       |
|---------------------|-----------------------------------------------|
| `cudalearn.cuh`     | Master header — includes everything           |
| `module.cuh`        | Module base class                             |
| `layers.cuh`        | Conv2d, BN, ReLU, Linear, GAP, Sequential     |
| `optimizer.cuh`     | SGD, Adam, CosineAnnealingLR                  |
| `loss.cuh`          | CrossEntropyLoss                              |
| `dataloader.cuh`    | Batching + shuffling + pinned memory          |
| `library_test.cu`   | Integration test — PyTorch-like training loop |

---

## Compilation

```bash
make          # build library_test
make run      # build and run
make clean    # remove binary
```

Requires: CUDA 11.7, g++-11, Quadro P4200 (compute capability 6.1).
