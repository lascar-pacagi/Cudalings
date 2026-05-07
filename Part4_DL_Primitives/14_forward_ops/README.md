# Chapter 14: Forward-Pass Operations for a ResNet

## Overview

This chapter implements every forward-pass operation needed to run inference
on a ResNet-style convolutional neural network. By the end, we can push a
tensor through: Conv2D, BatchNorm2D, ReLU, Global Average Pooling, Linear
(fully connected), and Cross-Entropy Loss -- all in custom CUDA kernels.

We target a specific ResNet architecture used for board-game evaluation:

```
Input: (B, 4, 8, 8)  -- batch of 4-channel 8x8 boards

Stem:     Conv2d(4 -> C, 3x3, pad=1, no bias)
          BatchNorm2d(C) + ReLU

Residual: 20x ResBlock
              BN -> ReLU -> Conv(C->C, 3x3, pad=1) ->
              BN -> ReLU -> Conv(C->C, 3x3, pad=1) + skip

Head:     BN -> ReLU -> GlobalAvgPool
          Linear(C -> fc) -> ReLU -> Linear(fc -> 3)

Loss:     CrossEntropyLoss
```

---

## Data Flow Through a ResBlock

Each residual block adds the input back to the output (skip connection).
The key insight: if the block learns the identity, the gradient flows
straight through, solving the vanishing gradient problem.

```
                    input x: (B, C, H, W)
                         |
          +--------------+---------------+
          |                              |
          v                              |  (skip / identity)
    BatchNorm2d(C)                       |
          |                              |
        ReLU                             |
          |                              |
    Conv2d(C->C, 3x3, pad=1)            |
          |                              |
    BatchNorm2d(C)                       |
          |                              |
        ReLU                             |
          |                              |
    Conv2d(C->C, 3x3, pad=1)            |
          |                              |
          v                              v
          +------------- ADD ------------+
                         |
                         v
                  output: (B, C, H, W)
```

Tensor shapes remain constant through the block because:
- padding=1 with 3x3 kernel preserves spatial dimensions
- in_channels == out_channels == C

---

## Conv2D Forward Pass

### How a 3x3 Filter Slides Over Input

For a single channel, a 3x3 filter slides across the input spatially.
With padding=1, the output has the same H x W as the input.

```
  Input (1 channel, 5x5, padded to 7x7 with pad=1):

  0  0  0  0  0  0  0        Filter (3x3):
  0 [a  b  c] d  e  0
  0 [f  g  h] i  j  0        w0 w1 w2
  0 [k  l  m] n  o  0        w3 w4 w5
  0  p  q  r  s  t  0        w6 w7 w8
  0  u  v  w  x  y  0
  0  0  0  0  0  0  0

  output[0][0] = a*w0 + b*w1 + c*w2
               + f*w3 + g*w4 + h*w5
               + k*w6 + l*w7 + m*w8

  Slide right by 1:

  0  0  0  0  0  0  0
  0  a [b  c  d] e  0        output[0][1] = b*w0 + c*w1 + d*w2
  0  f [g  h  i] j  0                     + g*w3 + h*w4 + i*w5
  0  k [l  m  n] o  0                     + l*w6 + m*w7 + n*w8
  0  p  q  r  s  t  0
  0  u  v  w  x  y  0
  0  0  0  0  0  0  0
```

### Full Tensor Indexing (Multi-Channel Convolution)

The complete convolution sums over all input channels and the kernel
spatial dimensions:

```
  output[b][oc][oh][ow] =
      SUM over ic  = 0..IC-1          (input channels)
      SUM over kh  = 0..KH-1          (kernel height)
      SUM over kw  = 0..KW-1          (kernel width)
        input[b][ic][oh*stride + kh - pad][ow*stride + kw - pad]
        * weight[oc][ic][kh][kw]

  + bias[oc]  (if bias exists; we use no bias)
```

Dimensions:
```
  input:   (B,  IC, IH, IW)     e.g. (2, 4, 8, 8)
  weight:  (OC, IC, KH, KW)     e.g. (16, 4, 3, 3)
  output:  (B,  OC, OH, OW)     where OH = (IH + 2*pad - KH)/stride + 1

  For 3x3, pad=1, stride=1:   OH = (8 + 2 - 3)/1 + 1 = 8   (same size)
```

### Direct Convolution vs im2col

Two common approaches:
1. **Direct**: Each thread computes one output element by looping over
   ic, kh, kw. Simple, educational, and what we implement here.
2. **im2col**: Rearrange input patches into columns of a matrix, then
   call GEMM. Faster in practice (leverages optimized matmul), but
   harder to understand and uses more memory.

We choose direct convolution for clarity.

---

## BatchNorm2D Forward Pass

BatchNorm normalizes each channel independently across the batch and
spatial dimensions. This stabilizes training by reducing internal
covariate shift.

### Per-Channel Normalization

```
  Input: (B, C, H, W)

  For each channel c:

    1. Compute mean over all B*H*W elements in channel c:

       mean[c] = (1 / (B*H*W)) * SUM_{b,h,w} input[b][c][h][w]

    2. Compute variance:

       var[c] = (1 / (B*H*W)) * SUM_{b,h,w} (input[b][c][h][w] - mean[c])^2

    3. Normalize:

       x_hat[b][c][h][w] = (input[b][c][h][w] - mean[c]) / sqrt(var[c] + eps)

    4. Scale and shift (learnable parameters gamma, beta):

       output[b][c][h][w] = gamma[c] * x_hat[b][c][h][w] + beta[c]


  Channel 0:    Channel 1:    Channel 2:
  +--------+   +--------+    +--------+
  |  batch |   |  batch |    |  batch |    Each channel is normalized
  |  0..B  |   |  0..B  |    |  0..B  |    independently using its
  |  HxW   |   |  HxW   |    |  HxW   |    own mean, var, gamma, beta
  +--------+   +--------+    +--------+
       |             |             |
    mean[0]       mean[1]      mean[2]
    var[0]        var[1]       var[2]
    gamma[0]      gamma[1]     gamma[2]
    beta[0]       beta[1]      beta[2]
```

### Training vs Inference Mode

- **Training**: Compute mean/var from the current mini-batch.
  Update running statistics via exponential moving average:
  ```
  running_mean = (1 - momentum) * running_mean + momentum * batch_mean
  running_var  = (1 - momentum) * running_var  + momentum * batch_var
  ```
  (PyTorch uses momentum=0.1 by default)

- **Inference**: Use the accumulated running_mean and running_var
  (no batch statistics computed).

---

## ReLU: Rectified Linear Unit

The simplest activation function:

```
  ReLU(x) = max(0, x)

  output[i] = x[i]  if x[i] > 0
            = 0     otherwise
```

Element-wise, no parameters, no cross-element dependencies. One thread
per element is the natural parallelization.

---

## Linear (Fully Connected) Layer

A matrix multiply plus optional bias:

```
  y = x @ W^T + b

  input:   (B, in_features)
  weight:  (out_features, in_features)
  bias:    (out_features,)
  output:  (B, out_features)

  output[b][j] = SUM_{k=0}^{in_features-1} input[b][k] * weight[j][k] + bias[j]
```

We implement this as a tiled matrix multiplication kernel similar to
Chapter 12, operating on the flattened input.

---

## Global Average Pooling

Reduces spatial dimensions by averaging:

```
  input:  (B, C, H, W)
  output: (B, C)

  output[b][c] = (1 / (H*W)) * SUM_{h,w} input[b][c][h][w]

  Example: (2, 16, 8, 8) -> (2, 16)
           Each of the 16 channels is averaged over its 64 spatial elements
```

This bridges the convolutional layers (4D tensors) and the fully
connected layers (2D tensors).

---

## Cross-Entropy Loss

Combines softmax and negative log-likelihood:

```
  1. Numerically stable log-softmax:
     log_softmax[b][c] = logits[b][c] - log(SUM_j exp(logits[b][j]))

     For stability, subtract max before exp:
     m = max_j(logits[b][j])
     log_softmax[b][c] = (logits[b][c] - m) - log(SUM_j exp(logits[b][j] - m))

  2. Negative log-likelihood:
     loss[b] = -log_softmax[b][target[b]]

  3. Mean over batch:
     loss = (1/B) * SUM_b loss[b]
```

---

## File Organization

| File | Contents |
|------|----------|
| `conv2d_forward.cu` | Conv2D forward kernel + test |
| `batchnorm_forward.cu` | BatchNorm2D forward (mean/var + normalize) + test |
| `linear_relu_pool.cu` | Linear, ReLU, GlobalAvgPool, CrossEntropy + tests |
| `forward_test.cu` | Integration test: full ResNet forward pass |
| `Makefile` | Build all targets |

All files use the Tensor class from Chapter 13 via relative include paths.

---

## Building and Running

```bash
make              # Build all targets
./conv2d_test     # Test Conv2D forward
./batchnorm_test  # Test BatchNorm forward
./lrp_test        # Test Linear, ReLU, Pool, CrossEntropy
./forward_test    # Integration: full ResNet forward pass
make clean        # Remove binaries
```
