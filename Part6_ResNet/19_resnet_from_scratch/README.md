# Chapter 19: ResNet from Scratch in CUDA

## The Capstone — A Complete Deep Learning System

This chapter is the culmination of the entire course. We take every component
built in Parts 4 and 5 — tensors with autograd, convolutions, batch normalization,
optimizers, loss functions — and compose them into a **complete ResNet** that can
be trained end-to-end on GPU.

The target architecture comes from the `cnn_resnet.py` Yolah board game evaluator:
a pre-activation ResNet that classifies 8x8 board positions into three outcomes
(black wins, draw, white wins).

---

## Architecture Overview

### Full Network Diagram

```
    Input: (B, 4, 8, 8)
    |    4 planes: black pieces, white pieces, empty squares, turn indicator
    |
    v
 +------------------------------------------------------+
 |  STEM                                                 |
 |  Conv2d(4 -> C, 3x3, pad=1, no bias)                 |
 |  BatchNorm2d(C)                                       |
 |  ReLU                                                 |
 |                                      (B, C, 8, 8)     |
 +------------------------------------------------------+
    |
    v
 +------------------------------------------------------+
 |  BODY: N x ResBlock (pre-activation)                  |
 |                                                       |
 |  +--------------------------------------------------+ |
 |  | ResBlock 0                                       | |
 |  |  BN -> ReLU -> Conv(C->C) -> BN -> ReLU -> Conv  | |
 |  |  + skip connection from input                    | |
 |  +--------------------------------------------------+ |
 |  | ResBlock 1                                       | |
 |  |  BN -> ReLU -> Conv(C->C) -> BN -> ReLU -> Conv  | |
 |  |  + skip connection from input                    | |
 |  +--------------------------------------------------+ |
 |  |           ...                                    | |
 |  +--------------------------------------------------+ |
 |  | ResBlock N-1                                     | |
 |  |  BN -> ReLU -> Conv(C->C) -> BN -> ReLU -> Conv  | |
 |  |  + skip connection from input                    | |
 |  +--------------------------------------------------+ |
 |                                      (B, C, 8, 8)     |
 +------------------------------------------------------+
    |
    v
 +------------------------------------------------------+
 |  HEAD                                                 |
 |  BatchNorm2d(C)                                       |
 |  ReLU                                                 |
 |  GlobalAvgPool2d            (B, C, 8, 8) -> (B, C)   |
 |  Linear(C -> fc_size)                                 |
 |  ReLU                                                 |
 |  Linear(fc_size -> 3)                     (B, 3)      |
 +------------------------------------------------------+
    |
    v
  Logits: (B, 3)  -->  CrossEntropyLoss  -->  scalar loss
```

For the full Yolah evaluator: C=256, N=20, fc_size=256.

---

### Pre-Activation ResBlock (ResNet-v2) — Detailed Data Flow

```
  Input x: (B, C, 8, 8)
    |
    |-----------------------------+  (identity skip connection)
    |                             |
    v                             |
  BatchNorm2d(C)                  |
    | (B, C, 8, 8)                |
    v                             |
  ReLU                            |
    | (B, C, 8, 8)                |
    v                             |
  Conv2d(C->C, 3x3, pad=1)       |
    | (B, C, 8, 8)                |
    v                             |
  BatchNorm2d(C)                  |
    | (B, C, 8, 8)                |
    v                             |
  ReLU                            |
    | (B, C, 8, 8)                |
    v                             |
  Conv2d(C->C, 3x3, pad=1)       |
    | (B, C, 8, 8)                |
    v                             |
  (+) <---------------------------+  element-wise addition
    |
    v
  Output: (B, C, 8, 8)
```

**Key observation**: All tensors within a ResBlock have the same shape (B, C, 8, 8).
Because input channels == output channels and padding preserves spatial dimensions,
no projection is needed on the skip connection — it is a pure identity.

---

### Post-Activation vs Pre-Activation (ResNet-v1 vs ResNet-v2)

```
  ORIGINAL ResNet (post-act)        PRE-ACTIVATION ResNet (v2)
  He et al. 2015                    He et al. 2016
  ========================          ========================

  Input x                           Input x
    |                                 |
    |---> skip                        |---> skip (IDENTITY)
    |                                 |
    v                                 v
  Conv(C->C)                        BN(C)
    |                                 |
    v                                 v
  BN(C)                             ReLU
    |                                 |
    v                                 v
  ReLU                              Conv(C->C)
    |                                 |
    v                                 v
  Conv(C->C)                        BN(C)
    |                                 |
    v                                 v
  BN(C)                             ReLU
    |                                 |
    v                                 v
  (+) <--- skip                     Conv(C->C)
    |                                 |
    v                                 v
  ReLU  <-- after addition!         (+) <--- skip
    |                                 |
    v                                 v
  Output                            Output
```

**Why pre-activation is better:**

In the original ResNet, the ReLU after the addition means the skip path passes
through a non-linearity. This disrupts the clean gradient flow:

```
  POST-ACTIVATION gradient path:
  grad -> ReLU' -> (+) -> [through block OR skip]
                          The ReLU' can zero out gradients!

  PRE-ACTIVATION gradient path:
  grad -> (+) -> [through block OR skip]
                  The (+) passes gradients UNCHANGED to the skip!
```

With pre-activation, the identity skip creates a **gradient highway** — gradients
flow directly from the loss all the way to the stem without any non-linearities
or multiplicative factors on the shortcut path.

---

### The Gradient Highway

```
  loss
   |
   | d_loss/d_output
   v
  Block N-1: out = F(x) + x
   |         d_out/d_x = d_F/d_x + I    <-- gradient = block_grad + IDENTITY
   |
   | d_loss/d_output * (d_F/d_x + I)
   v
  Block N-2: out = F(x) + x
   |         d_out/d_x = d_F/d_x + I    <-- again, +I means signal passes through
   |
   v
   ...
   |
   v
  Block 0: out = F(x) + x
   |       d_out/d_x = d_F/d_x + I
   |
   v
  Stem
   |
   v
  Input

  Through N blocks, the gradient along the skip path is:
    d_loss/d_stem = d_loss/d_output * I * I * ... * I = d_loss/d_output
                                      (N identity terms)

  The gradient arrives at the stem UNATTENUATED!
  This is why ResNets can be 1000+ layers deep without vanishing gradients.
```

---

## Parameter Count Calculation

For the full Yolah architecture: C=256, N=20, fc_size=256.

```
  Component              Parameters          Count
  =====================  ==================  ===========
  STEM:
    Conv2d(4->256, 3x3)  4*256*3*3           9,216
    BatchNorm2d(256)      256 + 256           512
                                              ---------
    Stem total:                               9,728

  BODY (per ResBlock):
    BatchNorm2d(256)      256 + 256           512
    Conv2d(256->256, 3x3) 256*256*3*3         589,824
    BatchNorm2d(256)      256 + 256           512
    Conv2d(256->256, 3x3) 256*256*3*3         589,824
                                              ---------
    Per-block total:                          1,180,672

  BODY (20 blocks):       20 * 1,180,672      23,613,440

  HEAD:
    BatchNorm2d(256)      256 + 256           512
    Linear(256->256)      256*256 + 256       65,792
    Linear(256->3)        256*3 + 3           771
                                              ---------
    Head total:                               67,075

  =============================================
  GRAND TOTAL:                                23,690,243
  =============================================
```

That is ~23.7 million parameters — a substantial model, but feasible on a
Quadro P4200 (8 GB VRAM). The convolutions dominate: 98.6% of parameters
are in the 3x3 convolution weights.

---

## Initialization Strategy

All convolution weights use **Kaiming (He) initialization**:

```
  weight ~ N(0, sqrt(2 / fan_in))

  For Conv2d(C_in, C_out, 3x3):
    fan_in = C_in * 3 * 3
    std = sqrt(2 / (C_in * 9))

  For Conv2d(256, 256, 3x3):
    std = sqrt(2 / 2304) = 0.0295
```

This ensures that the variance of activations is preserved through each layer,
preventing the signal from exploding or vanishing during the initial forward pass.
Combined with batch normalization (which further stabilizes the activation scale),
this makes training deep networks reliable from the start.

BatchNorm parameters: gamma=1, beta=0 (the "do nothing" starting point).
Linear biases: initialized to 0.

---

## Why This Architecture Works for Board Games

Board game evaluation has specific properties that match ResNet well:

1. **Spatial features**: The 8x8 board is naturally a 2D grid, perfect for
   convolutions. Early layers detect local patterns (pieces, threats);
   deeper layers detect global patterns (pawn structures, king safety).

2. **Translation invariance**: A pattern in one corner is similar to the same
   pattern in another corner. Convolutions share weights across positions.

3. **Deep reasoning**: Board evaluation requires multi-step reasoning
   (if I move here, they respond there, then I...). Each ResBlock adds
   one level of reasoning. 20 blocks = 40 convolution layers of reasoning.

4. **Global classification**: The final output is a single classification
   (black wins / draw / white wins). GlobalAvgPool aggregates information
   from all 64 squares into a single decision.

5. **Compact input**: Only 4 channels (vs 3 for RGB images), making the
   stem very lightweight. The network's capacity is concentrated in the body.

---

## PyTorch vs C++/CUDA: Side-by-Side

```python
# ==================== cnn_resnet.py (PyTorch) ====================

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        return x + residual


class ResNet(nn.Module):
    def __init__(self, channels, nb_blocks, fc_size=256, in_channels=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(
            [ResBlock(channels) for _ in range(nb_blocks)]
        )
        self.head_bn = nn.BatchNorm2d(channels)
        self.fc1 = nn.Linear(channels, fc_size)
        self.fc2 = nn.Linear(fc_size, 3)

    def forward(self, x):
        # Stem
        x = F.relu(self.bn(self.conv(x)))
        # Body
        for block in self.blocks:
            x = block(x)
        # Head
        x = F.relu(self.head_bn(x))
        x = x.mean(dim=[2, 3])  # Global average pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

```cpp
// ==================== resnet.cuh (C++/CUDA) ====================

class ResBlock : public Module {
    BatchNorm2d *bn1_, *bn2_;
    Conv2d *conv1_, *conv2_;

    GradTensor* forward(GradTensor* x) {
        GradTensor* residual = x;                    // residual = x
        x = conv1_->forward(relu1_->forward(         // x = conv1(relu(bn1(x)))
                bn1_->forward(x)));
        x = conv2_->forward(relu2_->forward(         // x = conv2(relu(bn2(x)))
                bn2_->forward(x)));
        return add(x, residual);                     // return x + residual
    }
};

class ResNet : public Module {
    Conv2d* stem_conv_;        // self.conv
    BatchNorm2d* stem_bn_;     // self.bn
    ReLU* stem_relu_;
    vector<ResBlock*> blocks_; // self.blocks
    BatchNorm2d* head_bn_;     // self.head_bn
    GlobalAvgPool2d* gap_;     // x.mean(dim=[2,3])
    Linear* fc1_;              // self.fc1
    Linear* fc2_;              // self.fc2

    GradTensor* forward(GradTensor* x) {
        x = stem_relu_->forward(stem_bn_->forward(   // Stem
                stem_conv_->forward(x)));
        for (auto* block : blocks_)                   // Body
            x = block->forward(x);
        x = head_relu_->forward(                      // Head
                head_bn_->forward(x));
        x = gap_->forward(x);                         // GAP
        x = fc1_relu_->forward(fc1_->forward(x));     // FC1
        x = fc2_->forward(x);                         // FC2
        return x;
    }
};
```

The C++ code is a direct translation. Every `nn.Module` becomes a `Module`,
every `nn.Conv2d` becomes a `Conv2d`, every `F.relu` becomes a `ReLU` layer.
The autograd system handles backward passes automatically, just like PyTorch.

---

## Files in This Chapter

| File | Description |
|------|-------------|
| `resnet.cuh` | Self-contained ResNet implementation (all kernels, layers, optimizer, loss) |
| `resnet_test.cu` | Small ResNet: verify shapes, gradients, one training step |
| `resnet_train.cu` | Full training: 50 epochs on synthetic data, overfit to zero loss |
| `Makefile` | Build all targets with `nvcc -arch=sm_61` |

---

## Building and Running

```bash
# Build everything
make

# Run the test (quick — verifies correctness)
make run_test

# Run full training (takes a few minutes — shows overfitting)
make run_train

# Clean up
make clean
```

---

## What You Built

Starting from raw CUDA kernels in Chapter 12, you have now built:

1. **GPU Tensors** with automatic gradient tracking (Chapter 13)
2. **Forward operations**: convolution, batch norm, ReLU, linear, pooling (Chapter 14)
3. **Backward operations**: gradient computation for every forward op (Chapter 15)
4. **Computational graph**: topological sort + reverse-mode autodiff (Chapter 16)
5. **Library architecture**: Module, Sequential, optimizers, loss functions (Chapter 17)
6. **Python frontend**: PyTorch-like API with pybind11 (Chapter 18)
7. **ResNet**: A complete, trainable deep neural network (THIS CHAPTER)

You have a working deep learning framework. From scratch. In CUDA.
