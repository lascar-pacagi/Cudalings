# Chapter 15: Backward-Pass Operations (Gradient Computation)

## Overview

This chapter implements the **backward pass** (gradient computation) for every
forward-pass operation from Chapter 14. This is the heart of backpropagation:
the chain rule applied systematically on the GPU.

Every neural network trains by:
1. **Forward pass**: compute the output (loss) given inputs and weights.
2. **Backward pass**: compute the gradient of the loss with respect to every
   weight and intermediate activation, flowing backwards from the loss.
3. **Update**: adjust weights in the direction that reduces the loss.

This chapter focuses entirely on step 2.

---

## The Chain Rule and Backpropagation

### Single Operation

If we have a function `y = f(x)` and a downstream loss `L`, then:

```
dL/dx = dL/dy * dy/dx
        ^^^^^   ^^^^^
        |       local gradient (Jacobian of f)
        |
        upstream gradient (what we receive from the next layer)
```

### Chained Operations

For a chain `x -> f -> g -> h -> L`:

```
dL/dx = dL/dh * dh/dg * dg/df * df/dx
```

We compute this **right to left**, reusing intermediate results.

### Forward vs. Backward Data Flow

```
                     FORWARD PASS
                     ============

  Input -----> Conv -----> BN -----> ReLU -----> ... -----> Loss
  x            y=conv(x)  z=bn(y)   a=relu(z)              L

                     BACKWARD PASS
                     =============

  dL/dx <----- dL/dy <--- dL/dz <-- dL/da <---- ... <----- dL/dL = 1
  grad_input   grad_y     grad_z    grad_a                  (seed = 1)

  Direction of computation: RIGHT to LEFT (loss -> input)
  Each layer receives dL/d(output) and produces dL/d(input) + dL/d(params)
```

**Key insight**: in the backward pass, every layer receives the gradient of the
loss with respect to its *output* (called `grad_output` or `upstream gradient`),
and must produce:
1. The gradient w.r.t. its *input* (to pass to the previous layer)
2. The gradient w.r.t. its *parameters* (to update the weights)

---

## Operation-by-Operation Gradient Derivations

### 1. ReLU Backward

**Forward**: `y = max(0, x)`

**Backward**: The derivative of max(0, x) is a step function:

```
dy/dx = 1   if x > 0
      = 0   if x <= 0

Therefore:
  dL/dx = dL/dy * (x > 0)
```

This is the simplest backward operation: element-wise multiply the upstream
gradient by a binary mask indicating where the input was positive.

```
  Forward:  x = [-2, 3, -1, 5, 0, 7]
                  |  |   |  |  |  |
  mask:          [0, 1,  0, 1, 0, 1]
                  |  |   |  |  |  |
  Backward: dL/dx = grad_output * mask

  If grad_output = [0.5, -0.3, 0.1, 0.8, -0.2, 0.4]
  Then dL/dx =     [0.0, -0.3, 0.0, 0.8,  0.0, 0.4]
```

**No parameters** -- ReLU has nothing to update.

---

### 2. Linear (Fully Connected) Backward

**Forward**: `y = x @ W^T + b`  where x is (B, K), W is (J, K), b is (J,)

**Backward** -- three gradients needed:

```
  Given: dL/dy  shape (B, J)     -- upstream gradient

  (a) grad_input:
      y[b][j] = sum_k x[b][k] * W[j][k] + b[j]

      dL/dx[b][k] = sum_j dL/dy[b][j] * dy[b][j]/dx[b][k]
                   = sum_j dL/dy[b][j] * W[j][k]

      In matrix form:  dL/dx = dL/dy @ W      shape (B, K)

  (b) grad_weight:
      dL/dW[j][k] = sum_b dL/dy[b][j] * dy[b][j]/dW[j][k]
                   = sum_b dL/dy[b][j] * x[b][k]

      In matrix form:  dL/dW = (dL/dy)^T @ x    shape (J, K)

  (c) grad_bias:
      dL/db[j] = sum_b dL/dy[b][j]    (sum over batch)

      In matrix form:  dL/db = sum_b dL/dy[b][:]   shape (J,)
```

---

### 3. Conv2D Backward

This is the most complex gradient geometrically. There are two gradients:

#### 3a. Gradient w.r.t. Input (grad_input)

The forward pass computes:
```
  output[b][oc][oh][ow] = sum_{ic,kh,kw} input[b][ic][ih][iw] * weight[oc][ic][kh][kw]
  where ih = oh*s + kh - pad, iw = ow*s + kw - pad
```

For the backward pass, we need dL/d(input[b][ic][ih][iw]):
```
  dL/d(input[b][ic][ih][iw]) = sum over all (oc, oh, ow) where this input contributed

  For stride=1:
    oh = ih - kh + pad   (rearranging ih = oh + kh - pad)

  So:
    dL/d(input[b][ic][ih][iw]) =
      sum_{oc} sum_{kh,kw}  grad_output[b][oc][ih-kh+pad][iw-kw+pad]
                             * weight[oc][ic][kh][kw]
```

This is equivalent to convolving grad_output with the **180-degree rotated**
(flipped) filter:

```
  FORWARD CONVOLUTION:
  +---+---+---+         +---+---+---+
  | w00 w01 w02|         | i  i  i  |
  | w10 w11 w12|  (*)    | i  i  i  |   =  output
  | w20 w21 w22|         | i  i  i  |
  +---+---+---+         +---+---+---+

  BACKWARD (grad_input):
  +---+---+---+         +---+---+---+
  | w22 w21 w20|         | g  g  g  |
  | w12 w11 w10|  (*)    | g  g  g  |   =  grad_input
  | w02 w01 w00|         | g  g  g  |
  +---+---+---+         +---+---+---+
  (filter rotated 180)   (grad_output, zero-padded by KH-1-pad)

  The rotation means: weight_rot[ic][oc][kh][kw] = weight[oc][ic][KH-1-kh][KW-1-kw]
```

#### Index Mapping for grad_input (stride=1)

```
  For each input position (b, ic, ih, iw):

    dL/d(input[b][ic][ih][iw]) = SUM_{oc} SUM_{kh=0}^{KH-1} SUM_{kw=0}^{KW-1}
        weight[oc][ic][kh][kw] * grad_output[b][oc][oh][ow]

    where oh = ih - kh + pad
          ow = iw - kw + pad

    Bounds check: 0 <= oh < OH  and  0 <= ow < OW
```

#### 3b. Gradient w.r.t. Weight (grad_weight)

```
  dL/d(weight[oc][ic][kh][kw]) =
      sum_{b} sum_{oh,ow} grad_output[b][oc][oh][ow]
                           * input[b][ic][oh*s + kh - pad][ow*s + kw - pad]

  This is a correlation: slide grad_output over input to accumulate the gradient.
```

---

### 4. BatchNorm2D Backward

This is notoriously the most complex gradient in standard neural networks.

**Forward** (per channel c, N = B*H*W elements):
```
  mu    = (1/N) * sum(x)
  var   = (1/N) * sum((x - mu)^2)
  x_hat = (x - mu) / sqrt(var + eps)
  y     = gamma * x_hat + beta
```

**Backward** -- we need dL/dgamma, dL/dbeta, dL/dx:

```
  Step 1:  dL/dbeta[c] = sum_{b,h,w} dL/dy[b][c][h][w]

  Step 2:  dL/dgamma[c] = sum_{b,h,w} dL/dy[b][c][h][w] * x_hat[b][c][h][w]

  Step 3:  dL/dx_hat = dL/dy * gamma[c]

  Step 4:  dL/dvar = sum(dL/dx_hat * (x - mu)) * (-0.5) * (var + eps)^(-3/2)

  Step 5:  dL/dmu = sum(dL/dx_hat) * (-1/sqrt(var + eps))
                   + dL/dvar * (-2/N) * sum(x - mu)
                   Note: sum(x - mu) = 0 by definition of mu, so second term = 0

  Step 6:  dL/dx = dL/dx_hat * (1/sqrt(var + eps))
                 + dL/dvar * (2/N) * (x - mu)
                 + dL/dmu * (1/N)

  Simplification (combining steps 3-6):

    inv_std = 1 / sqrt(var + eps)

    dL/dx = (1/N) * gamma * inv_std * (
                N * dL/dy
              - sum(dL/dy)
              - x_hat * sum(dL/dy * x_hat)
            )
```

This simplified form requires two reductions per channel (sum of dL/dy and
sum of dL/dy * x_hat), then one element-wise pass.

---

### 5. Global Average Pooling Backward

**Forward**: `y[b][c] = (1/HW) * sum_{h,w} x[b][c][h][w]`

**Backward**: The gradient is distributed equally to all spatial positions:
```
  dL/dx[b][c][h][w] = dL/dy[b][c] * (1 / (H * W))

  Every spatial position gets the SAME gradient, scaled by 1/HW.

  Example (H=W=2, so HW=4):

    grad_output (after GAP): [0.8]      (scalar per channel)

    grad_input (before GAP):
    +------+------+
    | 0.2  | 0.2  |    Each position gets 0.8 / 4 = 0.2
    +------+------+
    | 0.2  | 0.2  |
    +------+------+
```

---

### 6. Cross-Entropy Loss Backward

**Forward**: `L = -log(softmax(logits)[target])`

**Backward**: The gradient of cross-entropy loss w.r.t. logits has a
beautifully simple form:

```
  dL/d(logits[b][j]) = softmax(logits[b])[j] - one_hot(target[b])[j]

  In other words:
    = softmax(logits)[j]       for j != target
    = softmax(logits)[j] - 1   for j == target

  This is divided by B for the mean reduction.
```

This is one of the rare cases where the gradient is *simpler* than the
forward pass.

---

### 7. Residual Addition Backward

**Forward**: `y = a + b` (element-wise addition for skip connection)

**Backward**: Addition distributes the gradient to both branches:
```
  dL/da = dL/dy
  dL/db = dL/dy

  The gradient is simply COPIED to both branches (no transformation).
```

---

## Gradient Flow Through a ResBlock

```
                        FORWARD                              BACKWARD
                        =======                              ========

  x ----+----------> BN -> ReLU -> Conv --+      dL/dx <----+--------- dL/d(bn) <-- ...
        |            -> BN -> ReLU -> Conv |            (sum)|
        |                                 |                  |
        |            (residual path)      v                  |
        +--------------------> (+) -----> y      dL/dy ------+-------> dL/dy (copy)
                            addition                         |
                                                             |
                                                    Both branches get
                                                    the SAME gradient.
                                                    They are SUMMED at
                                                    the branch point.

  Key insight: The residual connection provides a "gradient highway."
  Even if the conv path kills gradients (vanishing), the identity path
  ensures dL/dx >= dL/dy. This is WHY ResNets can train very deep networks.
```

---

## Files in This Chapter

| File | Description |
|------|-------------|
| `conv2d_backward.cu` | Conv2D backward: grad_input (rotated filter) + grad_weight (correlation) |
| `batchnorm_backward.cu` | BatchNorm2D backward: full derivation with parallel reductions |
| `simple_backward.cu` | ReLU, Linear, GAP, CrossEntropy, Residual backward ops |
| `gradient_check.cu` | Full chain: forward -> loss -> backward, validated via finite differences |
| `Makefile` | Build all four executables |

---

## Numerical Gradient Checking

The **finite difference** method validates our analytical gradients:

```
  Numerical gradient for parameter p:

    dL/dp_numerical = (L(p + eps) - L(p - eps)) / (2 * eps)

  Compare against:
    dL/dp_analytical = our backward pass result

  If they match (relative error < 1e-3 for float32), our backward pass is correct.
```

This is the **single most important debugging tool** in deep learning
framework development. Every gradient we implement is verified this way.

---

## Building and Running

```bash
make all          # Build all executables
./conv2d_bwd_test      # Test conv2d backward
./batchnorm_bwd_test   # Test batchnorm backward
./simple_bwd_test      # Test simple backward ops
./gradient_check       # Full end-to-end gradient check
make clean        # Remove executables
```
