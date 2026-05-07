# Chapter 16: The Computational Graph (Autograd Engine)

## Overview

This chapter builds the **autograd engine** — the computational graph that
records operations during the forward pass and automatically computes gradients
during the backward pass. This is the machinery that makes `loss.backward()`
work in PyTorch.

Inspired by:
- Andrej Karpathy's **micrograd** (a tiny scalar autograd engine)
- PyTorch's **autograd** system (tape-based, define-by-run)

After this chapter you will have a working GPU-accelerated autograd engine
that can differentiate through Conv2D → BatchNorm → ReLU → Residual → GAP →
Linear → CrossEntropy chains — everything needed to train a ResNet.

---

## What Is a Computational Graph?

A computational graph is a **Directed Acyclic Graph (DAG)** where:
- **Nodes** represent operations (add, multiply, conv2d, relu, ...)
- **Edges** represent tensors flowing between operations

Every mathematical expression can be decomposed into a DAG of elementary
operations. The graph records *what happened* during the forward pass so
that gradients can be computed during the backward pass.

### Example: z = (a * b) + c

```
 COMPUTATIONAL GRAPH for z = (a * b) + c
 =========================================

  Leaf tensors          Operations           Output
  (inputs)              (interior nodes)

   +---+
   | a |---\
   +---+    \    +-------+
             >---| mul   |---\
   +---+    /    | y=a*b |    \    +-------+     +---+
   | b |---/     +-------+     >---| add   |---->| z |
   +---+                      /    | z=y+c |     +---+
                  +---+       /    +-------+
                  | c |------/
                  +---+

  LEGEND:
   +---+
   | x |   = Tensor node (stores data + gradient)
   +---+

   +-------+
   | op    |   = Operation node (stores backward function)
   +-------+
```

Each tensor in the graph stores:
- **data**: the actual numerical values (on GPU)
- **grad**: the gradient dL/d(self), accumulated during backward pass
- **backward_fn**: a closure that computes gradients for this node's inputs
- **children**: pointers to the input tensors (edges in the DAG)

---

## Forward Pass vs. Backward Pass

```
                    FORWARD PASS (left to right)
                    ============================

    a = 2.0        y = a * b        z = y + c
    b = 3.0        y = 6.0          z = 6.0 + 1.0
    c = 1.0                         z = 7.0

   +---+          +-------+          +-------+         +---+
   | a |--------->|       |          |       |-------->| z |
   | 2 |          | mul   |--------->| add   |         | 7 |
   | b |--------->| y = 6 |          | z = 7 |         +---+
   | 3 |          +-------+          |       |
   +---+                       +---->|       |
                  +---+        |     +-------+
                  | c |--------+
                  | 1 |
                  +---+


                    BACKWARD PASS (right to left)
                    ==============================

    dz/da = b = 3       dz/dy = 1           dz/dz = 1  (seed)
    dz/db = a = 2       dz/dc = 1

   +--------+       +-----------+       +-----------+       +--------+
   | a      |<------| mul bwd   |<------| add bwd   |<------| z      |
   | grad=3 |       | da = b*dz |       | dy = 1*dz |       | grad=1 |
   | b      |<------| db = a*dz |       | dc = 1*dz |       +--------+
   | grad=2 |       +-----------+       +-----------+
   +--------+                                |
                    +--------+               |
                    | c      |<--------------+
                    | grad=1 |
                    +--------+
```

**Forward pass**: evaluate operations left-to-right, populating `.data` fields.
**Backward pass**: propagate gradients right-to-left using the chain rule.

---

## How PyTorch Records the Graph (Define-by-Run / Tape-based)

PyTorch uses a **define-by-run** approach (also called "tape-based"):

1. There is no separate "graph definition" step.
2. Every time you call an operation (e.g., `torch.matmul`), PyTorch:
   - Computes the forward result
   - Creates a `grad_fn` node recording the operation
   - Links the output tensor to its `grad_fn`
   - Links the `grad_fn` back to the input tensors
3. The graph is built *dynamically* as operations execute.
4. Calling `loss.backward()` traverses the graph backwards.

This is in contrast to TensorFlow 1.x's **define-and-run** approach where
you first build a static graph, then execute it.

**Advantage of define-by-run**: you can use normal Python control flow
(if/else, loops) and the graph naturally adapts.

---

## Karpathy's micrograd Approach

In micrograd, each `Value` object stores:

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data           # scalar value
        self.grad = 0.0            # dL/d(self)
        self._backward = lambda: None  # closure that computes children's grads
        self._prev = set(_children)    # set of input Values
        self._op = _op                 # operation name (for debugging)
```

When you do `c = a + b`, it creates a new Value with:
- `c._prev = {a, b}`
- `c._backward` = a closure that adds `c.grad` to `a.grad` and `b.grad`

Our approach extends this to GPU tensors with full CUDA kernel support.

---

## Our GradTensor Design

```
 GradTensor Object Layout
 =========================

  +--------------------------------------------------+
  |  GradTensor                                      |
  |                                                  |
  |  name: "conv1.weight"                            |
  |  shape: [64, 3, 3, 3]                            |
  |  size: 1728                                      |
  |  requires_grad: true                             |
  |                                                  |
  |  data -----> [GPU float* array, 1728 elements]   |
  |  grad -----> [GPU float* array, 1728 elements]   |
  |                                                  |
  |  backward_fn: std::function<void()>              |
  |    (closure capturing input tensors + saved       |
  |     activations, calls CUDA backward kernels)    |
  |                                                  |
  |  children: vector<shared_ptr<GradTensor>>        |
  |    [ptr to input1, ptr to input2, ...]           |
  |                                                  |
  +--------------------------------------------------+
```

Key design decisions:
- Use `shared_ptr<GradTensor>` everywhere to manage lifetimes
- `backward_fn` captures inputs by shared_ptr (prevents dangling pointers)
- Leaf tensors (parameters) have no backward_fn or children
- `requires_grad` controls whether gradients are tracked

---

## Topological Sort for Backward Pass

When `loss.backward()` is called, we must process nodes in the correct order.
We need every node's gradient to be fully accumulated *before* we propagate
through it. This requires a **reverse topological sort**.

```
 TOPOLOGICAL SORT via DFS
 =========================

  Graph:                    DFS Post-order:       Reverse (backward order):

    a --\                   Visit order:           Process order:
         >-- y --\          1. a (leaf, push)      5. z  (start here, grad=1)
    b --/         >-- z     2. b (leaf, push)      4. y  (receives grad from z)
                 /          3. y (push after a,b)  3. c  (receives grad from z)
    c ----------/           4. c (leaf, push)      2. b  (receives grad from y)
                            5. z (push after y,c)  1. a  (receives grad from y)

  Algorithm:
  ==========

  function topo_sort(node):
      visited = set()
      order = []

      function dfs(v):
          if v in visited: return
          visited.add(v)
          for child in v.children:
              dfs(child)
          order.append(v)          // post-order: add after visiting children

      dfs(node)
      return reverse(order)        // reverse post-order = valid backward order
```

**Why reverse post-order?** Because in post-order, a node appears *after*
all its children. Reversing this means a node appears *before* all its
children, so when we process a node, all nodes that depend on it have
already propagated their gradients to it.

---

## Gradient Accumulation

When a tensor is used in multiple operations (i.e., it has multiple consumers
in the graph), its gradient must be **accumulated** (summed) from all paths.

```
 GRADIENT ACCUMULATION
 =====================

  When tensor 'x' is used twice:

                +-------+
           /--->| op1   |---> y1
   +---+  /     +-------+
   | x |--
   +---+  \     +-------+
           \--->| op2   |---> y2
                +-------+

  Forward: x feeds into both op1 and op2
  Backward: x receives gradients from BOTH paths

    x.grad = dL/dy1 * dy1/dx  +  dL/dy2 * dy2/dx
             ^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^
             gradient from op1    gradient from op2

  This is why we ACCUMULATE (+=) gradients, never overwrite (=).

  EXAMPLE (x is used in both addition and multiplication):

    y1 = x + 3     -->  dy1/dx = 1
    y2 = x * 2     -->  dy2/dx = 2
    z  = y1 + y2   -->  dz/dy1 = 1, dz/dy2 = 1

    dz/dx = dz/dy1 * dy1/dx + dz/dy2 * dy2/dx
          = 1 * 1 + 1 * 2
          = 3

    If x = 5:  y1 = 8, y2 = 10, z = 18
    Nudge x by eps: y1 = 8+eps, y2 = 10+2*eps, z = 18+3*eps
    => dz/dx = 3  ✓
```

---

## Residual Block: Fork/Join in the Graph

The ResNet skip connection creates a **fork** (one tensor feeds two paths)
and a **join** (two paths are added together). This is the canonical example
of gradient accumulation.

```
 RESIDUAL BLOCK COMPUTATIONAL GRAPH
 ====================================

                           RESIDUAL PATH (shortcut)
                    +---------------------------------------------+
                    |                                             |
                    |                                             v
   +---+      +--------+     +--------+     +--------+     +--------+     +--------+
   | x |----->| Conv   |---->| BN     |---->| ReLU   |---->|  ADD   |---->| out    |
   +---+  |   | 3x3    |     |        |     |        |     |  y + x |     +--------+
          |   +--------+     +--------+     +--------+     +--------+
          |                                                     ^
          |                                                     |
          +-----------------------------------------------------+
                           SKIP CONNECTION

  During BACKWARD:

                           grad flows through identity (1x)
                    +---------------------------------------------+
                    |                                             |
                    v                                             |
   +---+      +--------+     +--------+     +--------+     +--------+     +--------+
   | dx |<----| Conv   |<----| BN     |<----| ReLU   |<----|  ADD   |<----| dout   |
   +---+  |   | bwd    |     | bwd    |     | bwd    |     | dy=dout|     +--------+
          |   +--------+     +--------+     +--------+     | dx=dout|
          |                                                +--------+
          |                                                     |
          |   (accumulated!)                                    |
          +-----------------------------------------------------+

  KEY: The gradient at 'x' is the SUM of:
    1. Gradient flowing through the conv-bn-relu path (transformed)
    2. Gradient flowing through the skip connection (unchanged)

  This is why ResNets train so well: the skip connection provides a
  "gradient highway" where gradients flow unchanged, preventing
  vanishing gradients in deep networks.
```

---

## Files in This Chapter

| File               | Description                                           |
|--------------------|-------------------------------------------------------|
| `grad_tensor.cuh`  | GradTensor class — autograd-enabled GPU tensor        |
| `autograd_ops.cuh` | Forward+backward operations with automatic gradient   |
| `autograd_test.cu` | Test suite: polynomial, linear, conv chain, residual  |
| `graph_viz.cu`     | Visualize the computational graph as ASCII / DOT      |
| `Makefile`         | Build system for all targets                          |

---

## How to Build and Run

```bash
make                # Build everything
./autograd_test     # Run all gradient tests
./graph_viz         # Visualize a mini-ResNet computational graph
make clean          # Remove binaries
```

---

## Key Takeaways

1. **Computational graph** = DAG of operations, built dynamically during forward pass
2. **Each tensor** stores data, gradient, backward function, and links to its inputs
3. **Backward pass** = reverse topological sort + gradient propagation via chain rule
4. **Gradient accumulation**: when a tensor is used multiple times, gradients are summed
5. **Residual connections** create fork/join patterns that provide gradient highways
6. **Our engine** runs entirely on the GPU — all forward and backward kernels are CUDA
