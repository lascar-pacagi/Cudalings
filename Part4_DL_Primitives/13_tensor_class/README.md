# Chapter 13 -- The Tensor Class: Foundation of cudalearn

## Table of Contents
1. [What Is a Tensor?](#what-is-a-tensor)
2. [Tensor = Data + Metadata](#tensor--data--metadata)
3. [Memory Layout and Strides](#memory-layout-and-strides)
4. [Views and Reshapes](#views-and-reshapes)
5. [Device Management: CPU vs GPU](#device-management-cpu-vs-gpu)
6. [Reference Counting and Shared Storage](#reference-counting-and-shared-storage)
7. [Architecture of Our Tensor Class](#architecture-of-our-tensor-class)
8. [Comparison with Real Frameworks](#comparison-with-real-frameworks)
9. [Operations Our Tensor Needs](#operations-our-tensor-needs)
10. [Programs in This Chapter](#programs-in-this-chapter)

---

## What Is a Tensor?

In mathematics, a tensor is a multi-dimensional generalization of vectors and
matrices. In deep learning frameworks, "tensor" means something more specific:
it is the **fundamental data container** that holds numerical data and knows
how to move between CPU and GPU, how to reshape itself, and how to participate
in automatic differentiation.

```
  Scalar         Vector           Matrix            3D Tensor
  (0-dim)        (1-dim)          (2-dim)           (3-dim)

   42             [1,2,3]         [[1,2,3],         [[[1,2],
                                   [4,5,6]]          [3,4]],
                                                     [[5,6],
                                                      [7,8]],
                                                     [[9,10],
                                                      [11,12]]]

  shape=()       shape=(3,)      shape=(2,3)       shape=(3,2,2)
  0 dimensions   1 dimension     2 dimensions      3 dimensions
  1 element      3 elements      6 elements        12 elements
```

Every neural network is just a sequence of operations on tensors:
- **Weights** are tensors (e.g., a 512x256 matrix for a linear layer)
- **Inputs** are tensors (e.g., a batch of 32 images, each 3x224x224)
- **Activations** are tensors flowing through the network
- **Gradients** are tensors with the same shape as what they differentiate

So the Tensor class is the single most important piece of any DL framework.
Get it right and everything else follows naturally. Get it wrong and you
fight the design for the entire project.

---

## Tensor = Data + Metadata

A tensor is NOT just an array of floats. It is a **data pointer** combined
with **metadata** that describes how to interpret that flat memory as a
multi-dimensional object.

```
  Tensor object (lives on CPU, always)
  +--------------------------------------------------+
  |                                                    |
  |  data_ptr  ──────────> [1.0, 2.0, 3.0, 4.0,      |
  |                         5.0, 6.0, 7.0, 8.0,      |
  |                         9.0, 10.0, 11.0, 12.0]   |
  |                        (flat buffer, may be on    |
  |                         CPU or GPU memory)        |
  |                                                    |
  |  shape     = {3, 4}     (3 rows, 4 columns)      |
  |                                                    |
  |  strides   = {4, 1}     (jump 4 to next row,     |
  |                          jump 1 to next column)   |
  |                                                    |
  |  size      = 12          (total elements)         |
  |                                                    |
  |  device    = GPU         (where the data lives)   |
  |                                                    |
  |  ref_count = 2           (shared with a view)     |
  |                                                    |
  +--------------------------------------------------+
```

Key insight: the Tensor **object** always lives on the CPU (it is a C++
object with vectors, shared pointers, etc.). Only the **data buffer** it
points to may live on the GPU. This is true in PyTorch, tinygrad, and
every other framework.

---

## Memory Layout and Strides

All tensor data is stored as a **flat 1D buffer** in memory. The shape and
strides tell us how to map multi-dimensional indices to positions in that
flat buffer.

### Row-Major (C-order) Layout

By default, tensors use **row-major** order, also called C-order (as
opposed to column-major / Fortran-order). In row-major order, the last
dimension varies fastest.

```
  A 3x4 matrix (shape = {3, 4}):

  Logical view:              Memory layout (flat buffer):

  row 0: [ 0  1  2  3 ]     [ 0  1  2  3  4  5  6  7  8  9  10  11 ]
  row 1: [ 4  5  6  7 ]       ^           ^              ^
  row 2: [ 8  9  10 11]       |           |              |
                               row 0       row 1          row 2
                               starts      starts         starts
                               at offset   at offset      at offset
                               0           4              8
```

### How Strides Work

The **stride** for each dimension tells you how many elements to skip in
the flat buffer to advance by 1 in that dimension.

```
  For a 3x4 matrix (shape = {3, 4}):

  strides = {4, 1}
             ^  ^
             |  |
             |  +-- stride[1] = 1: to move one column right, skip 1 element
             |
             +-- stride[0] = 4: to move one row down, skip 4 elements


  Element at [i, j] is at offset: i * stride[0] + j * stride[1]
                                = i * 4 + j * 1

  Examples:
    [0, 0] -> 0*4 + 0*1 = 0     ✓  (first element)
    [0, 3] -> 0*4 + 3*1 = 3     ✓  (last in first row)
    [1, 0] -> 1*4 + 0*1 = 4     ✓  (first in second row)
    [2, 3] -> 2*4 + 3*1 = 11    ✓  (last element)
```

### 3D Example

```
  A 2x3x4 tensor (shape = {2, 3, 4}):

  strides = {12, 4, 1}
              ^   ^  ^
              |   |  |
              |   |  +-- stride[2] = 1:  move one step in last dim
              |   |
              |   +-- stride[1] = 4:  move one step in middle dim
              |                       (skip one row of 4 elements)
              |
              +-- stride[0] = 12: move one step in first dim
                                  (skip one "matrix" of 3*4 = 12 elements)

  General formula for contiguous row-major:
    stride[k] = product of shape[k+1] * shape[k+2] * ... * shape[ndim-1]
    stride[last] = 1
```

---

## Views and Reshapes

The power of strides is that many operations can be done **without copying
data**. A reshape or transpose just creates a new Tensor object pointing
at the same data buffer with different shape/strides.

### Reshape (No Data Copy)

```
  Original: shape = {3, 4}, strides = {4, 1}

  data = [ 0  1  2  3 | 4  5  6  7 | 8  9  10  11 ]

  After reshape({2, 6}): shape = {2, 6}, strides = {6, 1}

  SAME data = [ 0  1  2  3  4  5 | 6  7  8  9  10  11 ]

  After reshape({12}): shape = {12}, strides = {1}

  SAME data = [ 0  1  2  3  4  5  6  7  8  9  10  11 ]

  +---------+       +---------+       +---------+
  | Tensor A|       | Tensor B|       | Tensor C|
  | shape:  |       | shape:  |       | shape:  |
  | {3,4}   |       | {2,6}   |       | {12}    |
  | stride: |       | stride: |       | stride: |
  | {4,1}   |       | {6,1}   |       | {1}     |
  +----+----+       +----+----+       +----+----+
       |                 |                 |
       v                 v                 v
       +-------------------------------------+
       | SHARED DATA BUFFER (12 floats)      |
       | [ 0  1  2  3  4  5  6  7  8  9 ... ]|
       +-------------------------------------+

  Reshape is O(1) -- it only changes metadata!
```

### Transpose (Changes Strides, No Data Copy)

```
  Original: shape = {3, 4}, strides = {4, 1}

  Logical view:
  [ 0  1  2  3 ]
  [ 4  5  6  7 ]
  [ 8  9  10 11]

  After transpose: shape = {4, 3}, strides = {1, 4}  <-- strides swapped!

  Logical view (same data, different traversal):
  [ 0  4  8  ]    element [0,0] = data[0*1 + 0*4] = data[0] = 0  ✓
  [ 1  5  9  ]    element [1,0] = data[1*1 + 0*4] = data[1] = 1  ✓
  [ 2  6  10 ]    element [0,1] = data[0*1 + 1*4] = data[4] = 4  ✓
  [ 3  7  11 ]    element [2,2] = data[2*1 + 2*4] = data[10] = 10 ✓

  The data buffer is NOT rearranged. We just read it differently.
  But the tensor is now NON-CONTIGUOUS (strides don't decrease).
```

### When You Need contiguous()

Some operations (like feeding data to cuBLAS) require the data to be
contiguous in memory. If you have transposed a tensor, calling
`contiguous()` will copy the data into a new buffer arranged in the
standard row-major order for the current logical shape.

```
  Non-contiguous (transposed):           contiguous() copy:
  shape = {4, 3}, strides = {1, 4}      shape = {4, 3}, strides = {3, 1}

  data = [0 1 2 3 4 5 6 7 8 9 10 11]    data = [0 4 8 1 5 9 2 6 10 3 7 11]
         (original order)                       (new buffer, row-major order
                                                 for the transposed shape)
```

---

## Device Management: CPU vs GPU

A tensor's data can live on the CPU or the GPU. The Tensor object itself
is always on the CPU, but it tracks where its data buffer resides.

```
  CPU Memory                          GPU Memory (VRAM)
  +------------------+                +------------------+
  |                  |                |                  |
  |  Tensor object   |   to_gpu()    |                  |
  |  +------------+  |  ========>    |  data buffer     |
  |  | shape      |  |  cudaMemcpy  |  [1.0 2.0 3.0   |
  |  | strides    |  |  H -> D      |   4.0 5.0 6.0]  |
  |  | device=GPU |--+------ ptr ---|->               |
  |  | data_ptr --+--+              |                  |
  |  +------------+  |              |                  |
  |                  |   to_cpu()    |                  |
  |  data buffer     |  <========   |                  |
  |  [1.0 2.0 ...]  |  cudaMemcpy  |                  |
  |                  |  D -> H      |                  |
  +------------------+              +------------------+
```

Design decisions for our Tensor class:
- **Eager transfer**: `to_gpu()` / `to_cpu()` copy immediately (no lazy eval)
- **New allocation**: transfer creates a new buffer on the target device
- **Chainable**: `auto gpu_t = cpu_t.to_gpu();` returns a new GPU tensor
- We track device via an enum: `Device::CPU` or `Device::GPU`

---

## Reference Counting and Shared Storage

Multiple Tensor objects can share the same underlying data buffer. This
happens when you reshape, view, or slice a tensor. We use C++ `shared_ptr`
for automatic reference counting and cleanup.

```
  shared_ptr reference counting:

  Tensor a = Tensor::ones({3, 4});     // ref_count = 1

  Tensor b = a.reshape({2, 6});        // ref_count = 2
                                        // b shares a's data

  Tensor c = a.reshape({12});          // ref_count = 3
                                        // c also shares a's data

  // When a goes out of scope         -> ref_count = 2 (no free)
  // When b goes out of scope         -> ref_count = 1 (no free)
  // When c goes out of scope         -> ref_count = 0 -> FREE memory!

  +----------+    +----------+    +----------+
  | Tensor a |    | Tensor b |    | Tensor c |
  | {3,4}    |    | {2,6}    |    | {12}     |
  | {4,1}    |    | {6,1}    |    | {1}      |
  +----+-----+    +----+-----+    +----+-----+
       |              |               |
       v              v               v
       +---shared_ptr control block---+
       | ref_count = 3                |
       | data* ---------------------->+---> [12 floats in memory]
       +------------------------------+

  When ref_count hits 0, the custom deleter runs:
    - If device == CPU:  delete[] data;
    - If device == GPU:  cudaFree(data);
```

This is essentially how PyTorch's `Storage` class works. The `Tensor` is a
view into a `Storage`, and multiple tensors can share the same storage.

---

## Architecture of Our Tensor Class

```
  +============================================================+
  |                    Tensor<T> Class                          |
  +============================================================+
  |                                                            |
  |  METADATA (always on CPU):                                 |
  |  +------------------------------------------------------+  |
  |  | shape_     : vector<int>    -- e.g., {2, 3, 4}      |  |
  |  | strides_   : vector<int>    -- e.g., {12, 4, 1}     |  |
  |  | size_      : int            -- total elements (24)   |  |
  |  | device_    : Device         -- CPU or GPU            |  |
  |  | offset_    : int            -- start offset in data  |  |
  |  +------------------------------------------------------+  |
  |                                                            |
  |  DATA (on CPU or GPU):                                     |
  |  +------------------------------------------------------+  |
  |  | data_      : shared_ptr<T>  -- ref-counted pointer   |  |
  |  |              with custom deleter (delete[] or        |  |
  |  |              cudaFree depending on device)           |  |
  |  +------------------------------------------------------+  |
  |                                                            |
  |  CONSTRUCTORS:                                             |
  |  +------------------------------------------------------+  |
  |  | Tensor(shape, device)       -- allocate zeros        |  |
  |  | Tensor(shape, data, device) -- from existing data    |  |
  |  | Tensor(vector<T>)           -- from std::vector      |  |
  |  +------------------------------------------------------+  |
  |                                                            |
  |  STATIC FACTORIES:                                         |
  |  +------------------------------------------------------+  |
  |  | zeros(shape, device)        -- all zeros             |  |
  |  | ones(shape, device)         -- all ones              |  |
  |  | randn(shape, device)        -- normal distribution   |  |
  |  | arange(n, device)           -- [0, 1, 2, ..., n-1]  |  |
  |  +------------------------------------------------------+  |
  |                                                            |
  |  SHAPE OPERATIONS (O(1), no data copy):                    |
  |  +------------------------------------------------------+  |
  |  | reshape(new_shape)          -- change shape+strides  |  |
  |  | view(new_shape)             -- alias for reshape     |  |
  |  | transpose()                 -- swap last 2 strides   |  |
  |  | is_contiguous()             -- check stride pattern  |  |
  |  | contiguous()                -- copy if needed        |  |
  |  +------------------------------------------------------+  |
  |                                                            |
  |  DEVICE TRANSFER:                                          |
  |  +------------------------------------------------------+  |
  |  | to_gpu()                    -- copy data to GPU      |  |
  |  | to_cpu()                    -- copy data to CPU      |  |
  |  | to(Device)                  -- generic transfer      |  |
  |  +------------------------------------------------------+  |
  |                                                            |
  |  ACCESSORS:                                                |
  |  +------------------------------------------------------+  |
  |  | operator()(indices...)      -- element access        |  |
  |  | data_ptr()                  -- raw pointer           |  |
  |  | print()                     -- display contents      |  |
  |  +------------------------------------------------------+  |
  |                                                            |
  +============================================================+

  SEPARATE (tensor_ops.cuh):
  +============================================================+
  |              Element-wise Operations                       |
  +============================================================+
  |  CUDA kernels for:                                         |
  |  add, sub, mul, div          (element-wise, broadcasting) |
  |  add_scalar, mul_scalar      (scalar operations)          |
  |  neg, abs, exp, log, sqrt    (unary operations)           |
  |  Operator overloads: +, -, *, /                           |
  +============================================================+
```

---

## Comparison with Real Frameworks

### PyTorch's torch.Tensor

PyTorch separates `Tensor` and `Storage`. The `Storage` holds the raw data
and the `Tensor` is a view (with shape, strides, offset) into that storage.
Our design is similar but simpler -- we use `shared_ptr` instead of a
separate Storage class.

```
  PyTorch:                              Our cudalearn:

  Tensor -> TensorImpl -> Storage       Tensor<T> -> shared_ptr<T>
            (shape, strides)            (shape, strides, data all in one)

  More flexible (supports                Simpler, sufficient for learning.
  quantization, sparse, etc.)            Easy to understand the full picture.
```

### tinygrad's Approach

tinygrad uses a `LazyBuffer` that defers computation. Operations build a
graph and only execute when you call `.realize()`. This is the lazy
evaluation approach. We use **eager evaluation** -- operations execute
immediately, like PyTorch's default mode.

### Andrej Karpathy's micrograd

micrograd uses a `Value` class that wraps a single scalar and tracks the
computation graph for autograd. Our `Tensor` is the multi-dimensional
generalization. In Chapter 15 (backward_ops), we will add gradient
tracking similar to micrograd's approach but on full tensors.

---

## Operations Our Tensor Needs

For building a neural network, our Tensor class eventually needs:

| Category | Operations | Chapter |
|----------|-----------|---------|
| **Element-wise** | add, sub, mul, div, neg, exp, log, sqrt | This chapter |
| **Reduction** | sum, mean, max (along axes) | Ch 14 |
| **Matrix multiply** | matmul / @ (uses cuBLAS) | Ch 14 |
| **Activation** | ReLU, sigmoid, tanh, softmax | Ch 14 |
| **Shape** | reshape, transpose, contiguous | This chapter |
| **Autograd** | backward(), grad accumulation | Ch 15 |
| **Graph** | computational graph for backprop | Ch 16 |

This chapter builds the foundation (Tensor + element-wise ops). The next
chapters add the operations needed for forward and backward passes.

---

## Programs in This Chapter

| File | Description |
|------|-------------|
| `tensor.cuh` | Tensor class header -- data structure, constructors, shape ops, device transfer |
| `tensor.cu` | Tensor implementation -- memory management, cuRAND, print |
| `tensor_ops.cuh` | Element-wise CUDA kernels and operator overloads |
| `tensor_test.cu` | Test program demonstrating the full Tensor API |

### Building and Running

```bash
make              # build the test program
./tensor_test     # run all tests
```

### Expected Output

The test program creates tensors, transfers between CPU and GPU, performs
arithmetic, reshapes, transposes, and verifies correctness. It should feel
like a mini PyTorch session:

```
auto a = Tensor<float>::randn({3, 4}, Device::GPU);
auto b = Tensor<float>::ones({3, 4}, Device::GPU);
auto c = a + b;           // element-wise addition on GPU
auto d = c.reshape({4, 3}); // reshape, no data copy
d.print();                 // transfers to CPU for display
```
