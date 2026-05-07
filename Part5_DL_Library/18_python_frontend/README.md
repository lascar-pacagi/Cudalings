# Chapter 18: Python Frontend for cudalearn (pybind11 Bindings)

## Overview

We have spent 17 chapters building a complete deep learning library in CUDA C++.
Now we expose it to Python so users can write PyTorch-like training code backed
by our custom CUDA kernels. The bridge between Python and C++ is **pybind11**.

After this chapter, users will write:

```python
import cudalearn

model = cudalearn.Sequential()
model.add("conv1", cudalearn.Conv2d(3, 16, 3, padding=1))
model.add("relu",  cudalearn.ReLU())
model.add("fc",    cudalearn.Linear(16, 10))

optimizer = cudalearn.Adam(model.parameters(), lr=0.001)
criterion = cudalearn.CrossEntropyLoss()

logits = model.forward(x)
loss = criterion.forward(logits, labels)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

This looks almost identical to PyTorch -- but every operation runs our hand-written
CUDA kernels from Chapters 12-17.

---

## The Full Stack

```
  +=====================================================+
  |              Python User Code                       |
  |   model = cudalearn.Sequential()                    |
  |   loss = criterion.forward(model.forward(x), y)     |
  |   loss.backward()                                    |
  +=====================================================+
                        |
                        | Python function call
                        v
  +=====================================================+
  |              pybind11 Binding Layer                  |
  |   (bindings.cu  ->  cudalearn.cpython-*.so)         |
  |                                                      |
  |   - Converts Python args to C++ types                |
  |   - Handles numpy <-> GradTensor conversion          |
  |   - Provides __repr__, properties, docstrings        |
  |   - Transfers ownership via raw pointers             |
  +=====================================================+
                        |
                        | Direct C++ function call
                        v
  +=====================================================+
  |           C++ cudalearn Library (Ch. 17)             |
  |                                                      |
  |   Module, Sequential, Conv2d, Linear, ...            |
  |   SGD, Adam, CosineAnnealingLR                       |
  |   CrossEntropyLoss, MSELoss                          |
  |   DataLoader, GradTensor                             |
  +=====================================================+
                        |
                        | CUDA kernel launch (<<<>>>)
                        v
  +=====================================================+
  |           CUDA Kernels (Chs. 12-16)                 |
  |                                                      |
  |   conv2d_forward_kernel<<<blocks, threads>>>         |
  |   adam_step_kernel<<<blocks, threads>>>               |
  |   cross_entropy_loss_kernel<<<1, 1>>>                |
  |   ...                                                |
  +=====================================================+
                        |
                        | GPU hardware execution
                        v
  +=====================================================+
  |            NVIDIA GPU (Quadro P4200)                 |
  |                                                      |
  |   SM 0: [warp0] [warp1] ...                          |
  |   SM 1: [warp0] [warp1] ...                          |
  |   ...                                                |
  |   Global Memory: tensors, gradients, optimizer state |
  +=====================================================+
```

---

## What is pybind11?

pybind11 is a lightweight header-only C++ library that creates Python bindings
for C++ code. It is the de facto standard for exposing C++ libraries to Python.

Key properties:
- **Header-only**: no separate library to install, just `#include <pybind11/pybind11.h>`
- **C++11 and above**: uses modern C++ features (templates, variadic args, RAII)
- **Automatic type conversion**: converts between Python and C++ types seamlessly
- **Supports classes, virtual functions, operators, properties, NumPy arrays**
- **Used by**: PyTorch (internally), TensorFlow, OpenCV, SciPy, and hundreds more

### Installation

```bash
pip install pybind11
```

This installs the headers. To find include paths for compilation:

```bash
python3 -m pybind11 --includes
# Example output: -I/path/to/python/include -I/path/to/pybind11/include
```

---

## How a Python Call Traverses the Stack

When the user writes `logits = model.forward(x)`, here is exactly what happens:

```
  Python:  logits = model.forward(x)
     |
     |  1. Python looks up "forward" in model's type object
     |     -> finds pybind11-registered C++ method
     |
     v
  pybind11:  Unwrap Python args to C++ types
     |
     |  2. model is a py::class_<Sequential> wrapping a Sequential*
     |     x is a py::class_<GradTensor> wrapping a GradTensor*
     |     pybind11 extracts the raw C++ pointers
     |
     v
  C++:  Sequential::forward(GradTensor* input)
     |
     |  3. For each layer in layers_:
     |       x = layer->forward(x)
     |     Each forward() allocates GPU memory, launches CUDA kernels,
     |     builds the autograd computation graph
     |
     v
  CUDA:  conv2d_forward_kernel<<<blocks, threads>>>(...)
     |
     |  4. GPU executes the kernel: thousands of threads compute
     |     the convolution output in parallel on the Quadro P4200
     |
     v
  C++:  returns GradTensor* output (with backward_fn set)
     |
     |  5. pybind11 wraps the raw C++ pointer back into a Python object
     |     with the correct type (cudalearn.GradTensor)
     |
     v
  Python:  logits is now a cudalearn.GradTensor Python object
           backed by GPU memory with autograd support
```

---

## pybind11 Binding Patterns

### Binding a Class

```cpp
py::class_<GradTensor>(m, "GradTensor", "A GPU tensor with gradient support")
    .def(py::init<int, int, int, int, bool>(),  // Constructor
         py::arg("d0"), py::arg("d1") = 1,       // Named args with defaults
         py::arg("d2") = 1, py::arg("d3") = 1,
         py::arg("requires_grad") = false)
    .def("backward", &GradTensor::backward)       // Method binding
    .def_readonly("size", &GradTensor::size)       // Read-only property
    .def_readwrite("requires_grad",                // Read-write property
                   &GradTensor::requires_grad);
```

### Binding Inheritance (Virtual Methods with Trampolines)

When a C++ class has virtual methods that Python subclasses might override,
pybind11 requires a "trampoline" class:

```cpp
class PyModule : public Module {
public:
    using Module::Module;  // Inherit constructors

    // Override virtual method to dispatch to Python
    GradTensor* forward(GradTensor* input) override {
        PYBIND11_OVERRIDE_PURE(GradTensor*, Module, forward, input);
    }
};
```

This allows both C++ and Python code to call `module.forward()` and get the
correct behavior polymorphically.

### Binding with Return Value Policies

pybind11 needs to know who owns the returned object:

```
  py::return_value_policy::take_ownership
     Python takes ownership, will call delete when refcount hits 0

  py::return_value_policy::reference
     Python does NOT own the object, will NOT delete it
     (C++ side manages lifetime)

  py::return_value_policy::reference_internal
     Like reference, but prevents the parent from being garbage collected
     while the child exists
```

For our library:
- `forward()` returns new GradTensors -> `take_ownership`
- `parameters()` returns references to existing tensors -> `reference_internal`

---

## Memory Management Across the Python/C++ Boundary

```
  Python side                    C++ side                   GPU
  +------------------+          +------------------+       +----------+
  | Python object    |  ------> | C++ object       | ----> | GPU data |
  | (reference       |          | (Sequential,     |       | (float*) |
  |  counted by      |          |  GradTensor,     |       |          |
  |  Python GC)      |          |  Adam, etc.)     |       |          |
  +------------------+          +------------------+       +----------+
       |                              |
       | Python GC decrefs to 0       | C++ destructor runs
       | -> calls C++ destructor      | -> cudaFree(data)
       |                              | -> cudaFree(grad)
```

Key rules:
1. **Sequential owns its layers**: Sequential's destructor does NOT delete
   child modules (they are registered but not owned by raw pointer). In our
   bindings, we use `py::keep_alive<>` to ensure the child layers live as
   long as the Sequential container.

2. **forward() creates new tensors**: The returned GradTensor is owned by
   the Python caller (`take_ownership`). Python's GC will delete it.

3. **parameters() returns references**: The GradTensors returned by
   `parameters()` are owned by the layer (e.g., Conv2d owns its weight_).
   Python gets a reference, not ownership (`reference_internal`).

---

## Comparison with Alternative Binding Methods

| Method       | Complexity | Speed    | NumPy  | Classes | Best for                    |
|------------- |----------- |--------- |------- |-------- |---------------------------- |
| **pybind11** | Medium     | Native   | Yes    | Full    | C++ libraries with OOP      |
| ctypes       | Low        | Native   | Manual | No      | Simple C functions           |
| CFFI         | Low        | Native   | Manual | No      | Simple C functions           |
| Cython       | High       | Native   | Yes    | Partial | Python/C hybrid code         |
| PyTorch ext  | Medium     | Native   | Yes    | Full    | PyTorch-specific extensions  |

**pybind11** is the right choice because:
- Our library is heavily object-oriented (Module inheritance hierarchy)
- We need NumPy interop for data loading
- We want a PyTorch-like API with classes, methods, and properties
- It handles the C++/Python memory boundary cleanly

PyTorch's own C++ extension mechanism (`torch.utils.cpp_extension`) uses
pybind11 under the hood, so we are using the same technology.

---

## Building the Extension Module

### Method 1: Makefile (direct nvcc compilation)

```bash
make
```

This runs nvcc with pybind11 includes and produces a shared library
(`cudalearn.cpython-*.so`) that Python can import directly.

### Method 2: setup.py (setuptools)

```bash
python setup.py build_ext --inplace
```

This uses Python's build system to compile and link the extension. It
automatically finds Python and pybind11 headers.

### What gets produced

```
cudalearn.cpython-310-x86_64-linux-gnu.so    (or similar)
```

This is a standard Python extension module. The filename encodes:
- `cudalearn`: the module name
- `cpython-310`: built for CPython 3.10
- `x86_64-linux-gnu`: platform

Python can import it with just `import cudalearn`.

---

## Files in This Chapter

| File                | Purpose                                                |
|-------------------- |------------------------------------------------------- |
| `bindings.cu`       | pybind11 module definition (the actual bindings)        |
| `setup.py`          | Build script using setuptools                           |
| `Makefile`          | Alternative build using direct nvcc                     |
| `cudalearn_demo.py` | Demo: PyTorch-like training loop using cudalearn        |
| `resnet_preview.py` | Preview: ResNet architecture in Python (for Ch. 19)     |

---

## The Goal

After building, the user experience is:

```python
import cudalearn
import numpy as np

# Create data
x = cudalearn.randn(32, 3, 8, 8)        # Random input tensor on GPU
y_np = np.random.randint(0, 10, 32)       # Random labels

# Build model
model = cudalearn.Sequential()
model.add("conv1", cudalearn.Conv2d(3, 16, 3, padding=1))
model.add("bn1",   cudalearn.BatchNorm2d(16))
model.add("relu",  cudalearn.ReLU())
model.add("pool",  cudalearn.GlobalAvgPool2d())
model.add("fc",    cudalearn.Linear(16, 10))

# Optimizer and loss
optimizer = cudalearn.Adam(model.parameters(), lr=0.001)
criterion = cudalearn.CrossEntropyLoss()

# Training loop (looks like PyTorch!)
for epoch in range(10):
    logits = model.forward(x)
    loss = criterion.forward(logits, y_np)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

This is the power of pybind11: the user writes clean Python, and every
operation runs our hand-crafted CUDA kernels on the GPU.
