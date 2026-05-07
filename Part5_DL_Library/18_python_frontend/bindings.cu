/*******************************************************************************
 * bindings.cu -- Python Bindings for the cudalearn Library (pybind11)
 *
 * This file creates a Python extension module called "cudalearn" that exposes
 * all C++ classes and functions from our CUDA deep learning library to Python.
 *
 * After compilation, users can write:
 *
 *   import cudalearn
 *   model = cudalearn.Sequential()
 *   model.add("fc", cudalearn.Linear(784, 10))
 *   optimizer = cudalearn.Adam(model.parameters(), lr=0.001)
 *   ...
 *
 * This file is compiled with nvcc (because it includes CUDA kernels via
 * cudalearn.cuh) and linked as a shared library (.so) that Python can import.
 *
 * Build:
 *   make                              (uses Makefile)
 *   python setup.py build_ext --inplace  (uses setuptools)
 *
 * Key pybind11 concepts used:
 *   - py::class_<T>      : bind a C++ class to Python
 *   - py::init<Args...>  : bind a constructor
 *   - .def("name", &T::method) : bind a method
 *   - .def_readonly/readwrite  : bind data members as properties
 *   - py::arg("name") = default : named arguments with defaults
 *   - py::return_value_policy   : control ownership of returned objects
 *   - py::keep_alive<1, 2>     : prevent GC of arg while return value lives
 *   - PYBIND11_OVERRIDE_PURE   : trampoline for virtual methods
 *
 * Memory ownership rules:
 *   - forward() returns new GradTensors -> take_ownership
 *   - parameters() returns refs to existing tensors -> reference_internal
 *   - Sequential.add() keeps child alive -> keep_alive
 ******************************************************************************/

// =============================================================================
// Include the entire cudalearn library (all CUDA kernels + C++ classes)
// =============================================================================
// This single include pulls in:
//   module.cuh    -> GradTensor, Module
//   layers.cuh    -> Conv2d, BatchNorm2d, ReLU, Linear, GlobalAvgPool2d, Sequential
//   optimizer.cuh -> SGD, Adam, CosineAnnealingLR
//   loss.cuh      -> CrossEntropyLoss, MSELoss
//   dataloader.cuh-> DataLoader
#include "../17_library_architecture/cudalearn.cuh"

// =============================================================================
// pybind11 headers
// =============================================================================
// pybind11.h:       core binding functionality (classes, functions, modules)
// stl.h:            automatic conversion for std::vector, std::string, etc.
// numpy.h:          numpy array <-> C++ array interop
// functional.h:     std::function binding support
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

// Namespace alias -- pybind11 convention is to use "py"
namespace py = pybind11;


// =============================================================================
// Trampoline Class for Module (enables Python subclassing)
// =============================================================================
// Problem: Module has a pure virtual method forward(). If we want Python
// classes to inherit from Module and override forward(), pybind11 needs
// a "trampoline" class that redirects C++ virtual calls to Python.
//
// How it works:
//   1. PyModule inherits from Module
//   2. It overrides forward() with PYBIND11_OVERRIDE_PURE
//   3. When C++ code calls module->forward(x), it checks:
//      - Is this object actually a Python subclass? If yes, call Python's forward()
//      - Otherwise, call the C++ implementation
//
// This enables patterns like:
//   class MyModel(cudalearn.Module):
//       def forward(self, x):
//           return self.fc(x)
//
// The trampoline ensures that even when C++ code (e.g., Sequential) calls
// forward() on this object, it dispatches to the Python implementation.
// =============================================================================

class PyModule : public Module {
public:
    // Inherit all constructors from Module
    using Module::Module;

    // Trampoline: redirect virtual forward() calls to Python
    // PYBIND11_OVERRIDE_PURE means:
    //   - This is a PURE virtual method (no C++ fallback)
    //   - Return type is GradTensor*
    //   - The C++ class is Module
    //   - The method name is "forward"
    //   - The argument is "input"
    GradTensor* forward(GradTensor* input) override {
        PYBIND11_OVERRIDE_PURE(
            GradTensor*,    // Return type
            Module,         // Parent C++ class
            forward,        // Method name (must match Python method name)
            input           // Arguments
        );
    }
};


// =============================================================================
// Utility: Create a GradTensor from a NumPy array
// =============================================================================
// This function copies data FROM a numpy array on the CPU TO a GradTensor
// on the GPU. It handles the shape mapping and cudaMemcpy.
//
// Usage from Python:
//   x_np = np.random.randn(32, 3, 8, 8).astype(np.float32)
//   x = cudalearn.from_numpy(x_np)
//
// The numpy array must be:
//   - dtype float32 (our library uses single precision throughout)
//   - contiguous in memory (C-order, which is numpy's default)
//   - 1D to 4D
// =============================================================================

GradTensor* from_numpy(py::array_t<float> arr, bool requires_grad = false) {
    // Request a contiguous buffer from numpy.
    // py::array_t<float>::request() returns a buffer_info struct with:
    //   - ptr:    pointer to the raw data
    //   - shape:  vector of dimension sizes
    //   - strides: vector of byte strides
    //   - ndim:   number of dimensions
    py::buffer_info buf = arr.request();

    // Validate: we support 1D to 4D tensors
    if (buf.ndim < 1 || buf.ndim > 4) {
        throw std::runtime_error("from_numpy: array must be 1D to 4D");
    }

    // Extract dimensions, padding with 1 for missing dims.
    // Our GradTensor always has 4 dims internally: [d0, d1, d2, d3].
    // A 2D array [32, 784] becomes GradTensor(32, 784, 1, 1).
    int dims[4] = {1, 1, 1, 1};
    for (int i = 0; i < buf.ndim; i++) {
        dims[i] = (int)buf.shape[i];
    }

    // Create the GradTensor (allocates GPU memory, zeros it out)
    GradTensor* t = new GradTensor(dims[0], dims[1], dims[2], dims[3], requires_grad);

    // Copy data from CPU (numpy) to GPU (GradTensor)
    // buf.ptr is the raw pointer to the numpy array's data buffer
    cudaMemcpy(t->data, buf.ptr, t->size * sizeof(float), cudaMemcpyHostToDevice);

    return t;
}


// =============================================================================
// Utility: Convert a GradTensor to a NumPy array
// =============================================================================
// This copies data FROM the GPU (GradTensor) TO a new numpy array on the CPU.
//
// Usage from Python:
//   result_np = tensor.to_numpy()
//   print(result_np.shape)  # e.g., (32, 10)
//
// The returned numpy array owns its own memory (it is a copy, not a view).
// Modifying the numpy array does NOT affect the GradTensor, and vice versa.
// =============================================================================

py::array_t<float> to_numpy(GradTensor* t) {
    // Build the shape vector based on how many meaningful dimensions exist.
    // GradTensor stores ndim (1-4) telling us which dims are "real".
    std::vector<ssize_t> shape;
    for (int i = 0; i < t->ndim; i++) {
        shape.push_back(t->dims[i]);
    }

    // Allocate a numpy array with the correct shape
    py::array_t<float> result(shape);

    // Get a mutable pointer to the numpy array's data buffer
    py::buffer_info buf = result.request();

    // Copy data from GPU to CPU
    // This is a synchronous copy -- it blocks until the GPU finishes any
    // pending work on this memory region.
    cudaMemcpy(buf.ptr, t->data, t->size * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}


// =============================================================================
// Utility: Convert gradient data to a NumPy array
// =============================================================================
// Same as to_numpy() but reads from the grad field instead of data.
// Useful for debugging gradient values.
//
// Usage from Python:
//   grad_np = tensor.grad_to_numpy()
// =============================================================================

py::array_t<float> grad_to_numpy(GradTensor* t) {
    if (!t->grad) {
        throw std::runtime_error("grad_to_numpy: tensor has no gradient");
    }

    std::vector<ssize_t> shape;
    for (int i = 0; i < t->ndim; i++) {
        shape.push_back(t->dims[i]);
    }

    py::array_t<float> result(shape);
    py::buffer_info buf = result.request();
    cudaMemcpy(buf.ptr, t->grad, t->size * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}


// =============================================================================
// Utility: Get scalar value from a 1-element GradTensor (like PyTorch .item())
// =============================================================================
// This is essential for reading loss values in the training loop:
//   loss_val = loss.item()
//   print(f"Loss: {loss_val:.4f}")
// =============================================================================

float tensor_item(GradTensor* t) {
    if (t->size != 1) {
        throw std::runtime_error("item(): tensor must be a scalar (size 1)");
    }
    float val;
    cudaMemcpy(&val, t->data, sizeof(float), cudaMemcpyDeviceToHost);
    return val;
}


// =============================================================================
// Utility: String representation for GradTensor
// =============================================================================
// Called when Python does print(tensor) or repr(tensor).
// Shows shape, size, and whether gradients are tracked.
// =============================================================================

std::string tensor_repr(GradTensor* t) {
    // Build a shape string like "(32, 3, 8, 8)"
    std::string shape_str = "(";
    for (int i = 0; i < t->ndim; i++) {
        if (i > 0) shape_str += ", ";
        shape_str += std::to_string(t->dims[i]);
    }
    shape_str += ")";

    return "GradTensor(shape=" + shape_str +
           ", size=" + std::to_string(t->size) +
           ", requires_grad=" + (t->requires_grad ? "True" : "False") + ")";
}


// =============================================================================
// Utility: Create random tensors (like torch.randn, torch.zeros, torch.ones)
// =============================================================================
// These factory functions create GradTensors filled with specific values.
// They mirror PyTorch's tensor creation functions.
// =============================================================================

// Fill a GPU array with a constant value
__global__ void fill_kernel(float* data, float value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// Fill with random normal values using a simple LCG + Box-Muller transform
// (This is a simple approach -- cuRAND would be better for production)
__global__ void randn_kernel(float* data, int size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Linear congruential generator for pseudo-random numbers
    unsigned long long state = seed + (unsigned long long)idx * 6364136223846793005ULL;
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u1 = (float)(state >> 33) / (float)(1ULL << 31);
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    float u2 = (float)(state >> 33) / (float)(1ULL << 31);

    // Clamp to avoid log(0)
    if (u1 < 1e-7f) u1 = 1e-7f;

    // Box-Muller transform: convert uniform [0,1] to normal N(0,1)
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    data[idx] = z;
}

// randn: create a tensor filled with random normal values N(0,1)
GradTensor* make_randn(int d0, int d1 = 1, int d2 = 1, int d3 = 1) {
    GradTensor* t = new GradTensor(d0, d1, d2, d3, false);
    int blocks = (t->size + 255) / 256;

    // Use a time-based seed for different random values each call
    unsigned long long seed = (unsigned long long)clock() * 137 + 42;
    randn_kernel<<<blocks, 256>>>(t->data, t->size, seed);
    cudaDeviceSynchronize();
    return t;
}

// zeros: create a tensor filled with zeros (GradTensor already does this)
GradTensor* make_zeros(int d0, int d1 = 1, int d2 = 1, int d3 = 1) {
    // GradTensor constructor calls cudaMemset(data, 0, ...) so this is already zero
    return new GradTensor(d0, d1, d2, d3, false);
}

// ones: create a tensor filled with ones
GradTensor* make_ones(int d0, int d1 = 1, int d2 = 1, int d3 = 1) {
    GradTensor* t = new GradTensor(d0, d1, d2, d3, false);
    int blocks = (t->size + 255) / 256;
    fill_kernel<<<blocks, 256>>>(t->data, 1.0f, t->size);
    cudaDeviceSynchronize();
    return t;
}


// =============================================================================
// Utility: Create GPU integer labels from a numpy array
// =============================================================================
// CrossEntropyLoss.forward() expects int* labels on the GPU.
// This helper copies a numpy int32 array to GPU memory.
// =============================================================================

int* labels_to_gpu(py::array_t<int> arr) {
    py::buffer_info buf = arr.request();
    int count = 1;
    for (auto s : buf.shape) count *= (int)s;

    int* gpu_labels = nullptr;
    cudaMalloc(&gpu_labels, count * sizeof(int));
    cudaMemcpy(gpu_labels, buf.ptr, count * sizeof(int), cudaMemcpyHostToDevice);
    return gpu_labels;
}


// =============================================================================
// CrossEntropyLoss Python wrapper
// =============================================================================
// The C++ CrossEntropyLoss::forward() takes (GradTensor*, int*) where the
// int* is a GPU pointer. Python users will pass a numpy array instead.
// This wrapper handles the numpy -> GPU conversion automatically.
// =============================================================================

GradTensor* ce_forward_wrapper(CrossEntropyLoss& ce, GradTensor* logits,
                                py::array_t<int> labels_np) {
    py::buffer_info buf = labels_np.request();
    int N = (int)buf.shape[0];

    // Copy labels from numpy (CPU) to GPU
    int* gpu_labels = nullptr;
    cudaMalloc(&gpu_labels, N * sizeof(int));
    cudaMemcpy(gpu_labels, buf.ptr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Call the C++ forward method with the GPU labels
    GradTensor* loss = ce.forward(logits, gpu_labels);

    // Note: we do NOT free gpu_labels here because CrossEntropyLoss::forward()
    // copies them internally to labels_buf_. We can safely free our copy.
    cudaFree(gpu_labels);

    return loss;
}


// =============================================================================
// MSELoss Python wrapper
// =============================================================================
// Similar to CrossEntropyLoss wrapper: converts numpy targets to GPU memory.
// =============================================================================

GradTensor* mse_forward_wrapper(MSELoss& mse, GradTensor* predictions,
                                 py::array_t<float> targets_np) {
    py::buffer_info buf = targets_np.request();
    int N = (int)buf.shape[0];

    // Copy targets from numpy (CPU) to GPU
    float* gpu_targets = nullptr;
    cudaMalloc(&gpu_targets, N * sizeof(float));
    cudaMemcpy(gpu_targets, buf.ptr, N * sizeof(float), cudaMemcpyHostToDevice);

    GradTensor* loss = mse.forward(predictions, gpu_targets);

    cudaFree(gpu_targets);
    return loss;
}


// #############################################################################
//
// PYBIND11_MODULE: The main binding definition
//
// #############################################################################
//
// PYBIND11_MODULE(name, variable) is a macro that:
//   1. Creates a Python module with the given name ("cudalearn")
//   2. Gives you a py::module_ variable (m) to register classes/functions on
//   3. Generates the correct entry point symbol for Python's import machinery
//
// Everything inside this block defines what Python sees when it does:
//   import cudalearn
//
// The module name ("cudalearn") MUST match the filename of the compiled .so:
//   cudalearn.cpython-310-x86_64-linux-gnu.so
//
// #############################################################################

PYBIND11_MODULE(cudalearn, m) {

    // Module docstring -- shown by help(cudalearn) in Python
    m.doc() = "cudalearn: A CUDA deep learning library with a PyTorch-like API.\n\n"
              "This module provides GPU-accelerated tensor operations, neural network\n"
              "layers, optimizers, loss functions, and data loading -- all backed by\n"
              "custom CUDA kernels.\n\n"
              "Example:\n"
              "    import cudalearn\n"
              "    model = cudalearn.Sequential()\n"
              "    model.add('fc', cudalearn.Linear(784, 10))\n"
              "    optimizer = cudalearn.Adam(model.parameters(), lr=0.001)\n";


    // =========================================================================
    // GradTensor Bindings
    // =========================================================================
    // GradTensor is the fundamental data type -- a GPU tensor with optional
    // gradient tracking. This is analogous to torch.Tensor.
    //
    // py::class_<GradTensor> creates a Python type called "GradTensor" that
    // wraps the C++ GradTensor struct. The second template argument (no base
    // class here) tells pybind11 this is a standalone class.
    // =========================================================================

    py::class_<GradTensor>(m, "GradTensor",
        "A GPU tensor with optional gradient tracking.\n\n"
        "This is the fundamental data type, analogous to torch.Tensor.\n"
        "Data lives on the GPU; use to_numpy() to copy to CPU.")

        // --- Constructor ---
        // py::init<Args...> binds the C++ constructor GradTensor(int, int, int, int, bool)
        // py::arg("name") = default provides named arguments with defaults in Python
        //
        // Python usage:
        //   t = cudalearn.GradTensor(32, 3, 8, 8)               # 4D tensor
        //   t = cudalearn.GradTensor(100, requires_grad=True)    # 1D with grad
        .def(py::init<int, int, int, int, bool>(),
             py::arg("d0"),
             py::arg("d1") = 1,
             py::arg("d2") = 1,
             py::arg("d3") = 1,
             py::arg("requires_grad") = false,
             "Create a GradTensor with the given dimensions.\n\n"
             "Args:\n"
             "    d0: First dimension (batch size)\n"
             "    d1: Second dimension (channels) [default: 1]\n"
             "    d2: Third dimension (height) [default: 1]\n"
             "    d3: Fourth dimension (width) [default: 1]\n"
             "    requires_grad: Track gradients for this tensor [default: False]")

        // --- Methods ---

        // backward(): trigger reverse-mode autodiff from this tensor
        // This is the equivalent of PyTorch loss.backward()
        .def("backward", &GradTensor::backward,
             "Trigger reverse-mode autodiff from this tensor.\n\n"
             "Seeds gradient to 1.0 (for scalar tensors) and propagates\n"
             "gradients through the computation graph.")

        // zero_grad(): reset gradients to zero
        .def("zero_grad", &GradTensor::zero_grad,
             "Zero out the gradient buffer.")

        // to_numpy(): copy GPU data to a numpy array on CPU
        // This is a free function (not a member), but we bind it as a method
        // using a lambda that calls our utility function.
        .def("to_numpy", [](GradTensor* self) { return to_numpy(self); },
             "Copy tensor data from GPU to a numpy array on CPU.\n\n"
             "Returns:\n"
             "    numpy.ndarray: A float32 array with the tensor's shape.")

        // grad_to_numpy(): copy gradient data to numpy
        .def("grad_to_numpy", [](GradTensor* self) { return grad_to_numpy(self); },
             "Copy gradient data from GPU to a numpy array on CPU.\n\n"
             "Returns:\n"
             "    numpy.ndarray: A float32 array with the gradient values.\n"
             "Raises:\n"
             "    RuntimeError: If the tensor has no gradient.")

        // item(): get scalar value (like PyTorch .item())
        .def("item", [](GradTensor* self) { return tensor_item(self); },
             "Get the scalar value of a 1-element tensor.\n\n"
             "Returns:\n"
             "    float: The tensor's value.\n"
             "Raises:\n"
             "    RuntimeError: If the tensor has more than 1 element.")

        // --- Properties ---
        // def_readonly exposes a C++ member as a read-only Python attribute.
        // def_readwrite allows both reading and writing from Python.

        .def_readonly("size", &GradTensor::size,
                      "Total number of elements in the tensor.")

        .def_readonly("ndim", &GradTensor::ndim,
                      "Number of meaningful dimensions (1-4).")

        .def_readwrite("requires_grad", &GradTensor::requires_grad,
                       "Whether this tensor tracks gradients.")

        // shape property: return dims as a Python tuple
        // We use a lambda because GradTensor::dims is a C array, not a std::vector
        .def_property_readonly("shape", [](GradTensor* self) {
            py::tuple shape(self->ndim);
            for (int i = 0; i < self->ndim; i++) {
                shape[i] = py::int_(self->dims[i]);
            }
            return shape;
        }, "Shape of the tensor as a tuple.")

        // --- Special methods ---

        // __repr__: string representation for print() and repr()
        .def("__repr__", [](GradTensor* self) { return tensor_repr(self); })

        // __len__: allows len(tensor) to return the first dimension
        .def("__len__", [](GradTensor* self) { return self->dims[0]; })
    ;


    // =========================================================================
    // Module Bindings (with Trampoline for virtual forward)
    // =========================================================================
    // Module is the abstract base class for all layers. We bind it with the
    // PyModule trampoline so Python subclasses can override forward().
    //
    // The template arguments <Module, PyModule> tell pybind11:
    //   - Module is the actual C++ class
    //   - PyModule is the trampoline class to use for Python subclassing
    //
    // Without PyModule, Python subclasses of Module would crash when C++
    // code calls forward() on them, because the virtual dispatch would not
    // know to look in the Python object for the implementation.
    // =========================================================================

    py::class_<Module, PyModule>(m, "Module",
        "Abstract base class for all neural network modules.\n\n"
        "Subclass this and implement forward() to create custom layers.\n"
        "Use register_module() and register_parameter() in __init__.")

        .def(py::init<>())

        // forward(): pure virtual -- must be overridden in subclasses
        .def("forward", &Module::forward,
             py::arg("input"),
             py::return_value_policy::take_ownership,
             "Run the forward pass. Must be overridden by subclasses.\n\n"
             "Args:\n"
             "    input: GradTensor input\n"
             "Returns:\n"
             "    GradTensor: output tensor (new, caller owns)")

        // parameters(): collect all learnable tensors recursively
        // return_value_policy::reference_internal means:
        //   - Python gets references (not copies) to the GradTensors
        //   - The Module is kept alive as long as any returned parameter exists
        .def("parameters", &Module::parameters,
             py::return_value_policy::reference_internal,
             "Recursively collect all learnable parameters.\n\n"
             "Returns:\n"
             "    list[GradTensor]: All learnable parameters from this\n"
             "    module and its sub-modules.")

        // train() / set_eval(): switch between training and inference mode
        .def("train", &Module::train,
             py::arg("mode") = true,
             "Set training mode (affects BatchNorm, Dropout, etc.).")

        .def("set_eval", &Module::set_eval,
             "Set evaluation mode (uses running stats for BatchNorm).")

        // print(): display architecture summary
        .def("print", &Module::print,
             py::arg("indent") = 0,
             "Print model architecture with parameter counts.")

        // Properties
        .def_readwrite("name_", &Module::name_,
                       "Human-readable name of this module.")
        .def_readwrite("training_", &Module::training_,
                       "Whether the module is in training mode.")
    ;


    // =========================================================================
    // Layer Bindings: Conv2d
    // =========================================================================
    // py::class_<Conv2d, Module> tells pybind11 that Conv2d inherits from Module.
    // This means:
    //   - Conv2d objects can be used anywhere a Module is expected
    //   - Methods from Module (parameters, train, etc.) are available
    //   - Virtual dispatch works correctly
    // =========================================================================

    py::class_<Conv2d, Module>(m, "Conv2d",
        "2D convolution layer.\n\n"
        "Applies a 2D convolution over an input tensor [N, C_in, H, W].\n"
        "Uses Kaiming (He) initialization for weights.\n\n"
        "Args:\n"
        "    in_channels: Number of input channels\n"
        "    out_channels: Number of output channels (number of filters)\n"
        "    kernel_size: Size of the square convolution kernel\n"
        "    padding: Zero-padding added to both sides [default: 0]\n"
        "    stride: Stride of the convolution [default: 1]\n"
        "    use_bias: Whether to add a bias term [default: True]")

        .def(py::init<int, int, int, int, int, bool>(),
             py::arg("in_channels"),
             py::arg("out_channels"),
             py::arg("kernel_size"),
             py::arg("padding") = 0,
             py::arg("stride") = 1,
             py::arg("use_bias") = true)

        // forward() is inherited from Module but we can also bind it explicitly
        // for a clearer Python API.
        .def("forward", &Conv2d::forward,
             py::arg("input"),
             py::return_value_policy::take_ownership,
             "Apply 2D convolution to the input tensor.\n\n"
             "Args:\n"
             "    input: GradTensor of shape [N, C_in, H, W]\n"
             "Returns:\n"
             "    GradTensor of shape [N, C_out, H_out, W_out]")

        // Expose layer attributes as read-only properties
        .def_readonly("in_channels", &Conv2d::in_channels_)
        .def_readonly("out_channels", &Conv2d::out_channels_)
        .def_readonly("kernel_size", &Conv2d::kernel_size_)
        .def_readonly("padding", &Conv2d::padding_)
        .def_readonly("stride", &Conv2d::stride_)

        .def("__repr__", [](Conv2d& self) { return self.name_; })
    ;


    // =========================================================================
    // Layer Bindings: BatchNorm2d
    // =========================================================================

    py::class_<BatchNorm2d, Module>(m, "BatchNorm2d",
        "Batch Normalization for 2D inputs [N, C, H, W].\n\n"
        "Normalizes each channel across the batch using batch statistics\n"
        "(in training) or running statistics (in evaluation).\n\n"
        "Learnable parameters: gamma (scale) and beta (shift).\n\n"
        "Args:\n"
        "    num_features: Number of channels (C)\n"
        "    eps: Small constant for numerical stability [default: 1e-5]\n"
        "    momentum: Weight for running stat updates [default: 0.1]")

        .def(py::init<int, float, float>(),
             py::arg("num_features"),
             py::arg("eps") = 1e-5f,
             py::arg("momentum") = 0.1f)

        .def("forward", &BatchNorm2d::forward,
             py::arg("input"),
             py::return_value_policy::take_ownership,
             "Apply batch normalization to the input tensor.\n\n"
             "Args:\n"
             "    input: GradTensor of shape [N, C, H, W]\n"
             "Returns:\n"
             "    GradTensor of shape [N, C, H, W] (normalized)")

        .def_readonly("num_features", &BatchNorm2d::num_features_)

        .def("__repr__", [](BatchNorm2d& self) { return self.name_; })
    ;


    // =========================================================================
    // Layer Bindings: ReLU
    // =========================================================================

    py::class_<ReLU, Module>(m, "ReLU",
        "Rectified Linear Unit activation: ReLU(x) = max(0, x).\n\n"
        "This is a stateless layer (no learnable parameters).")

        .def(py::init<>())

        .def("forward", &ReLU::forward,
             py::arg("input"),
             py::return_value_policy::take_ownership,
             "Apply ReLU element-wise.\n\n"
             "Args:\n"
             "    input: GradTensor of any shape\n"
             "Returns:\n"
             "    GradTensor of same shape with negative values zeroed")

        .def("__repr__", [](ReLU& self) { return std::string("ReLU()"); })
    ;


    // =========================================================================
    // Layer Bindings: Linear
    // =========================================================================

    py::class_<Linear, Module>(m, "Linear",
        "Fully connected (dense) layer.\n\n"
        "Computes output = input @ weight.T + bias.\n"
        "Uses Kaiming (He) initialization.\n\n"
        "Args:\n"
        "    in_features: Size of the input feature vector\n"
        "    out_features: Size of the output feature vector\n"
        "    use_bias: Whether to add a bias term [default: True]")

        .def(py::init<int, int, bool>(),
             py::arg("in_features"),
             py::arg("out_features"),
             py::arg("use_bias") = true)

        .def("forward", &Linear::forward,
             py::arg("input"),
             py::return_value_policy::take_ownership,
             "Apply linear transformation.\n\n"
             "Args:\n"
             "    input: GradTensor of shape [N, in_features]\n"
             "Returns:\n"
             "    GradTensor of shape [N, out_features]")

        .def_readonly("in_features", &Linear::in_features_)
        .def_readonly("out_features", &Linear::out_features_)

        .def("__repr__", [](Linear& self) { return self.name_; })
    ;


    // =========================================================================
    // Layer Bindings: GlobalAvgPool2d
    // =========================================================================

    py::class_<GlobalAvgPool2d, Module>(m, "GlobalAvgPool2d",
        "Global Average Pooling for 2D inputs.\n\n"
        "Averages each channel across all spatial positions.\n"
        "Input [N, C, H, W] -> Output [N, C].\n"
        "No learnable parameters.")

        .def(py::init<>())

        .def("forward", &GlobalAvgPool2d::forward,
             py::arg("input"),
             py::return_value_policy::take_ownership,
             "Apply global average pooling.\n\n"
             "Args:\n"
             "    input: GradTensor of shape [N, C, H, W]\n"
             "Returns:\n"
             "    GradTensor of shape [N, C]")

        .def("__repr__", [](GlobalAvgPool2d& self) {
            return std::string("GlobalAvgPool2d()");
        })
    ;


    // =========================================================================
    // Layer Bindings: Sequential
    // =========================================================================
    // Sequential is a container that chains modules. We need special care
    // with memory management:
    //
    // py::keep_alive<1, 3>() on add() means:
    //   "Keep argument 3 (the Module*) alive as long as argument 1 (self) lives"
    //   This prevents Python from garbage-collecting a layer while it is still
    //   registered inside a Sequential.
    //
    // Without keep_alive, this would crash:
    //   model = cudalearn.Sequential()
    //   model.add("fc", cudalearn.Linear(10, 5))  # Linear created and added
    //   # If Python GCs the Linear here (no other reference), forward() crashes!
    // =========================================================================

    py::class_<Sequential, Module>(m, "Sequential",
        "A container that chains modules sequentially.\n\n"
        "Modules are added with add(name, module) and run in order.\n"
        "This is the equivalent of PyTorch nn.Sequential.\n\n"
        "Example:\n"
        "    model = cudalearn.Sequential()\n"
        "    model.add('fc1', cudalearn.Linear(784, 128))\n"
        "    model.add('relu', cudalearn.ReLU())\n"
        "    model.add('fc2', cudalearn.Linear(128, 10))")

        .def(py::init<>())

        // add(): register a named child module
        // py::keep_alive<1, 3>() keeps the 3rd argument (layer) alive as long as
        // the 1st argument (self, the Sequential) is alive.
        // Note: argument indices are 1-based. 1=self, 2=name, 3=layer.
        .def("add", &Sequential::add,
             py::arg("name"),
             py::arg("layer"),
             py::keep_alive<1, 3>(),
             "Add a named layer to the sequence.\n\n"
             "Args:\n"
             "    name: Name for this layer (e.g., 'conv1', 'relu')\n"
             "    layer: The Module to add")

        .def("forward", &Sequential::forward,
             py::arg("input"),
             py::return_value_policy::take_ownership,
             "Run input through all layers in sequence.\n\n"
             "Args:\n"
             "    input: GradTensor input\n"
             "Returns:\n"
             "    GradTensor: Output of the last layer")

        .def("__repr__", [](Sequential& self) {
            std::string s = "Sequential(\n";
            for (auto& m : self.sub_modules_) {
                s += "  (" + m.first + "): " + m.second->name_ + "\n";
            }
            s += ")";
            return s;
        })
    ;


    // =========================================================================
    // Optimizer Bindings: SGD
    // =========================================================================
    // The SGD constructor takes a std::vector<GradTensor*>. pybind11 + stl.h
    // automatically converts a Python list of GradTensors to this type.
    //
    // Usage:
    //   params = model.parameters()
    //   optimizer = cudalearn.SGD(params, lr=0.01, momentum=0.9)
    // =========================================================================

    py::class_<SGD>(m, "SGD",
        "Stochastic Gradient Descent with optional momentum and weight decay.\n\n"
        "Update rule:\n"
        "    v = momentum * v + grad + weight_decay * param\n"
        "    param = param - lr * v\n\n"
        "Args:\n"
        "    params: List of GradTensors from model.parameters()\n"
        "    lr: Learning rate [default: 0.01]\n"
        "    momentum: Momentum coefficient [default: 0.0]\n"
        "    weight_decay: L2 regularization strength [default: 0.0]")

        .def(py::init<std::vector<GradTensor*>, float, float, float>(),
             py::arg("params"),
             py::arg("lr") = 0.01f,
             py::arg("momentum") = 0.0f,
             py::arg("weight_decay") = 0.0f)

        .def("step", &SGD::step,
             "Apply one SGD update to all parameters.\n"
             "Call after loss.backward().")

        .def("zero_grad", &SGD::zero_grad,
             "Reset all parameter gradients to zero.\n"
             "Call before each forward pass.")

        // Expose lr_ as a read-write property so schedulers can modify it
        .def_readwrite("lr", &SGD::lr_,
                       "Current learning rate.")
    ;


    // =========================================================================
    // Optimizer Bindings: Adam
    // =========================================================================

    py::class_<Adam>(m, "Adam",
        "Adam optimizer (Adaptive Moment Estimation).\n\n"
        "Maintains per-parameter first and second moment estimates.\n"
        "Generally converges faster than SGD on many problems.\n\n"
        "Update rule:\n"
        "    m = beta1 * m + (1-beta1) * grad\n"
        "    v = beta2 * v + (1-beta2) * grad^2\n"
        "    param -= lr * m_hat / (sqrt(v_hat) + eps)\n\n"
        "Args:\n"
        "    params: List of GradTensors from model.parameters()\n"
        "    lr: Learning rate [default: 0.001]\n"
        "    beta1: First moment decay rate [default: 0.9]\n"
        "    beta2: Second moment decay rate [default: 0.999]\n"
        "    eps: Numerical stability constant [default: 1e-8]\n"
        "    weight_decay: L2 regularization strength [default: 0.0]")

        .def(py::init<std::vector<GradTensor*>, float, float, float, float, float>(),
             py::arg("params"),
             py::arg("lr") = 0.001f,
             py::arg("beta1") = 0.9f,
             py::arg("beta2") = 0.999f,
             py::arg("eps") = 1e-8f,
             py::arg("weight_decay") = 0.0f)

        .def("step", &Adam::step,
             "Apply one Adam update to all parameters.\n"
             "Call after loss.backward().")

        .def("zero_grad", &Adam::zero_grad,
             "Reset all parameter gradients to zero.\n"
             "Call before each forward pass.")

        .def_readwrite("lr", &Adam::lr_,
                       "Current learning rate.")
    ;


    // =========================================================================
    // Scheduler Bindings: CosineAnnealingLR
    // =========================================================================
    // CosineAnnealingLR takes a float* to the optimizer's lr_ field.
    // From Python, we provide convenience constructors that accept SGD or Adam.
    //
    // Usage:
    //   optimizer = cudalearn.Adam(params, lr=0.001)
    //   scheduler = cudalearn.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    //   scheduler.step(epoch)
    // =========================================================================

    py::class_<CosineAnnealingLR>(m, "CosineAnnealingLR",
        "Cosine annealing learning rate scheduler.\n\n"
        "Decays the learning rate following a cosine curve:\n"
        "    lr = eta_min + 0.5*(lr_max - eta_min)*(1 + cos(pi*epoch/T_max))\n\n"
        "Args:\n"
        "    optimizer: SGD or Adam optimizer\n"
        "    T_max: Number of epochs for one half-cycle\n"
        "    eta_min: Minimum learning rate [default: 0.0]")

        // Constructor that takes an SGD optimizer
        // We pass &optimizer.lr_ to give the scheduler direct access to
        // modify the optimizer's learning rate.
        .def(py::init([](SGD& opt, int T_max, float eta_min) {
            return new CosineAnnealingLR(&opt.lr_, T_max, eta_min);
        }),
             py::arg("optimizer"),
             py::arg("T_max"),
             py::arg("eta_min") = 0.0f,
             // keep_alive<1, 2>: keep optimizer alive as long as scheduler exists
             py::keep_alive<1, 2>())

        // Constructor that takes an Adam optimizer (overload)
        .def(py::init([](Adam& opt, int T_max, float eta_min) {
            return new CosineAnnealingLR(&opt.lr_, T_max, eta_min);
        }),
             py::arg("optimizer"),
             py::arg("T_max"),
             py::arg("eta_min") = 0.0f,
             py::keep_alive<1, 2>())

        .def("step", &CosineAnnealingLR::step,
             py::arg("epoch"),
             "Update the learning rate based on the current epoch.\n\n"
             "Args:\n"
             "    epoch: Current epoch number (0-indexed)")
    ;


    // =========================================================================
    // Loss Function Bindings: CrossEntropyLoss
    // =========================================================================
    // We wrap forward() to accept numpy labels instead of raw GPU pointers.
    // This makes the Python API much cleaner:
    //
    //   criterion = cudalearn.CrossEntropyLoss()
    //   labels = np.array([3, 1, 0, 7], dtype=np.int32)
    //   loss = criterion.forward(logits, labels)
    //   loss.backward()
    // =========================================================================

    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss",
        "Cross-entropy loss (softmax + negative log-likelihood).\n\n"
        "Input: raw logits [N, C] (NOT softmax probabilities).\n"
        "Target: integer class labels [N] as a numpy int32 array.\n\n"
        "The loss is:\n"
        "    loss = -(1/N) * sum_n log(softmax(logits[n])[labels[n]])\n\n"
        "Backward computes:\n"
        "    d_logits = (1/N) * (softmax - one_hot)")

        .def(py::init<>())

        // We use the wrapper function instead of the raw C++ method
        // because C++ forward() takes int* (GPU pointer), but Python
        // users will pass numpy arrays.
        .def("forward", &ce_forward_wrapper,
             py::arg("logits"),
             py::arg("labels"),
             py::return_value_policy::take_ownership,
             "Compute cross-entropy loss.\n\n"
             "Args:\n"
             "    logits: GradTensor of shape [N, C] (raw model outputs)\n"
             "    labels: numpy int32 array of shape [N] (class indices 0 to C-1)\n"
             "Returns:\n"
             "    GradTensor: Scalar loss value")
    ;


    // =========================================================================
    // Loss Function Bindings: MSELoss
    // =========================================================================

    py::class_<MSELoss>(m, "MSELoss",
        "Mean Squared Error loss for regression.\n\n"
        "loss = (1/N) * sum_n (predictions[n] - targets[n])^2\n\n"
        "Backward computes:\n"
        "    d_pred = (2/N) * (predictions - targets)")

        .def(py::init<>())

        .def("forward", &mse_forward_wrapper,
             py::arg("predictions"),
             py::arg("targets"),
             py::return_value_policy::take_ownership,
             "Compute MSE loss.\n\n"
             "Args:\n"
             "    predictions: GradTensor of shape [N] (model outputs)\n"
             "    targets: numpy float32 array of shape [N] (true values)\n"
             "Returns:\n"
             "    GradTensor: Scalar loss value")
    ;


    // =========================================================================
    // DataLoader Bindings
    // =========================================================================
    // The C++ DataLoader takes raw float* and int* pointers.
    // We wrap it to accept numpy arrays instead.
    //
    // Usage:
    //   loader = cudalearn.DataLoader(x_np, y_np, batch_size=32, shuffle=True)
    //   while loader.has_next():
    //       x, y_gpu_ptr = loader.next_batch()
    //       ...
    //   loader.reset()
    //
    // Note: DataLoader is less commonly needed in Python because numpy/PyTorch
    // already have good data loading. But we bind it for completeness.
    // =========================================================================

    py::class_<DataLoader>(m, "DataLoader",
        "Mini-batch data loader with shuffling.\n\n"
        "Feeds data to the model in mini-batches, with optional shuffling\n"
        "and pinned memory for fast CPU-to-GPU transfers.\n\n"
        "Note: For Python workflows, you may prefer to use numpy directly\n"
        "with cudalearn.from_numpy() to create batches.")

        // Constructor: we wrap to accept numpy arrays and store them
        .def(py::init([](py::array_t<float> data_np, py::array_t<int> labels_np,
                         int batch_size, bool shuffle) {
            py::buffer_info data_buf = data_np.request();
            py::buffer_info labels_buf = labels_np.request();

            int num_samples = (int)data_buf.shape[0];
            int sample_size = 1;
            for (int i = 1; i < data_buf.ndim; i++) {
                sample_size *= (int)data_buf.shape[i];
            }

            // The DataLoader does NOT own the data, so we must ensure
            // the numpy arrays stay alive. We use raw pointers here.
            float* data_ptr = static_cast<float*>(data_buf.ptr);
            int* labels_ptr = static_cast<int*>(labels_buf.ptr);

            return new DataLoader(data_ptr, labels_ptr, num_samples,
                                  sample_size, batch_size, shuffle);
        }),
             py::arg("data"),
             py::arg("labels"),
             py::arg("batch_size") = 32,
             py::arg("shuffle") = true,
             // Keep numpy arrays alive as long as DataLoader exists
             py::keep_alive<1, 2>(),   // Keep data alive
             py::keep_alive<1, 3>())   // Keep labels alive

        .def("has_next", &DataLoader::has_next,
             "Check if there are more batches in this epoch.")

        .def("reset", &DataLoader::reset,
             "Reset for a new epoch (optionally reshuffles).")

        .def("num_batches", &DataLoader::num_batches,
             "Total number of complete batches in the dataset.")
    ;


    // =========================================================================
    // Factory Functions: randn, zeros, ones
    // =========================================================================
    // These create GradTensors filled with specific values, mirroring
    // PyTorch torch.randn(), torch.zeros(), torch.ones().
    //
    // Usage:
    //   x = cudalearn.randn(32, 3, 8, 8)   # Random normal
    //   z = cudalearn.zeros(10)              # All zeros
    //   o = cudalearn.ones(5, 5)             # All ones
    // =========================================================================

    m.def("randn", &make_randn,
          py::arg("d0"),
          py::arg("d1") = 1,
          py::arg("d2") = 1,
          py::arg("d3") = 1,
          py::return_value_policy::take_ownership,
          "Create a tensor filled with random normal values N(0,1).\n\n"
          "Args:\n"
          "    d0, d1, d2, d3: Tensor dimensions (unused dims default to 1)\n"
          "Returns:\n"
          "    GradTensor: Random tensor on GPU");

    m.def("zeros", &make_zeros,
          py::arg("d0"),
          py::arg("d1") = 1,
          py::arg("d2") = 1,
          py::arg("d3") = 1,
          py::return_value_policy::take_ownership,
          "Create a tensor filled with zeros.\n\n"
          "Args:\n"
          "    d0, d1, d2, d3: Tensor dimensions\n"
          "Returns:\n"
          "    GradTensor: Zero tensor on GPU");

    m.def("ones", &make_ones,
          py::arg("d0"),
          py::arg("d1") = 1,
          py::arg("d2") = 1,
          py::arg("d3") = 1,
          py::return_value_policy::take_ownership,
          "Create a tensor filled with ones.\n\n"
          "Args:\n"
          "    d0, d1, d2, d3: Tensor dimensions\n"
          "Returns:\n"
          "    GradTensor: Ones tensor on GPU");

    // from_numpy: convert a numpy array to a GradTensor (CPU -> GPU copy)
    m.def("from_numpy", &from_numpy,
          py::arg("arr"),
          py::arg("requires_grad") = false,
          py::return_value_policy::take_ownership,
          "Create a GradTensor from a numpy float32 array.\n\n"
          "Copies data from CPU (numpy) to GPU (GradTensor).\n\n"
          "Args:\n"
          "    arr: numpy.ndarray of dtype float32 (1D to 4D)\n"
          "    requires_grad: Whether to track gradients [default: False]\n"
          "Returns:\n"
          "    GradTensor: GPU tensor with the same shape and data");

}  // end PYBIND11_MODULE
