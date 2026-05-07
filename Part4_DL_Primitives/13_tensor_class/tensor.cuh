// ===========================================================================
// Chapter 13: tensor.cuh -- The Tensor Class Header
// ===========================================================================
// This is the fundamental data structure of our "cudalearn" library.
// Every neural network operation will consume and produce Tensor objects.
//
// Design philosophy:
//   1. The Tensor OBJECT always lives on the CPU (it's a C++ class with
//      vectors, shared_ptr, etc.). Only the DATA BUFFER it points to may
//      live on the GPU.
//
//   2. We use shared_ptr with a custom deleter for reference counting.
//      When multiple tensors share the same buffer (via reshape/view),
//      the memory is freed only when the last tensor goes out of scope.
//
//   3. Shape operations (reshape, view, transpose) are O(1) -- they
//      change metadata without touching the data buffer.
//
//   4. Device transfers (to_gpu, to_cpu) create NEW tensors with
//      fresh allocations on the target device. They are explicit and
//      eager (no lazy evaluation).
//
//   5. Template parameter T is float for now but the design supports
//      future extension to double, half, int, etc.
//
// Comparison with PyTorch:
//   PyTorch: Tensor -> TensorImpl -> Storage (separate classes)
//   cudalearn: Tensor<T> with shared_ptr<T> (all in one, simpler)
//
// ===========================================================================

#ifndef CUDALEARN_TENSOR_CUH
#define CUDALEARN_TENSOR_CUH

#include <vector>
#include <memory>
#include <cassert>
#include <iostream>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <initializer_list>
#include <cstring>

#include <cuda_runtime.h>
#include <curand.h>

// ===========================================================================
// CUDA Error Checking Macro
// ===========================================================================
// We wrap every CUDA call in this macro. In production code you might use
// exceptions; for learning, assert + stderr is clearer.
// ===========================================================================

#define CUDA_CHECK(call)                                                     \
    do {                                                                      \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// cuRAND error checking (cuRAND uses its own error type)
#define CURAND_CHECK(call)                                                   \
    do {                                                                      \
        curandStatus_t status = (call);                                      \
        if (status != CURAND_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuRAND error at %s:%d -- status %d\n",         \
                    __FILE__, __LINE__, (int)status);                        \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// ===========================================================================
// Device Enum
// ===========================================================================
// Tracks whether the data buffer lives in CPU (host) or GPU (device) memory.
// This is essential because we must use the correct free function:
//   CPU: delete[]
//   GPU: cudaFree
// ===========================================================================

enum class Device {
    CPU,
    GPU
};

// Helper: convert Device to string for printing
inline const char* device_to_string(Device d) {
    return (d == Device::CPU) ? "CPU" : "GPU";
}

// ===========================================================================
// Tensor<T> Class
// ===========================================================================
// The core data structure. Holds:
//   - data_    : shared_ptr<T> pointing to a flat buffer (CPU or GPU)
//   - shape_   : vector of dimension sizes, e.g., {2, 3, 4}
//   - strides_ : vector of strides, e.g., {12, 4, 1} for contiguous
//   - size_    : total number of elements (product of shape)
//   - device_  : CPU or GPU
//   - offset_  : starting offset into the data buffer (for views/slices)
//
// The shared_ptr uses a custom deleter that calls delete[] for CPU data
// and cudaFree for GPU data. This gives us automatic reference counting:
// multiple tensors can share the same buffer, and it's freed when the
// last reference dies.
// ===========================================================================

template <typename T>
class Tensor {
public:
    // -----------------------------------------------------------------------
    // Data members
    // -----------------------------------------------------------------------

    // The actual data buffer. shared_ptr gives us reference counting so
    // multiple tensors (e.g., a tensor and its reshaped view) can share
    // the same allocation. The custom deleter handles CPU vs GPU cleanup.
    std::shared_ptr<T> data_;

    // Shape: the size of each dimension. A {3, 4} tensor has 3 rows, 4 cols.
    std::vector<int> shape_;

    // Strides: how many elements to skip in the flat buffer to advance by 1
    // in each dimension. For a contiguous {3, 4} tensor, strides = {4, 1}.
    // For a transposed version, strides might be {1, 4} (non-contiguous).
    std::vector<int> strides_;

    // Total number of elements in the tensor (product of all dimensions).
    int size_;

    // Where the data lives: CPU (host memory) or GPU (device memory).
    Device device_;

    // Offset into the data buffer. Usually 0, but can be nonzero for
    // views/slices that start partway into a shared buffer.
    int offset_;

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    // Default constructor: creates an empty tensor (size 0, CPU, no data)
    Tensor()
        : data_(nullptr), shape_(), strides_(), size_(0),
          device_(Device::CPU), offset_(0) {}

    // -------------------------------------------------------------------
    // Constructor: from shape and device
    // -------------------------------------------------------------------
    // Allocates a zero-initialized buffer of the appropriate size on the
    // specified device. This is the most common constructor.
    //
    // Example: Tensor<float>({3, 4}, Device::GPU)
    //   -> allocates 12 floats on the GPU, all zeros
    // -------------------------------------------------------------------
    Tensor(const std::vector<int>& shape, Device device = Device::CPU);

    // -------------------------------------------------------------------
    // Constructor: from shape + existing host data + device
    // -------------------------------------------------------------------
    // Copies the provided data into a new buffer. If device is GPU, the
    // data is first copied to a host buffer and then transferred to GPU.
    //
    // The input data pointer must point to at least size_ elements.
    //
    // Example: Tensor<float>({2, 3}, my_array, Device::CPU)
    // -------------------------------------------------------------------
    Tensor(const std::vector<int>& shape, const T* data,
           Device device = Device::CPU);

    // -------------------------------------------------------------------
    // Static factory: from std::vector
    // -------------------------------------------------------------------
    // Creates a 1D tensor from a vector. Convenient for testing.
    // This is a static method rather than a constructor to avoid ambiguity
    // when T=int or T=float (braced-init-lists would match both the
    // shape constructor and the data constructor).
    //
    // Example: Tensor<float>::from_vec({1.0f, 2.0f, 3.0f})  -> shape {3}
    // -------------------------------------------------------------------
    static Tensor<T> from_vec(const std::vector<T>& data,
                              Device device = Device::CPU);

    // -------------------------------------------------------------------
    // Private constructor: from existing shared_ptr (for views/reshapes)
    // -------------------------------------------------------------------
    // Used internally when creating views that share the same data buffer.
    // Not exposed publicly -- use reshape() and view() instead.
    // -------------------------------------------------------------------
    Tensor(std::shared_ptr<T> data, const std::vector<int>& shape,
           const std::vector<int>& strides, int size, Device device,
           int offset = 0);

    // -----------------------------------------------------------------------
    // Static Factory Methods
    // -----------------------------------------------------------------------
    // These create tensors with specific initialization patterns.
    // Named constructors are clearer than overloading the constructor.
    // -----------------------------------------------------------------------

    // All zeros: the default for weight initialization placeholders
    static Tensor<T> zeros(const std::vector<int>& shape,
                           Device device = Device::CPU);

    // All ones: useful for bias initialization and testing
    static Tensor<T> ones(const std::vector<int>& shape,
                          Device device = Device::CPU);

    // Normal distribution (mean=0, std=1): standard weight initialization
    // Uses cuRAND on GPU for fast generation, or std::normal_distribution
    // on CPU.
    static Tensor<T> randn(const std::vector<int>& shape,
                           Device device = Device::CPU);

    // Sequential values [0, 1, 2, ..., n-1]: useful for testing/debugging
    static Tensor<T> arange(int n, Device device = Device::CPU);

    // Fill with a constant value
    static Tensor<T> full(const std::vector<int>& shape, T value,
                          Device device = Device::CPU);

    // -----------------------------------------------------------------------
    // Device Transfer
    // -----------------------------------------------------------------------
    // These create a NEW tensor on the target device by copying data.
    // The original tensor is unchanged.
    //
    // Usage:
    //   auto gpu_tensor = cpu_tensor.to_gpu();
    //   auto cpu_tensor = gpu_tensor.to_cpu();
    //   auto target_tensor = tensor.to(Device::GPU);
    // -----------------------------------------------------------------------

    Tensor<T> to_gpu() const;
    Tensor<T> to_cpu() const;
    Tensor<T> to(Device target) const;

    // -----------------------------------------------------------------------
    // Shape Operations (O(1), metadata only, no data copy)
    // -----------------------------------------------------------------------

    // -------------------------------------------------------------------
    // reshape: Change the shape (and recompute strides) without copying.
    // -------------------------------------------------------------------
    // Requirements:
    //   - The tensor must be contiguous (otherwise we'd need to copy)
    //   - The new shape must have the same total number of elements
    //
    // One dimension can be -1, meaning "infer from the others":
    //   {3, 4}.reshape({-1, 2}) -> {6, 2}  (because 12/2 = 6)
    //
    // Returns a NEW Tensor sharing the same data buffer (ref count +1).
    // -------------------------------------------------------------------
    Tensor<T> reshape(const std::vector<int>& new_shape) const;

    // view: Alias for reshape (PyTorch uses both names)
    Tensor<T> view(const std::vector<int>& new_shape) const;

    // -------------------------------------------------------------------
    // transpose: Swap the last two dimensions' strides.
    // -------------------------------------------------------------------
    // For a 2D matrix, this is the standard matrix transpose.
    // For higher-dimensional tensors, it swaps the last two dims
    // (like PyTorch's .T for 2D or .transpose(-2, -1) for nD).
    //
    // This is LAZY: no data is copied. The tensor becomes non-contiguous.
    // Call contiguous() afterward if you need contiguous memory.
    // -------------------------------------------------------------------
    Tensor<T> transpose() const;

    // -------------------------------------------------------------------
    // is_contiguous: Check if strides match the standard row-major layout.
    // -------------------------------------------------------------------
    // A tensor is contiguous if stride[i] = product(shape[i+1:]).
    // Transposed tensors are typically not contiguous.
    // -------------------------------------------------------------------
    bool is_contiguous() const;

    // -------------------------------------------------------------------
    // contiguous: Return a contiguous copy if non-contiguous, or self.
    // -------------------------------------------------------------------
    // If already contiguous, returns *this (shared data, no copy).
    // If non-contiguous, allocates a new buffer and copies data in
    // row-major order.
    // -------------------------------------------------------------------
    Tensor<T> contiguous() const;

    // Number of dimensions
    int ndim() const { return static_cast<int>(shape_.size()); }

    // -----------------------------------------------------------------------
    // Element Access
    // -----------------------------------------------------------------------
    // These only work for CPU tensors. Accessing GPU memory element-by-
    // element would be catastrophically slow (each access = a kernel launch
    // or memcpy). For GPU tensors, use to_cpu() first.
    // -----------------------------------------------------------------------

    // Variadic element access: tensor(i, j, k) for a 3D tensor
    // Computes flat index using strides: offset + i*stride[0] + j*stride[1] + ...
    template <typename... Indices>
    T& operator()(Indices... indices);

    template <typename... Indices>
    const T& operator()(Indices... indices) const;

    // Access via vector of indices (when dimensionality isn't known at compile time)
    T& at(const std::vector<int>& indices);
    const T& at(const std::vector<int>& indices) const;

    // Raw data pointer (use with caution)
    T* data_ptr() { return data_.get() + offset_; }
    const T* data_ptr() const { return data_.get() + offset_; }

    // -----------------------------------------------------------------------
    // Display
    // -----------------------------------------------------------------------
    // Prints tensor contents. If on GPU, temporarily copies to CPU.
    // Shows shape, device, and values in a readable format.
    // -----------------------------------------------------------------------
    void print(const std::string& name = "") const;

    // -----------------------------------------------------------------------
    // Utility
    // -----------------------------------------------------------------------

    // Compute the flat index from multi-dimensional indices using strides
    int flat_index(const std::vector<int>& indices) const;

    // Compute contiguous strides from a shape
    static std::vector<int> compute_strides(const std::vector<int>& shape);

    // Compute total size from shape
    static int compute_size(const std::vector<int>& shape);
};

// ===========================================================================
// We include the implementation file here because Tensor is a template class.
// Template definitions must be visible at the point of instantiation, so
// they either go in the header or in a file included by the header.
//
// Alternative: explicit template instantiation in a .cu file. But including
// the .cu here is simpler and standard practice for small template classes.
// ===========================================================================

#include "tensor.cu"

#endif // CUDALEARN_TENSOR_CUH
