// ===========================================================================
// Chapter 13: tensor.cu -- Tensor Class Implementation
// ===========================================================================
// This file implements the Tensor<T> class declared in tensor.cuh.
//
// Because Tensor is a template class, this file is #included at the bottom
// of tensor.cuh (not compiled separately). This is the standard approach
// for template implementations in C++.
//
// Key implementation details:
//   - Memory allocation uses new[] for CPU, cudaMalloc for GPU
//   - shared_ptr custom deleters handle automatic cleanup
//   - cuRAND is used for GPU random number generation
//   - All CUDA calls are wrapped in CUDA_CHECK for error detection
// ===========================================================================

#ifndef CUDALEARN_TENSOR_CU
#define CUDALEARN_TENSOR_CU

#include <random>
#include <iomanip>
#include <algorithm>
#include <sstream>

// ===========================================================================
// CUDA Kernel: fill a buffer with a constant value
// ===========================================================================
// Grid-stride loop pattern: each thread handles multiple elements if the
// array is larger than the grid. This is the standard CUDA kernel pattern
// we've been using throughout the course.
// ===========================================================================

template <typename T>
__global__ void kernel_fill(T* data, T value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop: handles arrays larger than the grid
    for (int i = idx; i < n; i += stride) {
        data[i] = value;
    }
}

// ===========================================================================
// CUDA Kernel: copy non-contiguous data to contiguous layout
// ===========================================================================
// When a tensor is non-contiguous (e.g., after transpose), we need to
// rearrange the data into standard row-major order. This kernel computes
// the source index using the original strides and writes to sequential
// positions in the destination.
//
// For a 2D transpose example:
//   src has strides {1, 4} (transposed 3x4 -> 4x3)
//   dst has strides {3, 1} (contiguous 4x3)
//   For each dst index i, we compute the multi-dim indices, then use
//   src strides to find where that element actually lives.
// ===========================================================================

template <typename T>
__global__ void kernel_make_contiguous(
    const T* __restrict__ src,   // source data (non-contiguous layout)
    T* __restrict__ dst,         // destination data (contiguous layout)
    const int* __restrict__ shape,       // shape of the tensor
    const int* __restrict__ src_strides, // strides of the source
    int ndim,                    // number of dimensions
    int n                        // total number of elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        // Convert flat contiguous index 'i' to multi-dimensional indices
        // Then use src_strides to find the actual position in src
        int src_offset = 0;
        int remaining = i;

        for (int d = ndim - 1; d >= 0; d--) {
            int coord = remaining % shape[d];
            remaining /= shape[d];
            src_offset += coord * src_strides[d];
        }

        dst[i] = src[src_offset];
    }
}

// ===========================================================================
// Static Utility Methods
// ===========================================================================

// ---------------------------------------------------------------------------
// compute_strides: Given a shape, compute the contiguous row-major strides.
// ---------------------------------------------------------------------------
// For shape {2, 3, 4}:
//   stride[2] = 1           (last dim always has stride 1)
//   stride[1] = 4           (shape[2])
//   stride[0] = 12          (shape[1] * shape[2] = 3 * 4)
//   result: {12, 4, 1}
// ---------------------------------------------------------------------------
template <typename T>
std::vector<int> Tensor<T>::compute_strides(const std::vector<int>& shape) {
    int ndim = static_cast<int>(shape.size());
    if (ndim == 0) return {};

    std::vector<int> strides(ndim);
    strides[ndim - 1] = 1;

    // Walk backwards: each stride = stride[i+1] * shape[i+1]
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    return strides;
}

// ---------------------------------------------------------------------------
// compute_size: Product of all dimensions.
// ---------------------------------------------------------------------------
// For shape {2, 3, 4}: size = 2 * 3 * 4 = 24
// For empty shape {}: size = 1 (scalar)
// ---------------------------------------------------------------------------
template <typename T>
int Tensor<T>::compute_size(const std::vector<int>& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<int>());
}

// ===========================================================================
// Constructors
// ===========================================================================

// ---------------------------------------------------------------------------
// Constructor: allocate from shape
// ---------------------------------------------------------------------------
// Creates a zero-initialized tensor on the specified device.
// This is the workhorse constructor -- most factory methods call this.
//
// Memory management strategy:
//   CPU: new T[size]()  -- the () zero-initializes (value initialization)
//   GPU: cudaMalloc + cudaMemset to zero
//
// The shared_ptr gets a custom deleter lambda that knows whether to call
// delete[] (CPU) or cudaFree (GPU). This is critical -- calling the wrong
// free function crashes the program.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T>::Tensor(const std::vector<int>& shape, Device device)
    : shape_(shape),
      strides_(compute_strides(shape)),
      size_(compute_size(shape)),
      device_(device),
      offset_(0)
{
    if (size_ == 0) {
        data_ = nullptr;
        return;
    }

    if (device == Device::CPU) {
        // CPU allocation: new[] with value initialization (zeros)
        T* raw = new T[size_]();
        // Custom deleter: delete[] for CPU memory
        data_ = std::shared_ptr<T>(raw, [](T* p) { delete[] p; });
    } else {
        // GPU allocation: cudaMalloc + cudaMemset
        T* raw = nullptr;
        CUDA_CHECK(cudaMalloc(&raw, size_ * sizeof(T)));
        CUDA_CHECK(cudaMemset(raw, 0, size_ * sizeof(T)));
        // Custom deleter: cudaFree for GPU memory
        // Note: cudaFree is safe to call even during program shutdown
        data_ = std::shared_ptr<T>(raw, [](T* p) { cudaFree(p); });
    }
}

// ---------------------------------------------------------------------------
// Constructor: from shape + host data pointer
// ---------------------------------------------------------------------------
// Copies data from a host pointer into a new tensor. If the target device
// is GPU, we do a host-to-device transfer via cudaMemcpy.
//
// IMPORTANT: The input pointer must point to host (CPU) memory with at
// least size_ elements. We always copy -- we never take ownership of
// external pointers (that would be a dangling pointer disaster).
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T>::Tensor(const std::vector<int>& shape, const T* data, Device device)
    : shape_(shape),
      strides_(compute_strides(shape)),
      size_(compute_size(shape)),
      device_(device),
      offset_(0)
{
    if (size_ == 0) {
        data_ = nullptr;
        return;
    }

    if (device == Device::CPU) {
        // Allocate and copy on CPU
        T* raw = new T[size_];
        std::memcpy(raw, data, size_ * sizeof(T));
        data_ = std::shared_ptr<T>(raw, [](T* p) { delete[] p; });
    } else {
        // Allocate on GPU and copy from host to device
        T* raw = nullptr;
        CUDA_CHECK(cudaMalloc(&raw, size_ * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(raw, data, size_ * sizeof(T),
                              cudaMemcpyHostToDevice));
        data_ = std::shared_ptr<T>(raw, [](T* p) { cudaFree(p); });
    }
}

// ---------------------------------------------------------------------------
// from_vec: Static factory to create 1D tensor from std::vector
// ---------------------------------------------------------------------------
// Convenience factory for tests. Creates a 1D tensor from a vector.
// This is a static method rather than a constructor to avoid ambiguity
// when T matches int (braced-init-lists would match both vector<int>
// shape and vector<T> data constructors).
//
// Example:
//   auto t = Tensor<float>::from_vec({1.0f, 2.0f, 3.0f});  // shape {3}
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::from_vec(const std::vector<T>& data, Device device) {
    return Tensor<T>(std::vector<int>{static_cast<int>(data.size())},
                     data.data(), device);
}

// ---------------------------------------------------------------------------
// Internal constructor: from existing shared_ptr (for views/reshapes)
// ---------------------------------------------------------------------------
// This constructor shares the data buffer with another tensor. The
// shared_ptr reference count increases, so the buffer won't be freed
// until ALL tensors sharing it are destroyed.
//
// This is used by reshape(), view(), transpose() to create new Tensor
// objects that are views into existing data.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T>::Tensor(std::shared_ptr<T> data, const std::vector<int>& shape,
                  const std::vector<int>& strides, int size,
                  Device device, int offset)
    : data_(data), shape_(shape), strides_(strides), size_(size),
      device_(device), offset_(offset)
{}

// ===========================================================================
// Static Factory Methods
// ===========================================================================

// ---------------------------------------------------------------------------
// zeros: All elements initialized to 0.
// ---------------------------------------------------------------------------
// This is just the default constructor behavior, but having an explicit
// factory makes the intent clear (like torch.zeros()).
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::zeros(const std::vector<int>& shape, Device device) {
    // The shape constructor already zero-initializes
    return Tensor<T>(shape, device);
}

// ---------------------------------------------------------------------------
// ones: All elements initialized to 1.
// ---------------------------------------------------------------------------
// For CPU: fill after allocation.
// For GPU: launch a fill kernel.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::ones(const std::vector<int>& shape, Device device) {
    Tensor<T> t(shape, device);
    if (t.size_ == 0) return t;

    if (device == Device::CPU) {
        T* ptr = t.data_ptr();
        std::fill(ptr, ptr + t.size_, static_cast<T>(1));
    } else {
        // Launch fill kernel on GPU
        int threads = 256;
        int blocks = (t.size_ + threads - 1) / threads;
        kernel_fill<<<blocks, threads>>>(t.data_ptr(), static_cast<T>(1),
                                         t.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return t;
}

// ---------------------------------------------------------------------------
// randn: Standard normal distribution (mean=0, std=1).
// ---------------------------------------------------------------------------
// CPU: uses std::normal_distribution with the Mersenne Twister engine.
//      This is deterministic given a fixed seed.
//
// GPU: uses cuRAND's device API (curandGenerateNormal). cuRAND can
//      generate millions of random numbers in parallel on the GPU --
//      much faster than doing it on CPU and transferring.
//
// NOTE: curandGenerateNormal requires an EVEN number of elements.
//       If size is odd, we generate size+1 and ignore the last one.
//       This is a documented cuRAND requirement.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::randn(const std::vector<int>& shape, Device device) {
    Tensor<T> t(shape, device);
    if (t.size_ == 0) return t;

    if (device == Device::CPU) {
        // CPU random generation using C++ <random>
        static std::mt19937 gen(42);  // Fixed seed for reproducibility
        std::normal_distribution<float> dist(0.0f, 1.0f);

        T* ptr = t.data_ptr();
        for (int i = 0; i < t.size_; i++) {
            ptr[i] = static_cast<T>(dist(gen));
        }
    } else {
        // GPU random generation using cuRAND
        curandGenerator_t gen;
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42));

        // curandGenerateNormal requires even count -- allocate temp if odd
        int gen_size = t.size_;
        float* gen_ptr = t.data_ptr();
        float* temp_ptr = nullptr;

        if (gen_size % 2 != 0) {
            // Odd size: allocate one extra float, generate, then copy
            gen_size = t.size_ + 1;
            CUDA_CHECK(cudaMalloc(&temp_ptr, gen_size * sizeof(float)));
            gen_ptr = temp_ptr;
        }

        CURAND_CHECK(curandGenerateNormal(gen, gen_ptr, gen_size,
                                          0.0f, 1.0f));

        if (temp_ptr != nullptr) {
            // Copy only the needed elements back
            CUDA_CHECK(cudaMemcpy(t.data_ptr(), temp_ptr,
                                  t.size_ * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaFree(temp_ptr));
        }

        CURAND_CHECK(curandDestroyGenerator(gen));
    }

    return t;
}

// ---------------------------------------------------------------------------
// arange: Sequential integers [0, 1, 2, ..., n-1].
// ---------------------------------------------------------------------------
// Always generates on CPU first, then transfers to GPU if requested.
// Not worth a kernel for this -- it's only used for testing/debugging.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::arange(int n, Device device) {
    // Generate on CPU
    std::vector<T> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = static_cast<T>(i);
    }

    // Create tensor (constructor handles device transfer if needed)
    Tensor<T> t(std::vector<int>{n}, data.data(), device);
    return t;
}

// ---------------------------------------------------------------------------
// full: Fill with a constant value.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::full(const std::vector<int>& shape, T value,
                          Device device) {
    Tensor<T> t(shape, device);
    if (t.size_ == 0) return t;

    if (device == Device::CPU) {
        T* ptr = t.data_ptr();
        std::fill(ptr, ptr + t.size_, value);
    } else {
        int threads = 256;
        int blocks = (t.size_ + threads - 1) / threads;
        kernel_fill<<<blocks, threads>>>(t.data_ptr(), value, t.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return t;
}

// ===========================================================================
// Device Transfer
// ===========================================================================

// ---------------------------------------------------------------------------
// to_gpu: Copy data from CPU to GPU, return a new GPU tensor.
// ---------------------------------------------------------------------------
// If already on GPU, returns a copy (new allocation). This is intentional --
// it matches PyTorch behavior where .cuda() on a GPU tensor returns a
// reference to the same tensor, but for simplicity we always copy.
//
// The returned tensor has its own data buffer (ref_count = 1), independent
// of the source tensor.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::to_gpu() const {
    if (size_ == 0) return Tensor<T>(shape_, Device::GPU);

    if (device_ == Device::GPU) {
        // Already on GPU: make a copy (device-to-device)
        Tensor<T> result(shape_, Device::GPU);
        CUDA_CHECK(cudaMemcpy(result.data_ptr(), data_ptr(),
                              size_ * sizeof(T), cudaMemcpyDeviceToDevice));
        return result;
    }

    // CPU -> GPU transfer
    // If non-contiguous, we need to handle the stride pattern.
    // For simplicity, make contiguous first, then transfer.
    if (!is_contiguous()) {
        Tensor<T> contig = contiguous();
        return contig.to_gpu();
    }

    Tensor<T> result(shape_, Device::GPU);
    CUDA_CHECK(cudaMemcpy(result.data_ptr(), data_ptr(),
                          size_ * sizeof(T), cudaMemcpyHostToDevice));
    return result;
}

// ---------------------------------------------------------------------------
// to_cpu: Copy data from GPU to CPU, return a new CPU tensor.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::to_cpu() const {
    if (size_ == 0) return Tensor<T>(shape_, Device::CPU);

    if (device_ == Device::CPU) {
        // Already on CPU: make a copy
        Tensor<T> result(shape_, Device::CPU);
        std::memcpy(result.data_ptr(), data_ptr(), size_ * sizeof(T));
        return result;
    }

    // GPU -> CPU transfer
    // Make contiguous first if needed (on GPU), then transfer
    if (!is_contiguous()) {
        Tensor<T> contig = contiguous();
        return contig.to_cpu();
    }

    Tensor<T> result(shape_, Device::CPU);
    CUDA_CHECK(cudaMemcpy(result.data_ptr(), data_ptr(),
                          size_ * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

// ---------------------------------------------------------------------------
// to: Generic device transfer.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::to(Device target) const {
    if (target == Device::GPU) return to_gpu();
    else return to_cpu();
}

// ===========================================================================
// Shape Operations
// ===========================================================================

// ---------------------------------------------------------------------------
// reshape: Change shape without copying data.
// ---------------------------------------------------------------------------
// This is one of the most important operations in DL frameworks. It's O(1)
// because it only changes the metadata (shape and strides), not the data.
//
// Supports one -1 dimension: the size is inferred from the total elements
// and the other dimensions.
//
// Requirements:
//   1. Tensor must be contiguous (strides must match row-major layout)
//   2. New total size must equal old total size
//
// Why contiguity is required:
//   If a tensor is non-contiguous (e.g., transposed), the data in memory
//   doesn't match the logical layout. Changing shape+strides would give
//   wrong element mappings. You must call contiguous() first.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<int>& new_shape) const {
    // Handle the -1 dimension (infer from total size)
    std::vector<int> resolved_shape = new_shape;
    int neg_idx = -1;
    int known_size = 1;

    for (int i = 0; i < static_cast<int>(resolved_shape.size()); i++) {
        if (resolved_shape[i] == -1) {
            if (neg_idx != -1) {
                throw std::runtime_error(
                    "reshape: only one dimension can be -1");
            }
            neg_idx = i;
        } else {
            known_size *= resolved_shape[i];
        }
    }

    if (neg_idx != -1) {
        if (known_size == 0 || size_ % known_size != 0) {
            throw std::runtime_error(
                "reshape: cannot infer -1 dimension (sizes don't divide)");
        }
        resolved_shape[neg_idx] = size_ / known_size;
    }

    // Verify total size matches
    int new_size = compute_size(resolved_shape);
    if (new_size != size_) {
        throw std::runtime_error(
            "reshape: total size mismatch (old=" + std::to_string(size_) +
            ", new=" + std::to_string(new_size) + ")");
    }

    // Check contiguity: reshape only works on contiguous tensors
    if (!is_contiguous()) {
        throw std::runtime_error(
            "reshape: tensor is not contiguous. Call contiguous() first.");
    }

    // Create a new Tensor sharing the same data buffer
    return Tensor<T>(data_, resolved_shape, compute_strides(resolved_shape),
                     size_, device_, offset_);
}

// ---------------------------------------------------------------------------
// view: Alias for reshape (PyTorch compatibility).
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::view(const std::vector<int>& new_shape) const {
    return reshape(new_shape);
}

// ---------------------------------------------------------------------------
// transpose: Swap the last two dimensions.
// ---------------------------------------------------------------------------
// For a 2D tensor with shape {M, N} and strides {N, 1}:
//   After transpose: shape {N, M}, strides {1, N}
//
// For a 3D tensor with shape {B, M, N} and strides {M*N, N, 1}:
//   After transpose: shape {B, N, M}, strides {M*N, 1, N}
//
// The data is NOT moved -- we just reinterpret it by swapping strides.
// This means the tensor becomes non-contiguous after transpose.
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::transpose() const {
    int nd = ndim();
    if (nd < 2) {
        throw std::runtime_error(
            "transpose: need at least 2 dimensions");
    }

    // Swap the last two dimensions in shape and strides
    std::vector<int> new_shape = shape_;
    std::vector<int> new_strides = strides_;

    std::swap(new_shape[nd - 2], new_shape[nd - 1]);
    std::swap(new_strides[nd - 2], new_strides[nd - 1]);

    return Tensor<T>(data_, new_shape, new_strides, size_, device_, offset_);
}

// ---------------------------------------------------------------------------
// is_contiguous: Check if the memory layout is standard row-major.
// ---------------------------------------------------------------------------
// A tensor is contiguous if its strides match what compute_strides would
// produce for its shape. This means data is laid out sequentially in memory
// with no gaps or reordering.
//
// Examples:
//   shape {3, 4}, strides {4, 1}  -> contiguous (standard row-major)
//   shape {4, 3}, strides {1, 4}  -> NOT contiguous (transposed)
//   shape {12},   strides {1}     -> contiguous (1D is always contiguous)
// ---------------------------------------------------------------------------
template <typename T>
bool Tensor<T>::is_contiguous() const {
    std::vector<int> expected = compute_strides(shape_);
    return strides_ == expected;
}

// ---------------------------------------------------------------------------
// contiguous: Return a contiguous version of this tensor.
// ---------------------------------------------------------------------------
// If already contiguous, return *this (no copy, shared data).
// If non-contiguous, allocate a new buffer and copy data element-by-element
// in the correct logical order.
//
// This is needed before:
//   - reshape (which requires contiguous memory)
//   - passing to cuBLAS (which expects contiguous arrays)
//   - device transfer (simpler to transfer contiguous blocks)
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> Tensor<T>::contiguous() const {
    if (is_contiguous()) {
        // Already contiguous: return a view sharing the same data
        return *this;
    }

    // Need to copy data into a new contiguous buffer
    Tensor<T> result(shape_, device_);

    if (device_ == Device::CPU) {
        // CPU: iterate in logical order and copy element by element
        // Use the stride-based indexing to read from non-contiguous source
        std::vector<int> indices(ndim(), 0);
        for (int i = 0; i < size_; i++) {
            // Compute source offset using non-contiguous strides
            int src_offset = offset_;
            for (int d = 0; d < ndim(); d++) {
                src_offset += indices[d] * strides_[d];
            }

            // Write to contiguous destination
            result.data_ptr()[i] = data_.get()[src_offset];

            // Increment multi-dimensional index (like an odometer)
            for (int d = ndim() - 1; d >= 0; d--) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    } else {
        // GPU: launch a kernel that reads using source strides
        // and writes to contiguous positions
        //
        // We need to pass shape and strides to the GPU, so we allocate
        // small device buffers for them.
        int* d_shape = nullptr;
        int* d_strides = nullptr;
        int nd = ndim();

        CUDA_CHECK(cudaMalloc(&d_shape, nd * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_strides, nd * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_shape, shape_.data(), nd * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_strides, strides_.data(), nd * sizeof(int),
                              cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks = (size_ + threads - 1) / threads;
        kernel_make_contiguous<<<blocks, threads>>>(
            data_ptr(), result.data_ptr(), d_shape, d_strides, nd, size_);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_shape));
        CUDA_CHECK(cudaFree(d_strides));
    }

    return result;
}

// ===========================================================================
// Element Access
// ===========================================================================

// ---------------------------------------------------------------------------
// flat_index: Convert multi-dimensional indices to a flat buffer offset.
// ---------------------------------------------------------------------------
// Uses the dot product of indices and strides:
//   offset + indices[0]*strides[0] + indices[1]*strides[1] + ...
//
// This works for both contiguous and non-contiguous tensors, which is
// the whole point of strides.
// ---------------------------------------------------------------------------
template <typename T>
int Tensor<T>::flat_index(const std::vector<int>& indices) const {
    assert(static_cast<int>(indices.size()) == ndim() &&
           "Wrong number of indices");

    int idx = offset_;
    for (int d = 0; d < ndim(); d++) {
        assert(indices[d] >= 0 && indices[d] < shape_[d] &&
               "Index out of bounds");
        idx += indices[d] * strides_[d];
    }
    return idx;
}

// ---------------------------------------------------------------------------
// operator(): Variadic element access.
// ---------------------------------------------------------------------------
// This lets you write tensor(i, j, k) for a 3D tensor. The parameter pack
// is converted to a vector and passed to flat_index.
//
// ONLY works for CPU tensors. Accessing GPU memory element-by-element
// would require a cudaMemcpy per access -- absurdly slow.
// ---------------------------------------------------------------------------
template <typename T>
template <typename... Indices>
T& Tensor<T>::operator()(Indices... indices) {
    assert(device_ == Device::CPU &&
           "Element access only supported on CPU tensors");
    std::vector<int> idx_vec = {static_cast<int>(indices)...};
    return data_.get()[flat_index(idx_vec)];
}

template <typename T>
template <typename... Indices>
const T& Tensor<T>::operator()(Indices... indices) const {
    assert(device_ == Device::CPU &&
           "Element access only supported on CPU tensors");
    std::vector<int> idx_vec = {static_cast<int>(indices)...};
    return data_.get()[flat_index(idx_vec)];
}

// ---------------------------------------------------------------------------
// at(): Element access via vector of indices.
// ---------------------------------------------------------------------------
template <typename T>
T& Tensor<T>::at(const std::vector<int>& indices) {
    assert(device_ == Device::CPU &&
           "Element access only supported on CPU tensors");
    return data_.get()[flat_index(indices)];
}

template <typename T>
const T& Tensor<T>::at(const std::vector<int>& indices) const {
    assert(device_ == Device::CPU &&
           "Element access only supported on CPU tensors");
    return data_.get()[flat_index(indices)];
}

// ===========================================================================
// Print / Display
// ===========================================================================
// Prints the tensor contents in a human-readable format.
// If on GPU, first copies to CPU (we can't read GPU memory from host code).
//
// Format:
//   Tensor "name" (shape=[3, 4], device=GPU):
//   [[ 1.00  2.00  3.00  4.00]
//    [ 5.00  6.00  7.00  8.00]
//    [ 9.00 10.00 11.00 12.00]]
//
// For large tensors (>100 elements), shows only first/last few elements.
// ===========================================================================

template <typename T>
void Tensor<T>::print(const std::string& name) const {
    // Print header
    std::cout << "Tensor";
    if (!name.empty()) std::cout << " \"" << name << "\"";
    std::cout << " (shape=[";
    for (int i = 0; i < ndim(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << shape_[i];
    }
    std::cout << "], strides=[";
    for (int i = 0; i < ndim(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << strides_[i];
    }
    std::cout << "], device=" << device_to_string(device_)
              << ", contiguous=" << (is_contiguous() ? "true" : "false")
              << "):" << std::endl;

    if (size_ == 0) {
        std::cout << "  (empty)" << std::endl;
        return;
    }

    // Get CPU data for printing
    // If on GPU, copy to CPU first. This is a temporary copy just for display.
    Tensor<T> cpu_tensor = (device_ == Device::GPU) ? to_cpu() : *this;

    // If non-contiguous, make contiguous for easy sequential access
    if (!cpu_tensor.is_contiguous()) {
        cpu_tensor = cpu_tensor.contiguous();
    }

    const T* ptr = cpu_tensor.data_ptr();

    // For very large tensors, truncate
    bool truncate = (size_ > 100);

    if (ndim() == 1) {
        // 1D tensor: print as a flat list
        std::cout << "  [";
        for (int i = 0; i < size_; i++) {
            if (truncate && i == 5) {
                std::cout << " ...";
                i = size_ - 4;
                continue;
            }
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << ptr[i];
        }
        std::cout << "]" << std::endl;
    } else if (ndim() == 2) {
        // 2D tensor: print as matrix
        std::cout << "  [";
        for (int i = 0; i < shape_[0]; i++) {
            if (truncate && i == 3 && shape_[0] > 8) {
                std::cout << "   ..." << std::endl;
                i = shape_[0] - 3;
                continue;
            }
            if (i > 0) std::cout << "   ";
            std::cout << "[";
            for (int j = 0; j < shape_[1]; j++) {
                if (truncate && j == 4 && shape_[1] > 8) {
                    std::cout << " ...";
                    j = shape_[1] - 3;
                    continue;
                }
                if (j > 0) std::cout << ", ";
                std::cout << std::fixed << std::setprecision(4)
                          << std::setw(8) << ptr[i * shape_[1] + j];
            }
            std::cout << "]";
            if (i < shape_[0] - 1) std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
    } else {
        // Higher dimensions: print first few elements flat
        std::cout << "  [";
        int show = std::min(size_, 20);
        for (int i = 0; i < show; i++) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << ptr[i];
        }
        if (size_ > show) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
}

#endif // CUDALEARN_TENSOR_CU
