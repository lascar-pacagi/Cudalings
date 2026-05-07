// ===========================================================================
// Chapter 13: tensor_ops.cuh -- Element-wise CUDA Kernels & Operator Overloads
// ===========================================================================
// This file provides the basic arithmetic operations on Tensor objects:
//
//   Binary element-wise: add, subtract, multiply, divide
//   Scalar operations:   add scalar, multiply by scalar
//   Unary operations:    neg, abs, exp, log, sqrt
//   Operator overloads:  +, -, *, / for clean syntax
//
// Each operation is implemented as:
//   1. A CUDA kernel (runs on GPU, grid-stride loop)
//   2. A CPU fallback (simple for loop)
//   3. A wrapper function that dispatches based on device
//   4. An operator overload for syntactic sugar
//
// BROADCASTING:
//   We support limited broadcasting: when one tensor has size 1 in a
//   dimension where the other has size N, the size-1 tensor is "broadcast"
//   (repeated) to match. For simplicity, we only support the case where
//   both tensors have the same total number of elements OR one tensor is
//   a scalar (size 1). Full NumPy-style broadcasting is left for a future
//   enhancement.
//
// WHY SEPARATE FROM tensor.cuh?
//   Separation of concerns:
//   - tensor.cuh: the data structure (shape, strides, memory, device)
//   - tensor_ops.cuh: operations on that data structure
//   This mirrors how PyTorch separates Tensor from its dispatch/ops system.
//   It also keeps each file at a manageable size.
// ===========================================================================

#ifndef CUDALEARN_TENSOR_OPS_CUH
#define CUDALEARN_TENSOR_OPS_CUH

#include "tensor.cuh"
#include <cmath>

// ===========================================================================
// CUDA KERNELS: Binary Element-wise Operations
// ===========================================================================
// All kernels follow the same pattern:
//   1. Compute global thread index
//   2. Grid-stride loop (handle arrays larger than grid)
//   3. Perform the operation element-wise
//
// Template parameter T allows reuse for float, double, etc.
// __restrict__ hints tell the compiler that pointers don't alias,
// enabling better optimization (important for memory-bound kernels).
// ===========================================================================

// ---------------------------------------------------------------------------
// Addition: out[i] = a[i] + b[i]
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_add(const T* __restrict__ a,
                           const T* __restrict__ b,
                           T* __restrict__ out,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// Subtraction: out[i] = a[i] - b[i]
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_sub(const T* __restrict__ a,
                           const T* __restrict__ b,
                           T* __restrict__ out,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] - b[i];
    }
}

// ---------------------------------------------------------------------------
// Multiplication: out[i] = a[i] * b[i]  (element-wise, NOT matrix multiply)
// ---------------------------------------------------------------------------
// Also called Hadamard product. Matrix multiplication comes in Chapter 14.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_mul(const T* __restrict__ a,
                           const T* __restrict__ b,
                           T* __restrict__ out,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] * b[i];
    }
}

// ---------------------------------------------------------------------------
// Division: out[i] = a[i] / b[i]
// ---------------------------------------------------------------------------
// No division-by-zero check in the kernel (would cause divergence).
// The caller is responsible for ensuring b has no zeros, or accepting
// inf/nan results (which is what PyTorch does).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_div(const T* __restrict__ a,
                           const T* __restrict__ b,
                           T* __restrict__ out,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] / b[i];
    }
}

// ===========================================================================
// CUDA KERNELS: Scalar Operations
// ===========================================================================
// Apply a scalar to every element of a tensor.
// These are the workhorses of normalization, scaling, bias addition, etc.
// ===========================================================================

// ---------------------------------------------------------------------------
// Add scalar: out[i] = a[i] + scalar
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_add_scalar(const T* __restrict__ a, T scalar,
                                  T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] + scalar;
    }
}

// ---------------------------------------------------------------------------
// Multiply by scalar: out[i] = a[i] * scalar
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_mul_scalar(const T* __restrict__ a, T scalar,
                                  T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = a[i] * scalar;
    }
}

// ===========================================================================
// CUDA KERNELS: Unary Operations
// ===========================================================================
// Single-input operations that produce a new tensor.
// These are building blocks for activation functions, loss functions, etc.
// ===========================================================================

// ---------------------------------------------------------------------------
// Negate: out[i] = -a[i]
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_neg(const T* __restrict__ a,
                           T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = -a[i];
    }
}

// ---------------------------------------------------------------------------
// Absolute value: out[i] = |a[i]|
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_abs(const T* __restrict__ a,
                           T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = fabsf(a[i]);  // fabsf for float, fabs for double
    }
}

// ---------------------------------------------------------------------------
// Exponential: out[i] = e^(a[i])
// ---------------------------------------------------------------------------
// Used in softmax, sigmoid, and many probability distributions.
// Can overflow for large inputs (produces inf) -- this is expected behavior.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_exp(const T* __restrict__ a,
                           T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = expf(a[i]);
    }
}

// ---------------------------------------------------------------------------
// Natural logarithm: out[i] = ln(a[i])
// ---------------------------------------------------------------------------
// Used in log-likelihood, cross-entropy loss, etc.
// Returns -inf for 0, nan for negative inputs (standard IEEE 754 behavior).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_log(const T* __restrict__ a,
                           T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = logf(a[i]);
    }
}

// ---------------------------------------------------------------------------
// Square root: out[i] = sqrt(a[i])
// ---------------------------------------------------------------------------
// Used in normalization (batch norm, layer norm), RMSProp, Adam, etc.
// Returns nan for negative inputs.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void kernel_sqrt(const T* __restrict__ a,
                            T* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        out[i] = sqrtf(a[i]);
    }
}

// ===========================================================================
// KERNEL LAUNCH CONFIGURATION
// ===========================================================================
// Standard configuration for element-wise operations:
//   - 256 threads per block (good occupancy on most GPUs)
//   - enough blocks to cover all elements
// The grid-stride loop in each kernel handles the case where we launch
// fewer threads than elements.
// ===========================================================================

constexpr int TENSOR_OPS_THREADS = 256;

inline int tensor_ops_blocks(int n) {
    return (n + TENSOR_OPS_THREADS - 1) / TENSOR_OPS_THREADS;
}

// ===========================================================================
// WRAPPER FUNCTIONS: Binary Operations
// ===========================================================================
// These functions:
//   1. Validate inputs (same shape, same device)
//   2. Allocate output tensor
//   3. Dispatch to CUDA kernel (GPU) or CPU loop (CPU)
//   4. Return the result tensor
//
// They ensure both inputs are contiguous before operating, because the
// kernels assume flat sequential memory. A non-contiguous tensor would
// give wrong results with flat indexing.
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper: validate that two tensors are compatible for element-wise ops
// ---------------------------------------------------------------------------
template <typename T>
void validate_binary_op(const Tensor<T>& a, const Tensor<T>& b,
                        const char* op_name) {
    if (a.device_ != b.device_) {
        throw std::runtime_error(
            std::string(op_name) + ": tensors must be on the same device "
            "(got " + device_to_string(a.device_) + " and " +
            device_to_string(b.device_) + ")");
    }
    if (a.shape_ != b.shape_) {
        // Build error message with shapes
        std::string msg = std::string(op_name) + ": shape mismatch ([";
        for (int i = 0; i < a.ndim(); i++) {
            if (i) msg += ",";
            msg += std::to_string(a.shape_[i]);
        }
        msg += "] vs [";
        for (int i = 0; i < b.ndim(); i++) {
            if (i) msg += ",";
            msg += std::to_string(b.shape_[i]);
        }
        msg += "])";
        throw std::runtime_error(msg);
    }
}

// ---------------------------------------------------------------------------
// tensor_add: Element-wise addition
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_add(const Tensor<T>& a, const Tensor<T>& b) {
    validate_binary_op(a, b, "add");

    // Ensure contiguous memory for flat kernel access
    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> cb = b.is_contiguous() ? b : b.contiguous();

    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_add<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), cb.data_ptr(), result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        const T* pb = cb.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = pa[i] + pb[i];
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// tensor_sub: Element-wise subtraction
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_sub(const Tensor<T>& a, const Tensor<T>& b) {
    validate_binary_op(a, b, "sub");

    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> cb = b.is_contiguous() ? b : b.contiguous();

    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_sub<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), cb.data_ptr(), result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        const T* pb = cb.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = pa[i] - pb[i];
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// tensor_mul: Element-wise multiplication (Hadamard product)
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_mul(const Tensor<T>& a, const Tensor<T>& b) {
    validate_binary_op(a, b, "mul");

    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> cb = b.is_contiguous() ? b : b.contiguous();

    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_mul<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), cb.data_ptr(), result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        const T* pb = cb.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = pa[i] * pb[i];
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// tensor_div: Element-wise division
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_div(const Tensor<T>& a, const Tensor<T>& b) {
    validate_binary_op(a, b, "div");

    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> cb = b.is_contiguous() ? b : b.contiguous();

    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_div<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), cb.data_ptr(), result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        const T* pb = cb.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = pa[i] / pb[i];
        }
    }

    return result;
}

// ===========================================================================
// WRAPPER FUNCTIONS: Scalar Operations
// ===========================================================================

// ---------------------------------------------------------------------------
// tensor_add_scalar: Add a scalar to every element
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_add_scalar(const Tensor<T>& a, T scalar) {
    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_add_scalar<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), scalar, result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = pa[i] + scalar;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// tensor_mul_scalar: Multiply every element by a scalar
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_mul_scalar(const Tensor<T>& a, T scalar) {
    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_mul_scalar<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), scalar, result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = pa[i] * scalar;
        }
    }

    return result;
}

// ===========================================================================
// WRAPPER FUNCTIONS: Unary Operations
// ===========================================================================

// ---------------------------------------------------------------------------
// tensor_neg: Negate every element
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_neg(const Tensor<T>& a) {
    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_neg<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = -pa[i];
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// tensor_abs: Absolute value of every element
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_abs(const Tensor<T>& a) {
    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_abs<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = std::abs(pa[i]);
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// tensor_exp: e^x for every element
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_exp(const Tensor<T>& a) {
    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_exp<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = std::exp(pa[i]);
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// tensor_log: Natural logarithm for every element
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_log(const Tensor<T>& a) {
    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_log<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = std::log(pa[i]);
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// tensor_sqrt: Square root for every element
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> tensor_sqrt(const Tensor<T>& a) {
    Tensor<T> ca = a.is_contiguous() ? a : a.contiguous();
    Tensor<T> result(a.shape_, a.device_);

    if (a.device_ == Device::GPU) {
        kernel_sqrt<<<tensor_ops_blocks(a.size_), TENSOR_OPS_THREADS>>>(
            ca.data_ptr(), result.data_ptr(), a.size_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        const T* pa = ca.data_ptr();
        T* pr = result.data_ptr();
        for (int i = 0; i < a.size_; i++) {
            pr[i] = std::sqrt(pa[i]);
        }
    }

    return result;
}

// ===========================================================================
// OPERATOR OVERLOADS
// ===========================================================================
// These let you write natural math expressions:
//   auto c = a + b;       // element-wise addition
//   auto d = a * 2.0f;    // scalar multiplication
//   auto e = -a;           // negation
//
// We define them as free (non-member) functions so that they can handle
// mixed operand types (e.g., scalar + tensor, tensor + scalar).
//
// Note: These return NEW tensors. They don't modify the operands.
// This is the functional/immutable style used by most DL frameworks.
// ===========================================================================

// ---------------------------------------------------------------------------
// Binary operators: Tensor op Tensor
// ---------------------------------------------------------------------------

template <typename T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
    return tensor_add(a, b);
}

template <typename T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
    return tensor_sub(a, b);
}

template <typename T>
Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
    return tensor_mul(a, b);
}

template <typename T>
Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b) {
    return tensor_div(a, b);
}

// ---------------------------------------------------------------------------
// Scalar operators: Tensor op Scalar and Scalar op Tensor
// ---------------------------------------------------------------------------
// We need both orderings: tensor + scalar AND scalar + tensor.
// For subtraction and division, order matters:
//   tensor - scalar = add_scalar(tensor, -scalar)
//   scalar - tensor = neg(tensor) + scalar  (negate then add)
// ---------------------------------------------------------------------------

// Tensor + scalar
template <typename T>
Tensor<T> operator+(const Tensor<T>& a, T scalar) {
    return tensor_add_scalar(a, scalar);
}

// scalar + Tensor (commutative)
template <typename T>
Tensor<T> operator+(T scalar, const Tensor<T>& a) {
    return tensor_add_scalar(a, scalar);
}

// Tensor - scalar
template <typename T>
Tensor<T> operator-(const Tensor<T>& a, T scalar) {
    return tensor_add_scalar(a, -scalar);
}

// scalar - Tensor
template <typename T>
Tensor<T> operator-(T scalar, const Tensor<T>& a) {
    return tensor_add_scalar(tensor_neg(a), scalar);
}

// Tensor * scalar
template <typename T>
Tensor<T> operator*(const Tensor<T>& a, T scalar) {
    return tensor_mul_scalar(a, scalar);
}

// scalar * Tensor (commutative)
template <typename T>
Tensor<T> operator*(T scalar, const Tensor<T>& a) {
    return tensor_mul_scalar(a, scalar);
}

// Tensor / scalar
template <typename T>
Tensor<T> operator/(const Tensor<T>& a, T scalar) {
    return tensor_mul_scalar(a, static_cast<T>(1) / scalar);
}

// ---------------------------------------------------------------------------
// Unary minus: -Tensor
// ---------------------------------------------------------------------------
template <typename T>
Tensor<T> operator-(const Tensor<T>& a) {
    return tensor_neg(a);
}

#endif // CUDALEARN_TENSOR_OPS_CUH
