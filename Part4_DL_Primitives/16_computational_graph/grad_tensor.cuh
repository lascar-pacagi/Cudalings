// ===========================================================================
// Chapter 16: grad_tensor.cuh -- GradTensor: Autograd-Enabled GPU Tensor
// ===========================================================================
//
// This header defines the GradTensor class, which is the fundamental building
// block of our autograd engine. It wraps a GPU float* pointer with automatic
// gradient tracking, inspired by PyTorch's Tensor and Karpathy's micrograd.
//
// DESIGN OVERVIEW:
//
//   Each GradTensor stores:
//     - data         : float* on GPU -- the tensor's values
//     - grad         : float* on GPU -- dL/d(self), accumulated during backward
//     - shape        : vector<int>   -- dimensions (e.g., {64, 3, 32, 32})
//     - size         : int           -- total number of elements
//     - requires_grad: bool          -- whether to track gradients
//     - backward_fn  : function      -- closure that computes input gradients
//     - children     : vector        -- shared_ptrs to input tensors
//     - name         : string        -- human-readable name for debugging
//
//   We use shared_ptr<GradTensor> everywhere to manage memory lifetimes.
//   When a GradTensor is created by an operation (e.g., autograd::relu),
//   its backward_fn captures the input tensors by shared_ptr, which keeps
//   them alive as long as they're needed for the backward pass.
//
// BACKWARD PASS ALGORITHM:
//
//   1. Topological sort via DFS (post-order)
//   2. Reverse the order
//   3. Seed the output gradient with 1.0
//   4. For each node in reverse topological order:
//      - Call its backward_fn (which accumulates gradients into children)
//
// ===========================================================================

#pragma once

#include <cuda_runtime.h>
#include <curand.h>
#include <vector>
#include <string>
#include <functional>
#include <memory>
#include <set>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <algorithm>

// ===========================================================================
// Forward declaration
// ===========================================================================
class GradTensor;
using GradTensorPtr = std::shared_ptr<GradTensor>;

// ===========================================================================
// Simple CUDA error checking macro
// ===========================================================================
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
#endif

// ===========================================================================
// Kernel: fill a GPU array with a constant value
// ===========================================================================
__global__ void fill_kernel(float* data, float value, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = value;
    }
}

// ===========================================================================
// Kernel: elementwise addition for gradient accumulation
// dst[i] += src[i]
// ===========================================================================
__global__ void accumulate_kernel(float* dst, const float* src, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] += src[idx];
    }
}

// ===========================================================================
// GradTensor Class Definition
// ===========================================================================
class GradTensor {
public:
    // -----------------------------------------------------------------------
    // Member variables
    // -----------------------------------------------------------------------

    float* data;             // GPU pointer to tensor values
    float* grad;             // GPU pointer to gradient values (dL/d(self))
    std::vector<int> shape;  // Tensor dimensions (e.g., {B, C, H, W})
    int size;                // Total number of elements (product of shape)
    bool requires_grad;      // Whether this tensor tracks gradients
    std::string name;        // Human-readable name for debugging/visualization

    // The backward function: a closure that, when called, computes the
    // gradients of this node's children using this node's .grad field.
    // For leaf tensors (parameters, inputs), this is empty (nullptr).
    std::function<void()> backward_fn;

    // Pointers to the input tensors that produced this tensor.
    // These form the edges of the computational graph.
    // For leaf tensors, this vector is empty.
    std::vector<GradTensorPtr> children;

    // Operation name for visualization (e.g., "conv2d", "relu", "add")
    std::string op_name;

    // -----------------------------------------------------------------------
    // Constructor: allocate GPU memory for data and (optionally) grad
    // -----------------------------------------------------------------------
    //
    // Parameters:
    //   shape_         -- dimensions of the tensor
    //   requires_grad_ -- if true, allocate gradient storage
    //   name_          -- human-readable name (default: "unnamed")
    //
    // The constructor allocates GPU memory for data (always) and grad
    // (only if requires_grad is true). Both are initialized to zero.
    //
    GradTensor(const std::vector<int>& shape_, bool requires_grad_ = false,
               const std::string& name_ = "unnamed")
        : shape(shape_), requires_grad(requires_grad_), name(name_),
          data(nullptr), grad(nullptr), backward_fn(nullptr), op_name("")
    {
        // Compute total size as the product of all dimensions.
        // E.g., shape = {2, 3, 4} => size = 24
        size = 1;
        for (int d : shape) {
            size *= d;
        }

        // Allocate GPU memory for the tensor data
        CUDA_CHECK(cudaMalloc(&data, size * sizeof(float)));

        // Initialize data to zero
        int blocks = (size + 255) / 256;
        fill_kernel<<<blocks, 256>>>(data, 0.0f, size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // If tracking gradients, allocate and zero the gradient buffer
        if (requires_grad) {
            CUDA_CHECK(cudaMalloc(&grad, size * sizeof(float)));
            fill_kernel<<<blocks, 256>>>(grad, 0.0f, size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // -----------------------------------------------------------------------
    // Destructor: free GPU memory
    // -----------------------------------------------------------------------
    ~GradTensor() {
        if (data) {
            cudaFree(data);
            data = nullptr;
        }
        if (grad) {
            cudaFree(grad);
            grad = nullptr;
        }
    }

    // -----------------------------------------------------------------------
    // No copy (prevent accidental GPU memory duplication)
    // -----------------------------------------------------------------------
    GradTensor(const GradTensor&) = delete;
    GradTensor& operator=(const GradTensor&) = delete;

    // -----------------------------------------------------------------------
    // zero_grad: reset the gradient buffer to all zeros
    // -----------------------------------------------------------------------
    // Called before each training iteration to clear stale gradients.
    // In PyTorch: optimizer.zero_grad()
    //
    void zero_grad() {
        if (grad) {
            int blocks = (size + 255) / 256;
            fill_kernel<<<blocks, 256>>>(grad, 0.0f, size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // -----------------------------------------------------------------------
    // ensure_grad: allocate gradient buffer if not already allocated
    // -----------------------------------------------------------------------
    // Some tensors start without gradient storage but need it later
    // (e.g., intermediate activations that receive gradients during backward).
    //
    void ensure_grad() {
        if (!grad) {
            CUDA_CHECK(cudaMalloc(&grad, size * sizeof(float)));
            int blocks = (size + 255) / 256;
            fill_kernel<<<blocks, 256>>>(grad, 0.0f, size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // -----------------------------------------------------------------------
    // set_data_from_host: copy data from CPU array to GPU
    // -----------------------------------------------------------------------
    void set_data_from_host(const float* host_data) {
        CUDA_CHECK(cudaMemcpy(data, host_data, size * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // -----------------------------------------------------------------------
    // get_data_to_host: copy data from GPU to CPU array
    // -----------------------------------------------------------------------
    void get_data_to_host(float* host_data) const {
        CUDA_CHECK(cudaMemcpy(host_data, data, size * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // -----------------------------------------------------------------------
    // get_grad_to_host: copy gradient from GPU to CPU array
    // -----------------------------------------------------------------------
    void get_grad_to_host(float* host_grad) const {
        if (grad) {
            CUDA_CHECK(cudaMemcpy(host_grad, grad, size * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }
    }

    // -----------------------------------------------------------------------
    // backward: run the backward pass starting from this tensor
    // -----------------------------------------------------------------------
    //
    // This is the core of the autograd engine. It:
    //   1. Builds a topological ordering of the graph via DFS
    //   2. Seeds this tensor's gradient with 1.0 (dL/dL = 1)
    //   3. Iterates in reverse topological order, calling each backward_fn
    //
    // ALGORITHM:
    //
    //   topo_order = topological_sort(this)   // via DFS post-order
    //   this->grad = 1.0                      // seed gradient
    //
    //   for node in reverse(topo_order):
    //       if node.backward_fn exists:
    //           node.backward_fn()            // propagates grad to children
    //
    // This correctly handles:
    //   - Gradient accumulation (tensors used multiple times)
    //   - Arbitrary DAG structures (skip connections, etc.)
    //   - Any mix of requires_grad settings
    //
    void backward() {
        // ---- Step 1: Topological sort via DFS ----
        //
        // We do a depth-first traversal starting from this node (the loss).
        // Post-order means we add a node to the list AFTER visiting all
        // its children. Reversing gives us the correct backward order.
        //
        std::vector<GradTensor*> topo_order;
        std::set<GradTensor*> visited;

        // Lambda for DFS traversal (post-order)
        std::function<void(GradTensor*)> dfs = [&](GradTensor* node) {
            if (visited.count(node)) return;  // Already visited
            visited.insert(node);

            // Visit all children (inputs to this operation) first
            for (auto& child : node->children) {
                dfs(child.get());
            }

            // Post-order: add this node AFTER all children are visited
            topo_order.push_back(node);
        };

        dfs(this);

        // ---- Step 2: Seed gradient ----
        //
        // The gradient of the loss with respect to itself is always 1.0.
        // This is the starting point for backpropagation.
        //
        this->ensure_grad();
        int blocks = (this->size + 255) / 256;
        fill_kernel<<<blocks, 256>>>(this->grad, 1.0f, this->size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- Step 3: Reverse topological order gradient propagation ----
        //
        // Process nodes from output to input. For each node that has a
        // backward_fn, call it. The backward_fn reads this node's .grad
        // and ACCUMULATES gradients into its children's .grad fields.
        //
        // Why reverse? In post-order, children come before parents.
        // We want to process parents first (they have the gradients we need).
        //
        for (int i = (int)topo_order.size() - 1; i >= 0; i--) {
            GradTensor* node = topo_order[i];
            if (node->backward_fn) {
                node->backward_fn();
            }
        }
    }

    // -----------------------------------------------------------------------
    // shape_str: return a human-readable shape string for debugging
    // -----------------------------------------------------------------------
    std::string shape_str() const {
        std::string s = "(";
        for (int i = 0; i < (int)shape.size(); i++) {
            if (i > 0) s += ", ";
            s += std::to_string(shape[i]);
        }
        s += ")";
        return s;
    }

    // -----------------------------------------------------------------------
    // print_info: print tensor name, shape, and summary statistics
    // -----------------------------------------------------------------------
    void print_info() const {
        printf("GradTensor '%s': shape=%s, size=%d, requires_grad=%s",
               name.c_str(), shape_str().c_str(), size,
               requires_grad ? "true" : "false");
        if (!op_name.empty()) {
            printf(", op=%s", op_name.c_str());
        }
        printf("\n");
    }
};

// ===========================================================================
// Factory Functions
// ===========================================================================

// ---------------------------------------------------------------------------
// make_grad_tensor: create a GradTensor with the given shape
// ---------------------------------------------------------------------------
// This is the basic factory. Data is initialized to zero.
//
inline GradTensorPtr make_grad_tensor(const std::vector<int>& shape,
                                       bool requires_grad = false,
                                       const std::string& name = "unnamed") {
    return std::make_shared<GradTensor>(shape, requires_grad, name);
}

// ---------------------------------------------------------------------------
// make_parameter: create a learnable parameter with random initialization
// ---------------------------------------------------------------------------
//
// Allocates a GradTensor and fills it with random values from a normal
// distribution scaled by 1/sqrt(fan_in). This is a simplified version
// of Kaiming/He initialization.
//
// Parameters:
//   shape    -- dimensions of the parameter tensor
//   name     -- name for debugging (e.g., "conv1.weight")
//   fan_in   -- number of input units (for scaling); if 0, uses shape product / shape[0]
//
// The random values are generated using cuRAND on the GPU.
//
inline GradTensorPtr make_parameter(const std::vector<int>& shape,
                                     const std::string& name,
                                     int fan_in = 0) {
    auto param = make_grad_tensor(shape, /*requires_grad=*/true, name);

    // Compute fan_in if not provided
    // For a conv weight (OC, IC, KH, KW), fan_in = IC * KH * KW
    // For a linear weight (out, in), fan_in = in
    if (fan_in <= 0) {
        fan_in = 1;
        for (int i = 1; i < (int)shape.size(); i++) {
            fan_in *= shape[i];
        }
        if (fan_in == 0) fan_in = shape[0];  // 1D parameter (bias)
    }

    // Scale for Kaiming initialization: std = sqrt(2 / fan_in)
    float std_dev = sqrtf(2.0f / (float)fan_in);

    // Generate random normal values using cuRAND
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42ULL);
    curandGenerateNormal(gen, param->data, param->size, 0.0f, std_dev);
    curandDestroyGenerator(gen);

    CUDA_CHECK(cudaDeviceSynchronize());
    return param;
}

// ---------------------------------------------------------------------------
// make_ones: create a tensor filled with ones (useful for batch norm gamma)
// ---------------------------------------------------------------------------
inline GradTensorPtr make_ones(const std::vector<int>& shape,
                                bool requires_grad,
                                const std::string& name) {
    auto t = make_grad_tensor(shape, requires_grad, name);
    int blocks = (t->size + 255) / 256;
    fill_kernel<<<blocks, 256>>>(t->data, 1.0f, t->size);
    CUDA_CHECK(cudaDeviceSynchronize());
    return t;
}

// ---------------------------------------------------------------------------
// make_zeros: create a tensor filled with zeros (useful for bias, beta)
// ---------------------------------------------------------------------------
inline GradTensorPtr make_zeros(const std::vector<int>& shape,
                                 bool requires_grad,
                                 const std::string& name) {
    // Already zero from constructor, but explicit for clarity
    return make_grad_tensor(shape, requires_grad, name);
}
