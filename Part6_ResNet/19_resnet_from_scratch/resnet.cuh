/*******************************************************************************
 * resnet.cuh — Self-Contained ResNet Implementation in CUDA
 *
 * CAPSTONE: Chapter 19 — ResNet from Scratch
 *
 * This single header contains EVERYTHING needed to build, train, and evaluate
 * a pre-activation ResNet-v2 for board game evaluation:
 *
 *   - GPU memory helpers (CUDA_CHECK, malloc/free wrappers)
 *   - GradTensor class (data + grad on GPU, autograd graph)
 *   - CUDA kernels for all operations (forward AND backward)
 *   - Autograd wrappers that build the computational graph
 *   - Module base class + Layer classes (Conv2d, BatchNorm2d, ReLU, Linear, ...)
 *   - ResBlock (pre-activation, matching cnn_resnet.py)
 *   - ResNet class (stem + body + head)
 *   - Adam optimizer with CosineAnnealingLR
 *   - CrossEntropyLoss
 *
 * Target architecture from cnn_resnet.py:
 *   Input: (B, 4, 8, 8)  — board game planes
 *   Output: (B, 3)        — {black wins, draw, white wins}
 *
 * Hardware: Quadro P4200 (Compute Capability 6.1), CUDA 11.7
 *
 * All design decisions reference what we learned in earlier chapters:
 *   Ch12: Raw CUDA kernels — we write all kernels here
 *   Ch13: GPU tensors with grad tracking — our GradTensor
 *   Ch14: Forward ops — conv2d, batchnorm, relu, linear, pooling
 *   Ch15: Backward ops — gradient computation for every forward op
 *   Ch16: Computational graph — topological sort + reverse-mode AD
 *   Ch17: Module/Sequential/Optimizer — our class hierarchy
 *   Ch18: Python frontend concepts — PyTorch-like API
 *
 * Compile: nvcc -arch=sm_61 -O2 -ccbin g++-11 -std=c++14 -lcurand
 ******************************************************************************/

#ifndef RESNET_CUH
#define RESNET_CUH

// ============================================================================
// Standard includes
// ============================================================================
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <string>
#include <set>
#include <chrono>

// ============================================================================
// SECTION A: GPU Memory Helpers
// ============================================================================
//
// From Chapter 12: always check CUDA calls. A single unchecked error can
// silently corrupt everything downstream.
// ============================================================================

// CUDA_CHECK: wraps every CUDA API call with error checking.
// In production frameworks (cuDNN, etc.) this is the first thing you add.
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// CURAND_CHECK: same idea for cuRAND calls (used in random initialization).
#define CURAND_CHECK(call)                                                     \
    do {                                                                        \
        curandStatus_t status = (call);                                         \
        if (status != CURAND_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuRAND error at %s:%d: status=%d\n",              \
                    __FILE__, __LINE__, (int)status);                           \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// gpu_malloc: typed wrapper around cudaMalloc.
// Returns a float* pointing to 'count' floats on the GPU.
inline float* gpu_malloc(int count) {
    float* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
    return ptr;
}

// gpu_free: wrapper around cudaFree.
inline void gpu_free(float* ptr) {
    if (ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

// gpu_memset_zero: zero out GPU memory.
inline void gpu_memset_zero(float* ptr, int count) {
    CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(float)));
}

// gpu_copy_h2d: host-to-device copy.
inline void gpu_copy_h2d(float* dst, const float* src, int count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice));
}

// gpu_copy_d2h: device-to-host copy.
inline void gpu_copy_d2h(float* dst, const float* src, int count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost));
}

// gpu_copy_d2d: device-to-device copy.
inline void gpu_copy_d2d(float* dst, const float* src, int count) {
    CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice));
}

// ============================================================================
// SECTION B: GradTensor Class
// ============================================================================
//
// From Chapter 13: a tensor that lives on the GPU with optional gradient
// tracking. This is our equivalent of PyTorch's torch.Tensor with
// requires_grad=True.
//
// Key design choices:
//   - data and grad are raw float* on GPU (no smart pointer on the arrays
//     themselves — we manage lifetime through the GradTensor's own lifecycle)
//   - backward_fn: closure that computes gradients for this node's inputs
//   - children: the input tensors this node depends on (for topo sort)
//   - backward(): topological sort via DFS, then propagate in reverse
//
// Unlike PyTorch which uses reference counting + custom pointers, we use a
// simpler ownership model: tensors created by autograd ops are owned by
// a global arena that gets cleared between training steps.
// ============================================================================

struct GradTensor {
    // === Data members ===

    float* data;       // GPU pointer to tensor data
    float* grad;       // GPU pointer to gradient (same shape as data)
    std::vector<int> shape;  // e.g., {B, C, H, W} for a 4D tensor
    int size;          // total number of elements = product of shape

    bool requires_grad;  // whether to track gradients for this tensor

    // Autograd graph links
    std::function<void()> backward_fn;     // computes grad for children
    std::vector<GradTensor*> children;     // input tensors (graph edges)

    // === Constructor ===
    // Allocates data (and optionally grad) on GPU.
    GradTensor(const std::vector<int>& shape_, bool requires_grad_ = false)
        : data(nullptr), grad(nullptr), shape(shape_),
          requires_grad(requires_grad_), backward_fn(nullptr)
    {
        size = 1;
        for (int s : shape) size *= s;
        data = gpu_malloc(size);
        if (requires_grad) {
            grad = gpu_malloc(size);
            gpu_memset_zero(grad, size);
        }
    }

    // === Destructor ===
    ~GradTensor() {
        if (data) { gpu_free(data); data = nullptr; }
        if (grad) { gpu_free(grad); grad = nullptr; }
    }

    // No copy (GPU memory is expensive to duplicate)
    GradTensor(const GradTensor&) = delete;
    GradTensor& operator=(const GradTensor&) = delete;

    // === zero_grad: reset gradient to zero ===
    // Called before each backward pass (Chapter 16: accumulated gradients
    // must be reset between steps).
    void zero_grad() {
        if (grad) {
            gpu_memset_zero(grad, size);
        }
    }

    // === alloc_grad: lazily allocate gradient buffer ===
    void alloc_grad() {
        if (!grad) {
            grad = gpu_malloc(size);
            gpu_memset_zero(grad, size);
        }
    }

    // === to_host: copy data to a host vector ===
    std::vector<float> to_host() const {
        std::vector<float> h(size);
        gpu_copy_d2h(h.data(), data, size);
        return h;
    }

    // === grad_to_host: copy gradient to a host vector ===
    std::vector<float> grad_to_host() const {
        std::vector<float> h(size);
        if (grad) {
            gpu_copy_d2h(h.data(), grad, size);
        }
        return h;
    }

    // === from_host: copy data from a host vector ===
    void from_host(const std::vector<float>& h) {
        assert((int)h.size() == size);
        gpu_copy_h2d(data, h.data(), size);
    }

    // === fill: set all elements to a constant value ===
    // We copy a host buffer — simple but fine for initialization.
    void fill(float val) {
        std::vector<float> h(size, val);
        from_host(h);
    }

    // === backward(): reverse-mode automatic differentiation ===
    //
    // From Chapter 16: to compute gradients, we:
    //   1. Build a topological ordering of the graph via DFS
    //   2. Set the output gradient to 1.0 (d_loss/d_loss = 1)
    //   3. Walk the ordering in reverse, calling each node's backward_fn
    //
    // This is exactly how PyTorch's autograd engine works (simplified).
    // The topological sort ensures that when we process a node, all nodes
    // that USE its output have already propagated their gradients to it.
    void backward() {
        // Step 1: Topological sort via iterative DFS
        std::vector<GradTensor*> topo_order;
        std::set<GradTensor*> visited;

        // Iterative DFS using explicit stack (avoids stack overflow on deep graphs)
        struct DFSFrame { GradTensor* node; int child_idx; };
        std::vector<DFSFrame> stack;
        stack.push_back({this, 0});
        visited.insert(this);

        while (!stack.empty()) {
            DFSFrame& frame = stack.back();
            if (frame.child_idx < (int)frame.node->children.size()) {
                GradTensor* child = frame.node->children[frame.child_idx];
                frame.child_idx++;
                if (visited.find(child) == visited.end()) {
                    visited.insert(child);
                    stack.push_back({child, 0});
                }
            } else {
                topo_order.push_back(frame.node);
                stack.pop_back();
            }
        }

        // Step 2: Reverse the topological order (output first)
        std::reverse(topo_order.begin(), topo_order.end());

        // Step 3: Seed the output gradient with 1.0
        // d(loss)/d(loss) = 1
        alloc_grad();
        fill_grad_ones();

        // Step 4: Propagate gradients in reverse topological order
        for (GradTensor* node : topo_order) {
            if (node->backward_fn) {
                node->backward_fn();
            }
        }
    }

    // === fill_grad_ones: set gradient to all 1s ===
    void fill_grad_ones() {
        if (!grad) {
            grad = gpu_malloc(size);
        }
        std::vector<float> ones(size, 1.0f);
        gpu_copy_h2d(grad, ones.data(), size);
    }

    // === Static factory methods ===

    // zeros: create a tensor filled with zeros
    static GradTensor* zeros(const std::vector<int>& shape, bool requires_grad = false) {
        GradTensor* t = new GradTensor(shape, requires_grad);
        gpu_memset_zero(t->data, t->size);
        return t;
    }

    // ones: create a tensor filled with ones
    static GradTensor* ones(const std::vector<int>& shape, bool requires_grad = false) {
        GradTensor* t = new GradTensor(shape, requires_grad);
        t->fill(1.0f);
        return t;
    }

    // randn: create a tensor with random normal values using cuRAND
    // From Chapter 12: cuRAND is the GPU random number generator.
    // std_dev controls the scale (used for Kaiming initialization).
    static GradTensor* randn(const std::vector<int>& shape, float std_dev = 1.0f,
                             bool requires_grad = false) {
        GradTensor* t = new GradTensor(shape, requires_grad);

        // Create cuRAND generator
        curandGenerator_t gen;
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen,
            (unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count()));

        // Generate normal random numbers with mean=0, stddev=std_dev
        // cuRAND requires even count for normal generation
        int gen_count = (t->size + 1) & ~1;  // round up to even
        float* temp = gpu_malloc(gen_count);
        CURAND_CHECK(curandGenerateNormal(gen, temp, gen_count, 0.0f, std_dev));

        // Copy to tensor (in case size was odd)
        gpu_copy_d2d(t->data, temp, t->size);
        gpu_free(temp);
        CURAND_CHECK(curandDestroyGenerator(gen));

        return t;
    }
};

// Global arena to track intermediate tensors created during forward pass.
// Cleared between training steps to avoid memory leaks.
// This is a simplification — PyTorch uses reference counting instead.
static std::vector<GradTensor*> g_tensor_arena;

// Clear all intermediate tensors from the arena.
// Call this at the start of each training step.
inline void clear_tensor_arena() {
    for (auto* t : g_tensor_arena) {
        delete t;
    }
    g_tensor_arena.clear();
}

// ============================================================================
// SECTION C: CUDA Kernels — Forward and Backward for All Operations
// ============================================================================
//
// From Chapters 14-15: every forward operation needs a corresponding backward
// operation that computes gradients with respect to its inputs.
//
// We implement all kernels here as __global__ functions. Each kernel is
// designed for the shapes encountered in ResNet:
//   - Conv2d: 3x3 with padding=1, no bias (direct convolution)
//   - BatchNorm2d: per-channel normalization over (N,H,W)
//   - ReLU: element-wise max(0, x)
//   - Linear: matrix multiply (fully connected)
//   - GlobalAvgPool2d: mean over spatial dimensions
//   - CrossEntropy: stable softmax + negative log-likelihood
//   - Add: element-wise addition (for residual connections)
//
// Thread assignment strategy (Chapter 12):
//   - Most kernels: one thread per output element
//   - Reductions (batchnorm, GAP): one thread per channel with loops
// ============================================================================

// ---- Utility kernels ----

// fill_kernel: set all elements to a value
__global__ void fill_kernel(float* data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

// add_kernel: element-wise addition c = a + b
// Used for residual connections: output = conv_path + skip_connection
// Backward: d_a = d_c, d_b = d_c (gradient copies to both inputs)
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// add_grad_kernel: accumulate gradient (d_input += d_output)
// Both inputs of an add receive a COPY of the output gradient.
// This is the key insight behind residual networks (Chapter 19 README):
// the gradient flows unchanged through the skip connection.
__global__ void add_grad_kernel(float* d_input, const float* d_output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&d_input[idx], d_output[idx]);
    }
}

// scale_kernel: multiply all elements by a scalar
__global__ void scale_kernel(float* data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= scale;
}


// ---- ReLU kernels ----
//
// From Chapter 14: ReLU(x) = max(0, x)
// From Chapter 15: d_ReLU/d_x = (x > 0) ? 1 : 0
//
// The backward pass needs the original input to determine which elements
// were positive. We store the input during forward pass.

__global__ void relu_forward_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* input, const float* d_output,
                                     float* d_input, int n) {
    // d_input[i] += d_output[i] * (input[i] > 0)
    // The += is important: gradients accumulate when a tensor is used
    // multiple times (e.g., the input to a ResBlock is used both in the
    // main path and the skip connection).
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (input[idx] > 0.0f) {
            atomicAdd(&d_input[idx], d_output[idx]);
        }
    }
}


// ---- Conv2d kernels (direct convolution, NCHW format) ----
//
// From Chapter 14: direct convolution is O(N * C_out * C_in * kH * kW * oH * oW).
// For our ResNet with 3x3 kernels on 8x8 boards, this is perfectly fine.
// Using cuDNN or im2col+GEMM would be faster for large images, but for 8x8
// the overhead of those approaches isn't worth it (Chapter 12: kernel launch
// overhead matters for small workloads).
//
// Layout: NCHW
//   input:  (N, C_in,  H,  W)
//   weight: (C_out, C_in, kH, kW)
//   output: (N, C_out, oH, oW)
//   where oH = H + 2*pad - kH + 1, oW = W + 2*pad - kW + 1
//
// With 3x3 kernel and padding=1: oH = H, oW = W (spatial dims preserved).
// This is critical for ResNet: all feature maps in a ResBlock have the
// same spatial dimensions, enabling identity skip connections.

__global__ void conv2d_forward_kernel(
    const float* input,   // (N, C_in, H, W)
    const float* weight,  // (C_out, C_in, kH, kW)
    float* output,        // (N, C_out, oH, oW)
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int oH, int oW, int pad)
{
    // One thread per output element.
    // Total threads = N * C_out * oH * oW
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * oH * oW;
    if (idx >= total) return;

    // Decode linear index into (n, co, oh, ow)
    int ow = idx % oW;
    int oh = (idx / oW) % oH;
    int co = (idx / (oW * oH)) % C_out;
    int n  = idx / (oW * oH * C_out);

    // Compute the convolution sum for output[n][co][oh][ow]
    //
    // output[n][co][oh][ow] = sum over ci, kh, kw of:
    //     input[n][ci][oh - pad + kh][ow - pad + kw] * weight[co][ci][kh][kw]
    //
    // This is the definition from Chapter 14 — direct sliding window.
    float sum = 0.0f;
    for (int ci = 0; ci < C_in; ci++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                int ih = oh - pad + kh;  // input row
                int iw = ow - pad + kw;  // input col
                // Boundary check (padding with zeros)
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float in_val = input[((n * C_in + ci) * H + ih) * W + iw];
                    float w_val  = weight[((co * C_in + ci) * kH + kh) * kW + kw];
                    sum += in_val * w_val;
                }
            }
        }
    }
    output[idx] = sum;
}

// conv2d backward w.r.t. input (d_input):
//
// From Chapter 15: the gradient of the loss w.r.t. conv input is the
// "full convolution" of d_output with the ROTATED weight kernel.
//
// Mathematically:
//   d_input[n][ci][ih][iw] = sum over co, kh, kw of:
//       d_output[n][co][ih + pad - kh][iw + pad - kw] * weight[co][ci][kh][kw]
//
// This is computed by iterating over the output positions that this input
// pixel contributed to during the forward pass.
__global__ void conv2d_backward_input_kernel(
    const float* d_output,  // (N, C_out, oH, oW)
    const float* weight,    // (C_out, C_in, kH, kW)
    float* d_input,         // (N, C_in, H, W)  — accumulated
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int oH, int oW, int pad)
{
    // One thread per input element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H * W;
    if (idx >= total) return;

    int iw = idx % W;
    int ih = (idx / W) % H;
    int ci = (idx / (W * H)) % C_in;
    int n  = idx / (W * H * C_in);

    float sum = 0.0f;
    for (int co = 0; co < C_out; co++) {
        for (int kh = 0; kh < kH; kh++) {
            for (int kw = 0; kw < kW; kw++) {
                // Which output position did input[n][ci][ih][iw] contribute to
                // when multiplied by weight[co][ci][kh][kw]?
                // oh = ih + pad - kh, ow = iw + pad - kw
                int oh = ih + pad - kh;
                int ow = iw + pad - kw;
                if (oh >= 0 && oh < oH && ow >= 0 && ow < oW) {
                    float d_out_val = d_output[((n * C_out + co) * oH + oh) * oW + ow];
                    float w_val     = weight[((co * C_in + ci) * kH + kh) * kW + kw];
                    sum += d_out_val * w_val;
                }
            }
        }
    }
    atomicAdd(&d_input[idx], sum);
}

// conv2d backward w.r.t. weight (d_weight):
//
// d_weight[co][ci][kh][kw] = sum over n, oh, ow of:
//     d_output[n][co][oh][ow] * input[n][ci][oh - pad + kh][ow - pad + kw]
//
// This is a correlation between d_output and input.
// We use atomicAdd because multiple threads (one per output element)
// contribute to the same weight gradient.
__global__ void conv2d_backward_weight_kernel(
    const float* d_output,  // (N, C_out, oH, oW)
    const float* input,     // (N, C_in, H, W)
    float* d_weight,        // (C_out, C_in, kH, kW)  — accumulated
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int oH, int oW, int pad)
{
    // One thread per weight element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * kH * kW;
    if (idx >= total) return;

    int kw = idx % kW;
    int kh = (idx / kW) % kH;
    int ci = (idx / (kW * kH)) % C_in;
    int co = idx / (kW * kH * C_in);

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < oH; oh++) {
            for (int ow = 0; ow < oW; ow++) {
                int ih = oh - pad + kh;
                int iw = ow - pad + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float d_out_val = d_output[((n * C_out + co) * oH + oh) * oW + ow];
                    float in_val    = input[((n * C_in + ci) * H + ih) * W + iw];
                    sum += d_out_val * in_val;
                }
            }
        }
    }
    atomicAdd(&d_weight[idx], sum);
}


// ---- BatchNorm2d kernels ----
//
// From Chapter 14: BatchNorm normalizes each channel across the batch and
// spatial dimensions:
//
//   mu_c    = mean over (n, h, w) of x[n][c][h][w]
//   var_c   = variance over (n, h, w) of x[n][c][h][w]
//   x_hat   = (x - mu) / sqrt(var + eps)
//   output  = gamma * x_hat + beta
//
// During training: compute mu, var from current batch.
// During eval:     use running_mean, running_var.
//
// The backward pass is notoriously complex (Chapter 15):
//   d_x_hat = d_output * gamma
//   d_var   = sum(d_x_hat * (x - mu) * (-0.5) * (var + eps)^(-3/2))
//   d_mu    = sum(d_x_hat * (-1/sqrt(var + eps))) + d_var * (-2/M) * sum(x - mu)
//   d_x     = d_x_hat / sqrt(var + eps) + d_var * 2*(x - mu)/M + d_mu / M
//   d_gamma = sum(d_output * x_hat)
//   d_beta  = sum(d_output)
//
// where M = N * H * W (number of elements per channel being averaged over).

// Compute per-channel mean
// One thread per channel. Each thread loops over (N, H, W).
__global__ void batchnorm_mean_kernel(
    const float* input,   // (N, C, H, W)
    float* mean,          // (C,)
    int N, int C, int H, int W)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    int M = N * H * W;  // elements per channel
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += input[((n * C + c) * H + h) * W + w];
            }
        }
    }
    mean[c] = sum / M;
}

// Compute per-channel variance
__global__ void batchnorm_var_kernel(
    const float* input,   // (N, C, H, W)
    const float* mean,    // (C,)
    float* var,           // (C,)
    int N, int C, int H, int W)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float mu = mean[c];
    float sum = 0.0f;
    int M = N * H * W;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float diff = input[((n * C + c) * H + h) * W + w] - mu;
                sum += diff * diff;
            }
        }
    }
    var[c] = sum / M;
}

// BatchNorm forward: normalize and apply affine transform
// output = gamma * (input - mean) / sqrt(var + eps) + beta
__global__ void batchnorm_forward_kernel(
    const float* input,    // (N, C, H, W)
    const float* mean,     // (C,)
    const float* var,      // (C,)
    const float* gamma,    // (C,)
    const float* beta,     // (C,)
    float* output,         // (N, C, H, W)
    float* x_hat,          // (N, C, H, W)  — saved for backward
    int N, int C, int H, int W, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int c = (idx / (H * W)) % C;
    float mu = mean[c];
    float inv_std = 1.0f / sqrtf(var[c] + eps);
    float g = gamma[c];
    float b = beta[c];

    float xh = (input[idx] - mu) * inv_std;
    x_hat[idx] = xh;
    output[idx] = g * xh + b;
}

// Update running statistics (exponential moving average)
// running = (1 - momentum) * running + momentum * batch_stat
__global__ void batchnorm_update_running_kernel(
    float* running, const float* batch_stat,
    float momentum, int C)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    running[c] = (1.0f - momentum) * running[c] + momentum * batch_stat[c];
}

// BatchNorm backward: the full complex gradient computation
//
// This is the trickiest backward pass in the entire ResNet (Chapter 15).
//
// Given d_output (gradient from upstream):
//   d_gamma[c] = sum_{n,h,w} d_output[n][c][h][w] * x_hat[n][c][h][w]
//   d_beta[c]  = sum_{n,h,w} d_output[n][c][h][w]
//   d_x_hat    = d_output * gamma[c]
//   d_input    = (1/M) * inv_std * (M * d_x_hat - sum(d_x_hat) - x_hat * sum(d_x_hat * x_hat))
//
// The last formula is the efficient form that avoids computing d_var and d_mu separately.
// Reference: https://kevinzakka.github.io/2016/09/14/batch_normalization/

// Step 1: Compute d_gamma and d_beta (reductions over N, H, W)
__global__ void batchnorm_backward_params_kernel(
    const float* d_output,  // (N, C, H, W)
    const float* x_hat,     // (N, C, H, W)
    float* d_gamma,         // (C,)
    float* d_beta,          // (C,)
    int N, int C, int H, int W)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float dg = 0.0f, db = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = ((n * C + c) * H + h) * W + w;
                dg += d_output[idx] * x_hat[idx];
                db += d_output[idx];
            }
        }
    }
    atomicAdd(&d_gamma[c], dg);
    atomicAdd(&d_beta[c], db);
}

// Step 2: Compute d_input
// We first compute two per-channel sums, then use them element-wise.
__global__ void batchnorm_backward_sums_kernel(
    const float* d_output,  // (N, C, H, W)
    const float* gamma,     // (C,)
    const float* x_hat,     // (N, C, H, W)
    float* sum_dxhat,       // (C,) = sum of d_output * gamma
    float* sum_dxhat_xhat,  // (C,) = sum of d_output * gamma * x_hat
    int N, int C, int H, int W)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float g = gamma[c];
    float s1 = 0.0f, s2 = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = ((n * C + c) * H + h) * W + w;
                float dxh = d_output[idx] * g;
                s1 += dxh;
                s2 += dxh * x_hat[idx];
            }
        }
    }
    sum_dxhat[c] = s1;
    sum_dxhat_xhat[c] = s2;
}

__global__ void batchnorm_backward_input_kernel(
    const float* d_output,       // (N, C, H, W)
    const float* gamma,          // (C,)
    const float* x_hat,          // (N, C, H, W)
    const float* var,            // (C,)
    const float* sum_dxhat,      // (C,)
    const float* sum_dxhat_xhat, // (C,)
    float* d_input,              // (N, C, H, W)
    int N, int C, int H, int W, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int c = (idx / (H * W)) % C;
    float inv_std = 1.0f / sqrtf(var[c] + eps);
    float M = (float)(N * H * W);

    // d_x_hat for this element
    float dxh = d_output[idx] * gamma[c];

    // Efficient batch norm gradient:
    // d_input = (1/M) * inv_std * (M * d_x_hat - sum_dxhat - x_hat * sum_dxhat_xhat)
    float dinp = (1.0f / M) * inv_std * (M * dxh - sum_dxhat[c] - x_hat[idx] * sum_dxhat_xhat[c]);
    atomicAdd(&d_input[idx], dinp);
}


// ---- Global Average Pooling kernels ----
//
// From the ResNet architecture: after the body, we collapse spatial dims:
//   input:  (N, C, H, W)
//   output: (N, C)
//   output[n][c] = mean over (h, w) of input[n][c][h][w]
//
// Backward: d_input[n][c][h][w] = d_output[n][c] / (H * W)
//   The gradient is distributed UNIFORMLY across all spatial positions.
//   This is because each spatial position contributed equally to the mean.

__global__ void global_avg_pool_forward_kernel(
    const float* input,   // (N, C, H, W)
    float* output,        // (N, C)
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;

    int c = idx % C;
    int n = idx / C;
    int HW = H * W;

    float sum = 0.0f;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            sum += input[((n * C + c) * H + h) * W + w];
        }
    }
    output[idx] = sum / HW;
}

__global__ void global_avg_pool_backward_kernel(
    const float* d_output,  // (N, C)
    float* d_input,         // (N, C, H, W)
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // We only need channel and batch indices (not h, w) since the
    // gradient is uniform across all spatial positions in a channel.
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int HW = H * W;
    float grad_val = d_output[n * C + c] / (float)HW;
    atomicAdd(&d_input[idx], grad_val);
}


// ---- Linear (fully connected) kernels ----
//
// From Chapter 14: Linear layer = matrix multiplication + bias
//   output = input @ weight^T + bias
//
//   input:  (N, in_features)
//   weight: (out_features, in_features)
//   bias:   (out_features,) or nullptr
//   output: (N, out_features)
//
// Backward (Chapter 15):
//   d_input  = d_output @ weight          (N, out) x (out, in) = (N, in)
//   d_weight = d_output^T @ input         (out, N) x (N, in) = (out, in)
//   d_bias   = sum_over_N(d_output)       (out,)

__global__ void linear_forward_kernel(
    const float* input,   // (N, in_f)
    const float* weight,  // (out_f, in_f)
    const float* bias,    // (out_f,) or nullptr
    float* output,        // (N, out_f)
    int N, int in_f, int out_f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_f;
    if (idx >= total) return;

    int j = idx % out_f;   // output feature index
    int n = idx / out_f;   // batch index

    // output[n][j] = sum_i input[n][i] * weight[j][i] + bias[j]
    float sum = 0.0f;
    for (int i = 0; i < in_f; i++) {
        sum += input[n * in_f + i] * weight[j * in_f + i];
    }
    if (bias) sum += bias[j];
    output[idx] = sum;
}

// d_input = d_output @ weight
__global__ void linear_backward_input_kernel(
    const float* d_output,  // (N, out_f)
    const float* weight,    // (out_f, in_f)
    float* d_input,         // (N, in_f)
    int N, int in_f, int out_f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * in_f;
    if (idx >= total) return;

    int i = idx % in_f;
    int n = idx / in_f;

    float sum = 0.0f;
    for (int j = 0; j < out_f; j++) {
        sum += d_output[n * out_f + j] * weight[j * in_f + i];
    }
    atomicAdd(&d_input[idx], sum);
}

// d_weight = d_output^T @ input
__global__ void linear_backward_weight_kernel(
    const float* d_output,  // (N, out_f)
    const float* input,     // (N, in_f)
    float* d_weight,        // (out_f, in_f)
    int N, int in_f, int out_f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_f * in_f;
    if (idx >= total) return;

    int i = idx % in_f;
    int j = idx / in_f;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += d_output[n * out_f + j] * input[n * in_f + i];
    }
    atomicAdd(&d_weight[idx], sum);
}

// d_bias = sum_n d_output[n][j]
__global__ void linear_backward_bias_kernel(
    const float* d_output,  // (N, out_f)
    float* d_bias,          // (out_f,)
    int N, int out_f)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_f) return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += d_output[n * out_f + j];
    }
    atomicAdd(&d_bias[j], sum);
}


// ---- Cross-Entropy Loss kernels ----
//
// Cross-entropy = -log(softmax(logits)[target_class])
//
// From Chapter 14: we use the numerically stable version:
//   1. max_val = max(logits)                    (prevent overflow)
//   2. shifted = logits - max_val
//   3. exp_sum = sum(exp(shifted))
//   4. log_softmax = shifted - log(exp_sum)
//   5. loss = -log_softmax[target]
//
// Backward (Chapter 15):
//   d_logits = softmax_probs - one_hot(target)
//   This elegant formula is the reason cross-entropy is so popular:
//   the gradient is simply (predicted - actual), scaled by 1/N for mean.

// Compute per-sample loss
__global__ void cross_entropy_forward_kernel(
    const float* logits,   // (N, C)
    const int* targets,    // (N,)
    float* losses,         // (N,)
    int N, int C)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Find max for numerical stability
    float max_val = logits[n * C];
    for (int c = 1; c < C; c++) {
        float v = logits[n * C + c];
        if (v > max_val) max_val = v;
    }

    // Compute log-sum-exp
    float sum_exp = 0.0f;
    for (int c = 0; c < C; c++) {
        sum_exp += expf(logits[n * C + c] - max_val);
    }
    float log_sum_exp = logf(sum_exp) + max_val;

    // Loss = -logits[target] + log_sum_exp
    int t = targets[n];
    losses[n] = -logits[n * C + t] + log_sum_exp;
}

// Compute d_logits = (softmax - one_hot) / N
__global__ void cross_entropy_backward_kernel(
    const float* logits,    // (N, C)
    const int* targets,     // (N,)
    float* d_logits,        // (N, C)
    int N, int C)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Compute softmax for this sample
    float max_val = logits[n * C];
    for (int c = 1; c < C; c++) {
        float v = logits[n * C + c];
        if (v > max_val) max_val = v;
    }
    float sum_exp = 0.0f;
    for (int c = 0; c < C; c++) {
        sum_exp += expf(logits[n * C + c] - max_val);
    }

    int t = targets[n];
    for (int c = 0; c < C; c++) {
        float softmax_c = expf(logits[n * C + c] - max_val) / sum_exp;
        float one_hot = (c == t) ? 1.0f : 0.0f;
        // d_logits = (softmax - one_hot) / N
        // The /N gives us the mean loss gradient
        d_logits[n * C + c] = (softmax_c - one_hot) / (float)N;
    }
}

// Reduce per-sample losses to scalar mean
__global__ void reduce_mean_kernel(const float* losses, float* result, int N) {
    // Single thread — fine for small N (batch sizes up to ~256)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += losses[i];
    result[0] = sum / (float)N;
}


// ---- Adam optimizer kernel ----
//
// From Chapter 17: Adam combines momentum (first moment) with RMSProp
// (second moment) for adaptive learning rates per parameter.
//
//   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
//   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
//   m_hat = m_t / (1 - beta1^t)    (bias correction)
//   v_hat = v_t / (1 - beta2^t)    (bias correction)
//   theta = theta - lr * m_hat / (sqrt(v_hat) + eps) - weight_decay * lr * theta
//
// The weight_decay term implements decoupled weight decay (AdamW),
// which regularizes by shrinking weights directly rather than through
// the gradient. This is important for preventing overfitting in ResNets.
__global__ void adam_update_kernel(
    float* param,       // parameter to update
    const float* grad,  // gradient
    float* m,           // first moment
    float* v,           // second moment
    float lr,           // learning rate (after cosine schedule)
    float beta1, float beta2,
    float eps,
    float bias_corr1,   // 1 - beta1^t
    float bias_corr2,   // 1 - beta2^t
    float weight_decay,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = grad[idx];

    // Update moments
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    // Bias-corrected moments
    float m_hat = m[idx] / bias_corr1;
    float v_hat = v[idx] / bias_corr2;

    // Update parameter with AdamW-style weight decay
    param[idx] = param[idx] - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param[idx]);
}


// ============================================================================
// SECTION D: Autograd Wrappers
// ============================================================================
//
// From Chapter 16: each operation creates a new GradTensor with a backward_fn
// closure that captures the inputs. When backward() is called on the loss,
// it traverses the graph and calls these closures in reverse order.
//
// The closures capture raw pointers to the input tensors. This is safe as
// long as we don't delete inputs before calling backward(). The tensor
// arena (g_tensor_arena) keeps intermediate tensors alive.
// ============================================================================

inline int ceildiv(int a, int b) { return (a + b - 1) / b; }
const int BLOCK_SIZE = 256;

// ---- autograd_relu ----
// Creates output = relu(input) and stores backward_fn.
inline GradTensor* autograd_relu(GradTensor* input) {
    GradTensor* output = new GradTensor(input->shape);
    output->alloc_grad();  // intermediate nodes need grad for backprop
    g_tensor_arena.push_back(output);

    int n = input->size;
    relu_forward_kernel<<<ceildiv(n, BLOCK_SIZE), BLOCK_SIZE>>>(
        input->data, output->data, n);
    CUDA_CHECK(cudaGetLastError());

    // Store backward closure
    output->backward_fn = [input, output]() {
        input->alloc_grad();
        relu_backward_kernel<<<ceildiv(input->size, BLOCK_SIZE), BLOCK_SIZE>>>(
            input->data, output->grad, input->grad, input->size);
        CUDA_CHECK(cudaGetLastError());
    };
    output->children.push_back(input);
    return output;
}

// ---- autograd_add ----
// Creates output = a + b (element-wise).
// This is the RESIDUAL CONNECTION — the heart of ResNet.
// Backward: both inputs get a copy of the output gradient.
inline GradTensor* autograd_add(GradTensor* a, GradTensor* b) {
    assert(a->size == b->size);
    GradTensor* output = new GradTensor(a->shape);
    output->alloc_grad();
    g_tensor_arena.push_back(output);

    int n = a->size;
    add_kernel<<<ceildiv(n, BLOCK_SIZE), BLOCK_SIZE>>>(
        a->data, b->data, output->data, n);
    CUDA_CHECK(cudaGetLastError());

    output->backward_fn = [a, b, output]() {
        int n = a->size;
        a->alloc_grad();
        add_grad_kernel<<<ceildiv(n, BLOCK_SIZE), BLOCK_SIZE>>>(
            a->grad, output->grad, n);
        CUDA_CHECK(cudaGetLastError());
        b->alloc_grad();
        add_grad_kernel<<<ceildiv(n, BLOCK_SIZE), BLOCK_SIZE>>>(
            b->grad, output->grad, n);
        CUDA_CHECK(cudaGetLastError());
    };
    output->children.push_back(a);
    output->children.push_back(b);
    return output;
}

// ---- autograd_conv2d ----
// output = conv2d(input, weight) with given padding.
// No bias (all Conv2d layers in ResNet use bias=False).
inline GradTensor* autograd_conv2d(GradTensor* input, GradTensor* weight,
                                   int pad) {
    int N = input->shape[0];
    int C_in = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int C_out = weight->shape[0];
    int kH = weight->shape[2];
    int kW = weight->shape[3];
    int oH = H + 2 * pad - kH + 1;
    int oW = W + 2 * pad - kW + 1;

    GradTensor* output = new GradTensor({N, C_out, oH, oW});
    output->alloc_grad();
    g_tensor_arena.push_back(output);

    int total_out = N * C_out * oH * oW;
    conv2d_forward_kernel<<<ceildiv(total_out, BLOCK_SIZE), BLOCK_SIZE>>>(
        input->data, weight->data, output->data,
        N, C_in, H, W, C_out, kH, kW, oH, oW, pad);
    CUDA_CHECK(cudaGetLastError());

    output->backward_fn = [input, weight, output, N, C_in, H, W, C_out, kH, kW, oH, oW, pad]() {
        // Backward w.r.t. input
        input->alloc_grad();
        int total_in = N * C_in * H * W;
        conv2d_backward_input_kernel<<<ceildiv(total_in, BLOCK_SIZE), BLOCK_SIZE>>>(
            output->grad, weight->data, input->grad,
            N, C_in, H, W, C_out, kH, kW, oH, oW, pad);
        CUDA_CHECK(cudaGetLastError());

        // Backward w.r.t. weight (always has grad since it's a parameter)
        int total_w = C_out * C_in * kH * kW;
        conv2d_backward_weight_kernel<<<ceildiv(total_w, BLOCK_SIZE), BLOCK_SIZE>>>(
            output->grad, input->data, weight->grad,
            N, C_in, H, W, C_out, kH, kW, oH, oW, pad);
        CUDA_CHECK(cudaGetLastError());
    };
    output->children.push_back(input);
    output->children.push_back(weight);
    return output;
}

// ---- autograd_batchnorm2d ----
// Forward: normalize, scale, shift. Stores x_hat for backward.
// Updates running stats if training=true.
inline GradTensor* autograd_batchnorm2d(
    GradTensor* input, GradTensor* gamma, GradTensor* beta,
    float* running_mean, float* running_var,
    bool training, float momentum, float eps)
{
    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int total = N * C * H * W;

    GradTensor* output = new GradTensor(input->shape);
    output->alloc_grad();
    g_tensor_arena.push_back(output);

    // Allocate temporaries for mean, var, x_hat on GPU
    float* d_mean = gpu_malloc(C);
    float* d_var = gpu_malloc(C);
    float* d_x_hat = gpu_malloc(total);

    if (training) {
        // Compute batch mean and variance
        int blocks_c = ceildiv(C, BLOCK_SIZE);
        batchnorm_mean_kernel<<<blocks_c, BLOCK_SIZE>>>(input->data, d_mean, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
        batchnorm_var_kernel<<<blocks_c, BLOCK_SIZE>>>(input->data, d_mean, d_var, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());

        // Update running statistics
        batchnorm_update_running_kernel<<<blocks_c, BLOCK_SIZE>>>(
            running_mean, d_mean, momentum, C);
        CUDA_CHECK(cudaGetLastError());
        batchnorm_update_running_kernel<<<blocks_c, BLOCK_SIZE>>>(
            running_var, d_var, momentum, C);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Use running stats in eval mode
        gpu_copy_d2d(d_mean, running_mean, C);
        gpu_copy_d2d(d_var, running_var, C);
    }

    // Forward pass: normalize and affine
    batchnorm_forward_kernel<<<ceildiv(total, BLOCK_SIZE), BLOCK_SIZE>>>(
        input->data, d_mean, d_var, gamma->data, beta->data,
        output->data, d_x_hat, N, C, H, W, eps);
    CUDA_CHECK(cudaGetLastError());

    // Capture mean, var, x_hat for backward (they'll be freed after backward)
    output->backward_fn = [input, gamma, beta, output, d_mean, d_var, d_x_hat,
                           N, C, H, W, eps]() {
        int total = N * C * H * W;
        int blocks_c = ceildiv(C, BLOCK_SIZE);

        // d_gamma, d_beta (always have grad since they're parameters)
        batchnorm_backward_params_kernel<<<blocks_c, BLOCK_SIZE>>>(
            output->grad, d_x_hat, gamma->grad, beta->grad, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());

        // d_input
        input->alloc_grad();
        float* sum_dxhat = gpu_malloc(C);
        float* sum_dxhat_xhat = gpu_malloc(C);
        gpu_memset_zero(sum_dxhat, C);
        gpu_memset_zero(sum_dxhat_xhat, C);

        batchnorm_backward_sums_kernel<<<blocks_c, BLOCK_SIZE>>>(
            output->grad, gamma->data, d_x_hat,
            sum_dxhat, sum_dxhat_xhat, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());

        batchnorm_backward_input_kernel<<<ceildiv(total, BLOCK_SIZE), BLOCK_SIZE>>>(
            output->grad, gamma->data, d_x_hat, d_var,
            sum_dxhat, sum_dxhat_xhat, input->grad, N, C, H, W, eps);
        CUDA_CHECK(cudaGetLastError());

        gpu_free(sum_dxhat);
        gpu_free(sum_dxhat_xhat);

        // Free saved tensors
        gpu_free(d_mean);
        gpu_free(d_var);
        gpu_free(d_x_hat);
    };
    output->children.push_back(input);
    output->children.push_back(gamma);
    output->children.push_back(beta);
    return output;
}

// ---- autograd_global_avg_pool ----
// (N, C, H, W) -> (N, C)
inline GradTensor* autograd_global_avg_pool(GradTensor* input) {
    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];

    GradTensor* output = new GradTensor({N, C});
    output->alloc_grad();
    g_tensor_arena.push_back(output);

    int total_out = N * C;
    global_avg_pool_forward_kernel<<<ceildiv(total_out, BLOCK_SIZE), BLOCK_SIZE>>>(
        input->data, output->data, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());

    output->backward_fn = [input, output, N, C, H, W]() {
        input->alloc_grad();
        int total_in = N * C * H * W;
        global_avg_pool_backward_kernel<<<ceildiv(total_in, BLOCK_SIZE), BLOCK_SIZE>>>(
            output->grad, input->grad, N, C, H, W);
        CUDA_CHECK(cudaGetLastError());
    };
    output->children.push_back(input);
    return output;
}

// ---- autograd_linear ----
// output = input @ weight^T + bias
inline GradTensor* autograd_linear(GradTensor* input, GradTensor* weight,
                                   GradTensor* bias) {
    int N = input->shape[0];
    int in_f = input->shape[1];
    int out_f = weight->shape[0];

    GradTensor* output = new GradTensor({N, out_f});
    output->alloc_grad();
    g_tensor_arena.push_back(output);

    int total = N * out_f;
    linear_forward_kernel<<<ceildiv(total, BLOCK_SIZE), BLOCK_SIZE>>>(
        input->data, weight->data, bias ? bias->data : nullptr,
        output->data, N, in_f, out_f);
    CUDA_CHECK(cudaGetLastError());

    output->backward_fn = [input, weight, bias, output, N, in_f, out_f]() {
        // d_input
        input->alloc_grad();
        int total_in = N * in_f;
        linear_backward_input_kernel<<<ceildiv(total_in, BLOCK_SIZE), BLOCK_SIZE>>>(
            output->grad, weight->data, input->grad, N, in_f, out_f);
        CUDA_CHECK(cudaGetLastError());

        // d_weight (always has grad since it's a parameter)
        int total_w = out_f * in_f;
        linear_backward_weight_kernel<<<ceildiv(total_w, BLOCK_SIZE), BLOCK_SIZE>>>(
            output->grad, input->data, weight->grad, N, in_f, out_f);
        CUDA_CHECK(cudaGetLastError());

        // d_bias
        if (bias && bias->grad) {
            linear_backward_bias_kernel<<<ceildiv(out_f, BLOCK_SIZE), BLOCK_SIZE>>>(
                output->grad, bias->grad, N, out_f);
            CUDA_CHECK(cudaGetLastError());
        }
    };
    output->children.push_back(input);
    output->children.push_back(weight);
    if (bias) output->children.push_back(bias);
    return output;
}


// ============================================================================
// SECTION E: Module Base Class
// ============================================================================
//
// From Chapter 17: the Module class provides:
//   - parameters(): collect all learnable parameters
//   - train()/eval(): switch between training and evaluation modes
//   - Subclasses override forward() to define computation
//
// This mirrors PyTorch's nn.Module. The key insight is that parameters()
// collects recursively from sub-modules, so ResNet.parameters() gives us
// all parameters in the entire network.
// ============================================================================

class Module {
public:
    bool training_ = true;

    virtual ~Module() {}

    // Collect all learnable parameters from this module and sub-modules
    virtual std::vector<GradTensor*> parameters() { return {}; }

    // Switch to training mode (batch norm uses batch stats)
    virtual void train() { training_ = true; }

    // Switch to evaluation mode (batch norm uses running stats)
    virtual void eval() { training_ = false; }
};


// ============================================================================
// SECTION F: Layer Classes
// ============================================================================
//
// Each layer owns its parameters (GradTensor*) and implements forward().
// The forward() method calls the autograd wrapper, which builds the
// computational graph automatically.
//
// Parallels to cnn_resnet.py:
//   nn.Conv2d      -> Conv2dLayer
//   nn.BatchNorm2d -> BatchNorm2dLayer
//   F.relu         -> ReLULayer
//   nn.Linear      -> LinearLayer
//   x.mean([2,3])  -> GlobalAvgPool2dLayer
// ============================================================================

// ---- Conv2dLayer ----
// Matches nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=False)
//
// No bias: standard for conv layers before batch norm, because BN's
// beta parameter subsumes the bias.
//
// Initialization: Kaiming (He) normal
//   weight ~ N(0, sqrt(2 / (C_in * kH * kW)))
// This ensures activations don't explode or vanish in deep networks
// (Chapter 17: initialization matters).
class Conv2dLayer : public Module {
public:
    GradTensor* weight;  // (C_out, C_in, kH, kW)
    int in_channels, out_channels, kernel_size, padding;

    Conv2dLayer(int in_ch, int out_ch, int ksize, int pad = 0)
        : in_channels(in_ch), out_channels(out_ch),
          kernel_size(ksize), padding(pad)
    {
        // Kaiming initialization: std = sqrt(2 / fan_in)
        int fan_in = in_ch * ksize * ksize;
        float std_dev = sqrtf(2.0f / fan_in);
        weight = GradTensor::randn({out_ch, in_ch, ksize, ksize}, std_dev, true);
    }

    ~Conv2dLayer() { delete weight; }

    GradTensor* forward(GradTensor* input) {
        return autograd_conv2d(input, weight, padding);
    }

    std::vector<GradTensor*> parameters() override {
        return {weight};
    }
};

// ---- BatchNorm2dLayer ----
// Matches nn.BatchNorm2d(num_features)
//
// Parameters: gamma (scale, init=1), beta (shift, init=0)
// Buffers: running_mean (init=0), running_var (init=1)
// Hyperparameters: momentum=0.1, eps=1e-5
//
// In training mode: uses batch statistics and updates running stats.
// In eval mode: uses running statistics (no batch dependency).
class BatchNorm2dLayer : public Module {
public:
    GradTensor* gamma;   // (C,) — learnable scale
    GradTensor* beta;    // (C,) — learnable shift
    float* running_mean; // (C,) — GPU buffer, not a parameter
    float* running_var;  // (C,) — GPU buffer, not a parameter
    int num_features;
    float momentum;
    float eps;

    BatchNorm2dLayer(int nf, float mom = 0.1f, float e = 1e-5f)
        : num_features(nf), momentum(mom), eps(e)
    {
        // gamma = 1 (identity scale)
        gamma = GradTensor::ones({nf}, true);
        // beta = 0 (no shift)
        beta = GradTensor::zeros({nf}, true);
        // running_mean = 0
        running_mean = gpu_malloc(nf);
        gpu_memset_zero(running_mean, nf);
        // running_var = 1
        running_var = gpu_malloc(nf);
        std::vector<float> ones(nf, 1.0f);
        gpu_copy_h2d(running_var, ones.data(), nf);
    }

    ~BatchNorm2dLayer() {
        delete gamma;
        delete beta;
        gpu_free(running_mean);
        gpu_free(running_var);
    }

    GradTensor* forward(GradTensor* input) {
        return autograd_batchnorm2d(input, gamma, beta,
                                    running_mean, running_var,
                                    training_, momentum, eps);
    }

    std::vector<GradTensor*> parameters() override {
        return {gamma, beta};
    }

    void train() override { training_ = true; }
    void eval() override { training_ = false; }
};

// ---- ReLULayer ----
// Matches F.relu in cnn_resnet.py
class ReLULayer : public Module {
public:
    GradTensor* forward(GradTensor* input) {
        return autograd_relu(input);
    }
    // No parameters
};

// ---- LinearLayer ----
// Matches nn.Linear(in_features, out_features)
//
// Weight: Kaiming init (same reason as Conv2d)
// Bias: initialized to 0 (standard practice)
class LinearLayer : public Module {
public:
    GradTensor* weight;  // (out_f, in_f)
    GradTensor* bias;    // (out_f,) or nullptr
    int in_features, out_features;
    bool use_bias;

    LinearLayer(int in_f, int out_f, bool bias_ = true)
        : in_features(in_f), out_features(out_f), use_bias(bias_)
    {
        float std_dev = sqrtf(2.0f / in_f);
        weight = GradTensor::randn({out_f, in_f}, std_dev, true);
        if (use_bias) {
            bias = GradTensor::zeros({out_f}, true);
        } else {
            bias = nullptr;
        }
    }

    ~LinearLayer() {
        delete weight;
        if (bias) delete bias;
    }

    GradTensor* forward(GradTensor* input) {
        return autograd_linear(input, weight, bias);
    }

    std::vector<GradTensor*> parameters() override {
        std::vector<GradTensor*> params = {weight};
        if (bias) params.push_back(bias);
        return params;
    }
};

// ---- GlobalAvgPool2dLayer ----
// Matches x.mean(dim=[2, 3]) in cnn_resnet.py
// Collapses (N, C, H, W) -> (N, C) by averaging over spatial dims.
class GlobalAvgPool2dLayer : public Module {
public:
    GradTensor* forward(GradTensor* input) {
        return autograd_global_avg_pool(input);
    }
    // No parameters
};


// ============================================================================
// SECTION G: ResBlock (Pre-Activation ResNet-v2)
// ============================================================================
//
// Matching cnn_resnet.py exactly:
//
//   class ResBlock(nn.Module):
//       def __init__(self, channels):
//           self.bn1 = nn.BatchNorm2d(channels)
//           self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
//           self.bn2 = nn.BatchNorm2d(channels)
//           self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
//
//       def forward(self, x):
//           residual = x
//           x = self.conv1(F.relu(self.bn1(x)))
//           x = self.conv2(F.relu(self.bn2(x)))
//           return x + residual
//
// Key observations (from Chapter 19 README):
// 1. Pre-activation: BN and ReLU come BEFORE the conv, not after.
// 2. Identity skip: no projection needed because C_in == C_out and
//    padding preserves spatial dimensions.
// 3. The identity skip creates a gradient highway — gradients flow
//    unchanged through the addition, preventing vanishing gradients
//    even in very deep networks.
// ============================================================================

class ResBlock : public Module {
public:
    BatchNorm2dLayer* bn1;
    ReLULayer* relu1;
    Conv2dLayer* conv1;
    BatchNorm2dLayer* bn2;
    ReLULayer* relu2;
    Conv2dLayer* conv2;

    ResBlock(int channels) {
        bn1   = new BatchNorm2dLayer(channels);
        relu1 = new ReLULayer();
        conv1 = new Conv2dLayer(channels, channels, 3, 1);  // 3x3, pad=1
        bn2   = new BatchNorm2dLayer(channels);
        relu2 = new ReLULayer();
        conv2 = new Conv2dLayer(channels, channels, 3, 1);  // 3x3, pad=1
    }

    ~ResBlock() {
        delete bn1; delete relu1; delete conv1;
        delete bn2; delete relu2; delete conv2;
    }

    // forward: exactly mirrors cnn_resnet.py
    //   residual = x
    //   x = conv1(relu(bn1(x)))
    //   x = conv2(relu(bn2(x)))
    //   return x + residual
    GradTensor* forward(GradTensor* x) {
        GradTensor* residual = x;                           // identity skip

        x = bn1->forward(x);      // BatchNorm2d
        x = relu1->forward(x);    // ReLU
        x = conv1->forward(x);    // Conv2d(C -> C, 3x3, pad=1)

        x = bn2->forward(x);      // BatchNorm2d
        x = relu2->forward(x);    // ReLU
        x = conv2->forward(x);    // Conv2d(C -> C, 3x3, pad=1)

        return autograd_add(x, residual);  // x + residual (gradient highway!)
    }

    std::vector<GradTensor*> parameters() override {
        std::vector<GradTensor*> params;
        // Collect from bn1, conv1, bn2, conv2
        // (relu has no parameters)
        auto p1 = bn1->parameters();   params.insert(params.end(), p1.begin(), p1.end());
        auto p2 = conv1->parameters(); params.insert(params.end(), p2.begin(), p2.end());
        auto p3 = bn2->parameters();   params.insert(params.end(), p3.begin(), p3.end());
        auto p4 = conv2->parameters(); params.insert(params.end(), p4.begin(), p4.end());
        return params;
    }

    void train() override {
        training_ = true;
        bn1->train(); bn2->train();
    }

    void eval() override {
        training_ = false;
        bn1->eval(); bn2->eval();
    }
};


// ============================================================================
// SECTION H: ResNet Class
// ============================================================================
//
// Matching cnn_resnet.py:
//
//   class ResNet(nn.Module):
//       def __init__(self, channels, nb_blocks, fc_size=256, in_channels=4):
//           self.conv = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
//           self.bn = nn.BatchNorm2d(channels)
//           self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(nb_blocks)])
//           self.head_bn = nn.BatchNorm2d(channels)
//           self.fc1 = nn.Linear(channels, fc_size)
//           self.fc2 = nn.Linear(fc_size, 3)
//
//       def forward(self, x):
//           x = F.relu(self.bn(self.conv(x)))             # Stem
//           for block in self.blocks:                      # Body
//               x = block(x)
//           x = F.relu(self.head_bn(x))                   # Head: BN + ReLU
//           x = x.mean(dim=[2, 3])                        # Head: GAP
//           x = F.relu(self.fc1(x))                       # Head: FC1 + ReLU
//           x = self.fc2(x)                               # Head: FC2
//           return x
//
// Architecture summary:
//   STEM: Conv(4->C) -> BN -> ReLU
//   BODY: N x ResBlock(C)
//   HEAD: BN -> ReLU -> GAP -> FC(C->fc) -> ReLU -> FC(fc->3)
// ============================================================================

class ResNet : public Module {
public:
    // Stem layers
    Conv2dLayer* stem_conv;        // Conv2d(in_channels -> channels, 3x3, pad=1)
    BatchNorm2dLayer* stem_bn;     // BatchNorm2d(channels)
    ReLULayer* stem_relu;          // ReLU

    // Body: vector of ResBlocks
    std::vector<ResBlock*> blocks;

    // Head layers
    BatchNorm2dLayer* head_bn;     // BatchNorm2d(channels) — final normalization
    ReLULayer* head_relu;          // ReLU
    GlobalAvgPool2dLayer* gap;     // GlobalAvgPool2d: (B,C,H,W) -> (B,C)
    LinearLayer* fc1;              // Linear(channels -> fc_size)
    ReLULayer* fc1_relu;           // ReLU
    LinearLayer* fc2;              // Linear(fc_size -> num_classes)

    int channels_, nb_blocks_, fc_size_, num_classes_, in_channels_;

    // Constructor: mirrors cnn_resnet.py's __init__
    ResNet(int channels, int nb_blocks, int fc_size = 256,
           int num_classes = 3, int in_channels = 4)
        : channels_(channels), nb_blocks_(nb_blocks),
          fc_size_(fc_size), num_classes_(num_classes),
          in_channels_(in_channels)
    {
        // STEM: Conv2d(4 -> channels, 3x3, pad=1, no bias) + BN + ReLU
        stem_conv = new Conv2dLayer(in_channels, channels, 3, 1);
        stem_bn   = new BatchNorm2dLayer(channels);
        stem_relu = new ReLULayer();

        // BODY: nb_blocks x ResBlock(channels)
        for (int i = 0; i < nb_blocks; i++) {
            blocks.push_back(new ResBlock(channels));
        }

        // HEAD: BN -> ReLU -> GAP -> FC1 -> ReLU -> FC2
        head_bn   = new BatchNorm2dLayer(channels);
        head_relu = new ReLULayer();
        gap       = new GlobalAvgPool2dLayer();
        fc1       = new LinearLayer(channels, fc_size, true);
        fc1_relu  = new ReLULayer();
        fc2       = new LinearLayer(fc_size, num_classes, true);
    }

    ~ResNet() {
        delete stem_conv; delete stem_bn; delete stem_relu;
        for (auto* b : blocks) delete b;
        delete head_bn; delete head_relu; delete gap;
        delete fc1; delete fc1_relu; delete fc2;
    }

    // forward: mirrors cnn_resnet.py's forward method
    GradTensor* forward(GradTensor* x) {
        // STEM: x = relu(bn(conv(x)))
        // Transforms (B, 4, 8, 8) -> (B, channels, 8, 8)
        x = stem_conv->forward(x);   // Conv2d(4 -> C, 3x3, pad=1)
        x = stem_bn->forward(x);     // BatchNorm2d(C)
        x = stem_relu->forward(x);   // ReLU

        // BODY: for block in blocks: x = block(x)
        // Each block: (B, C, 8, 8) -> (B, C, 8, 8)
        for (auto* block : blocks) {
            x = block->forward(x);
        }

        // HEAD:
        x = head_bn->forward(x);     // BatchNorm2d(C)
        x = head_relu->forward(x);   // ReLU
        x = gap->forward(x);         // GAP: (B, C, 8, 8) -> (B, C)
        x = fc1->forward(x);         // Linear(C -> fc_size)
        x = fc1_relu->forward(x);    // ReLU
        x = fc2->forward(x);         // Linear(fc_size -> num_classes)

        return x;  // (B, num_classes)
    }

    // parameters: collect from ALL sub-modules (recursively)
    std::vector<GradTensor*> parameters() override {
        std::vector<GradTensor*> params;
        auto add_params = [&params](std::vector<GradTensor*> p) {
            params.insert(params.end(), p.begin(), p.end());
        };

        add_params(stem_conv->parameters());   // stem conv weight
        add_params(stem_bn->parameters());     // stem BN gamma, beta
        for (auto* block : blocks) {
            add_params(block->parameters());   // each block's params
        }
        add_params(head_bn->parameters());     // head BN gamma, beta
        add_params(fc1->parameters());         // fc1 weight, bias
        add_params(fc2->parameters());         // fc2 weight, bias
        return params;
    }

    // Count total parameters
    int parameter_count() {
        int count = 0;
        for (auto* p : parameters()) {
            count += p->size;
        }
        return count;
    }

    void train() override {
        training_ = true;
        stem_bn->train();
        for (auto* b : blocks) b->train();
        head_bn->train();
    }

    void eval() override {
        training_ = false;
        stem_bn->eval();
        for (auto* b : blocks) b->eval();
        head_bn->eval();
    }
};


// ============================================================================
// SECTION I: Adam Optimizer + CosineAnnealingLR
// ============================================================================
//
// From Chapter 17: Adam is the default optimizer for deep learning.
// We implement AdamW (decoupled weight decay) because it works better
// with learning rate scheduling (the weight decay is independent of lr).
//
// CosineAnnealingLR: smoothly decays the learning rate following a cosine
// curve from lr_max to eta_min over T_max steps. This helps fine-tune
// the network in later epochs when we want smaller updates.
//
//   lr(t) = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(pi * t / T_max))
// ============================================================================

class AdamOptimizer {
public:
    std::vector<GradTensor*> params;
    std::vector<float*> m_bufs;   // first moment (GPU)
    std::vector<float*> v_bufs;   // second moment (GPU)

    float lr, beta1, beta2, eps, weight_decay;
    int step_count;

    AdamOptimizer(std::vector<GradTensor*> params_,
                  float lr_ = 0.001f,
                  float beta1_ = 0.9f, float beta2_ = 0.999f,
                  float eps_ = 1e-8f, float weight_decay_ = 0.0f)
        : params(params_), lr(lr_), beta1(beta1_), beta2(beta2_),
          eps(eps_), weight_decay(weight_decay_), step_count(0)
    {
        // Allocate moment buffers (initialized to zero)
        for (auto* p : params) {
            float* m = gpu_malloc(p->size);
            float* v = gpu_malloc(p->size);
            gpu_memset_zero(m, p->size);
            gpu_memset_zero(v, p->size);
            m_bufs.push_back(m);
            v_bufs.push_back(v);
        }
    }

    ~AdamOptimizer() {
        for (auto* m : m_bufs) gpu_free(m);
        for (auto* v : v_bufs) gpu_free(v);
    }

    // step: update all parameters using their gradients
    void step() {
        step_count++;
        float bias_corr1 = 1.0f - powf(beta1, (float)step_count);
        float bias_corr2 = 1.0f - powf(beta2, (float)step_count);

        for (int i = 0; i < (int)params.size(); i++) {
            GradTensor* p = params[i];
            if (!p->grad) continue;

            int n = p->size;
            adam_update_kernel<<<ceildiv(n, BLOCK_SIZE), BLOCK_SIZE>>>(
                p->data, p->grad, m_bufs[i], v_bufs[i],
                lr, beta1, beta2, eps, bias_corr1, bias_corr2,
                weight_decay, n);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // zero_grad: reset all parameter gradients to zero
    void zero_grad() {
        for (auto* p : params) {
            p->zero_grad();
        }
    }
};

// CosineAnnealingLR: adjusts the learning rate of an Adam optimizer
// following a cosine decay schedule.
//
// lr(epoch) = eta_min + 0.5 * (lr_init - eta_min) * (1 + cos(pi * epoch / T_max))
//
// This gives a smooth decay from lr_init to eta_min over T_max epochs,
// which often works better than step-based schedules for ResNet training.
class CosineAnnealingLR {
public:
    AdamOptimizer* optimizer;
    float lr_init;
    float eta_min;
    int T_max;

    CosineAnnealingLR(AdamOptimizer* opt, int T_max_, float eta_min_ = 0.0f)
        : optimizer(opt), lr_init(opt->lr), eta_min(eta_min_), T_max(T_max_)
    {}

    // Call at the start of each epoch to update the learning rate
    void step(int epoch) {
        float lr = eta_min + 0.5f * (lr_init - eta_min) *
                   (1.0f + cosf((float)epoch * M_PI / (float)T_max));
        optimizer->lr = lr;
    }

    float get_lr() const { return optimizer->lr; }
};


// ============================================================================
// SECTION J: CrossEntropyLoss
// ============================================================================
//
// Combines softmax and negative log-likelihood in one operation.
// This is more numerically stable than computing softmax and log separately
// (Chapter 14: the log-sum-exp trick prevents overflow).
//
// Forward:  loss = -log(softmax(logits)[target])
// Backward: d_logits = (softmax(logits) - one_hot(target)) / N
//
// The division by N gives us the MEAN loss (matching PyTorch's default
// reduction='mean').
// ============================================================================

class CrossEntropyLoss {
public:
    // Compute the loss and set up backward pass.
    // Returns a scalar GradTensor (shape {1}) containing the mean loss.
    //
    // logits:  (N, num_classes) — raw network output
    // targets: host vector of int labels, length N
    GradTensor* forward(GradTensor* logits, const std::vector<int>& targets) {
        int N = logits->shape[0];
        int C = logits->shape[1];

        // Copy targets to GPU
        int* d_targets;
        CUDA_CHECK(cudaMalloc(&d_targets, N * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_targets, targets.data(), N * sizeof(int),
                              cudaMemcpyHostToDevice));

        // Compute per-sample losses
        float* d_losses = gpu_malloc(N);
        cross_entropy_forward_kernel<<<ceildiv(N, BLOCK_SIZE), BLOCK_SIZE>>>(
            logits->data, d_targets, d_losses, N, C);
        CUDA_CHECK(cudaGetLastError());

        // Reduce to scalar mean
        GradTensor* loss = new GradTensor({1});
        g_tensor_arena.push_back(loss);
        reduce_mean_kernel<<<1, 1>>>(d_losses, loss->data, N);
        CUDA_CHECK(cudaGetLastError());

        // Set up backward: compute d_logits
        loss->backward_fn = [logits, d_targets, d_losses, N, C, loss]() {
            logits->alloc_grad();
            cross_entropy_backward_kernel<<<ceildiv(N, BLOCK_SIZE), BLOCK_SIZE>>>(
                logits->data, d_targets, logits->grad, N, C);
            CUDA_CHECK(cudaGetLastError());

            // Free temporaries (but not d_targets — kept for backward)
            gpu_free(d_losses);
            CUDA_CHECK(cudaFree(d_targets));
        };
        loss->children.push_back(logits);
        return loss;
    }

    // Convenience: compute loss value as a float (for printing)
    static float get_loss_value(GradTensor* loss) {
        float val;
        gpu_copy_d2h(&val, loss->data, 1);
        return val;
    }
};


// ============================================================================
// SECTION K: Utility Functions
// ============================================================================

// Create an input GradTensor from host data.
// The input tensor does NOT require grad (it's the data, not a parameter),
// but it needs a grad buffer allocated so upstream layers can propagate
// gradients through it.
inline GradTensor* make_input(const std::vector<float>& data,
                              const std::vector<int>& shape) {
    GradTensor* t = new GradTensor(shape, true);
    t->from_host(data);
    return t;
}

// Compute accuracy: compare argmax(logits) with targets
inline float compute_accuracy(GradTensor* logits, const std::vector<int>& targets) {
    int N = logits->shape[0];
    int C = logits->shape[1];
    std::vector<float> h = logits->to_host();

    int correct = 0;
    for (int n = 0; n < N; n++) {
        int pred = 0;
        float max_val = h[n * C];
        for (int c = 1; c < C; c++) {
            if (h[n * C + c] > max_val) {
                max_val = h[n * C + c];
                pred = c;
            }
        }
        if (pred == targets[n]) correct++;
    }
    return (float)correct / N;
}

// Simple random number generator for host-side data generation
// (not for GPU — GPU uses cuRAND)
inline float host_randn() {
    // Box-Muller transform
    static bool has_spare = false;
    static float spare;
    if (has_spare) { has_spare = false; return spare; }
    float u, v, s;
    do {
        u = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        v = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    has_spare = true;
    return u * s;
}


#endif // RESNET_CUH
