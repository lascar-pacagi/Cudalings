/*******************************************************************************
 * module.cuh — Module Base Class for cudalearn
 *
 * This is the heart of the library's OOP design. Every neural network layer
 * inherits from Module. The key responsibilities:
 *
 *   1. forward()      — defines the computation (pure virtual)
 *   2. parameters()   — collects all learnable GradTensors (recursive)
 *   3. train()/set_eval() — switches mode (affects BatchNorm, Dropout, etc.)
 *   4. print()        — shows architecture summary with param counts
 *
 * Design mirrors PyTorch's nn.Module:
 *   - Sub-modules are registered with register_module()
 *   - Parameters are registered with register_parameter()
 *   - Buffers (non-learnable state) are just raw float* managed by the layer
 *
 * SELF-CONTAINED: This file defines the GradTensor struct inline so the
 * chapter is compilable without external dependencies.
 ******************************************************************************/

#ifndef CUDALEARN_MODULE_CUH
#define CUDALEARN_MODULE_CUH

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <functional>
#include <memory>

// =============================================================================
// GradTensor: A GPU tensor with optional gradient tracking
// =============================================================================
// This is the fundamental data type. It wraps:
//   - data:  GPU float array holding the tensor values
//   - grad:  GPU float array holding gradients (same size as data)
//   - shape: dimensions of the tensor (up to 4D: N, C, H, W)
//   - requires_grad: whether this tensor participates in autograd
//
// The backward() method triggers reverse-mode autodiff through the
// computation graph built during forward pass.
// =============================================================================

// Forward declaration for the autograd backward function type.
// Each operation stores a lambda that computes gradients w.r.t. its inputs.
struct GradTensor;

// A backward function takes no arguments and returns void.
// It reads from the GradTensor's grad field and propagates to parents.
using BackwardFn = std::function<void()>;

struct GradTensor {
    float* data;            // GPU memory for tensor values
    float* grad;            // GPU memory for gradients (nullptr if not needed)
    int dims[4];            // Shape: [N, C, H, W] — unused dims are 1
    int ndim;               // Number of meaningful dimensions (1-4)
    int size;               // Total number of elements (product of dims)
    bool requires_grad;     // Does this tensor need gradients?

    // ---------- Autograd graph ----------
    // Parents are the tensors that were inputs to the operation that created
    // this tensor. The backward_fn computes d(this)/d(parent) * this->grad
    // and accumulates into parent->grad.
    std::vector<GradTensor*> parents;
    BackwardFn backward_fn; // Function to propagate gradients to parents

    // Constructor: allocate GPU memory for data (and grad if needed)
    GradTensor(int d0, int d1 = 1, int d2 = 1, int d3 = 1,
               bool req_grad = false)
        : data(nullptr), grad(nullptr), requires_grad(req_grad),
          backward_fn(nullptr)
    {
        dims[0] = d0; dims[1] = d1; dims[2] = d2; dims[3] = d3;

        // Determine number of meaningful dimensions
        if (d3 > 1)      ndim = 4;
        else if (d2 > 1) ndim = 3;
        else if (d1 > 1) ndim = 2;
        else              ndim = 1;

        size = d0 * d1 * d2 * d3;

        // Allocate GPU memory for data
        cudaMalloc(&data, size * sizeof(float));
        cudaMemset(data, 0, size * sizeof(float));

        // Allocate gradient storage if this tensor requires gradients
        if (requires_grad) {
            cudaMalloc(&grad, size * sizeof(float));
            cudaMemset(grad, 0, size * sizeof(float));
        }
    }

    // Destructor: free GPU memory
    ~GradTensor() {
        if (data) cudaFree(data);
        if (grad) cudaFree(grad);
    }

    // Zero out the gradient (called by optimizer.zero_grad())
    void zero_grad() {
        if (grad) {
            cudaMemset(grad, 0, size * sizeof(float));
        }
    }

    // ---------- backward() ----------
    // Triggers reverse-mode autodiff. This tensor must be a scalar (size=1)
    // or you must have already seeded this->grad with the upstream gradient.
    //
    // Algorithm:
    //   1. Topological sort of the computation graph
    //   2. Walk in reverse order, calling each node's backward_fn
    //
    // This is equivalent to PyTorch's loss.backward().
    void backward() {
        // Seed gradient: if this is a scalar loss, grad = 1.0
        if (!grad) {
            cudaMalloc(&grad, size * sizeof(float));
        }
        float one = 1.0f;
        cudaMemcpy(grad, &one, sizeof(float), cudaMemcpyHostToDevice);

        // Topological sort via DFS
        std::vector<GradTensor*> order;
        std::vector<GradTensor*> visited;

        // Lambda for DFS traversal
        std::function<void(GradTensor*)> topo_sort = [&](GradTensor* node) {
            // Check if already visited
            for (auto* v : visited) {
                if (v == node) return;
            }
            visited.push_back(node);

            // Visit parents first (they come earlier in topo order)
            for (auto* parent : node->parents) {
                topo_sort(parent);
            }
            order.push_back(node);
        };

        topo_sort(this);

        // Walk in reverse topological order (from output back to inputs)
        for (int i = (int)order.size() - 1; i >= 0; i--) {
            if (order[i]->backward_fn) {
                order[i]->backward_fn();
            }
        }
    }

    // No copy (GPU memory is expensive — use pointers)
    GradTensor(const GradTensor&) = delete;
    GradTensor& operator=(const GradTensor&) = delete;
};


// =============================================================================
// Module: Abstract base class for all neural network layers
// =============================================================================
// Every layer (Conv2d, Linear, BatchNorm2d, ReLU, etc.) inherits from Module.
//
// The Module class provides:
//   - forward():      pure virtual — subclasses define the computation
//   - parameters():   recursively collects all learnable GradTensors
//   - train()/set_eval(): switches between training and evaluation mode
//   - print():        displays architecture summary
//   - register_module(): registers a child Module
//   - register_parameter(): registers a learnable GradTensor
//
// This mirrors PyTorch's nn.Module almost exactly.
// =============================================================================

class Module {
public:
    // ---- State ----
    std::string name_;                           // Human-readable name (e.g., "Conv2d(4->16)")
    bool training_;                              // true = training mode, false = inference mode
    std::vector<std::pair<std::string, Module*>> sub_modules_;   // Named child modules
    std::vector<std::pair<std::string, GradTensor*>> params_;    // Named parameters

    // Constructor
    Module() : name_("Module"), training_(true) {}

    // Virtual destructor (so derived classes' destructors are called)
    virtual ~Module() {}

    // ---- forward() ----
    // Pure virtual: every subclass must define how inputs map to outputs.
    // The input is a GradTensor pointer; the output is a NEW GradTensor
    // that is part of the computation graph (for autograd).
    virtual GradTensor* forward(GradTensor* input) = 0;

    // ---- register_module() ----
    // Register a child module by name. This enables recursive parameter
    // collection and architecture printing.
    // Analogous to PyTorch's self.conv1 = nn.Conv2d(...) inside __init__.
    void register_module(const std::string& name, Module* module) {
        sub_modules_.push_back({name, module});
    }

    // ---- register_parameter() ----
    // Register a learnable GradTensor by name. The tensor must have
    // requires_grad = true.
    void register_parameter(const std::string& name, GradTensor* param) {
        params_.push_back({name, param});
    }

    // ---- parameters() ----
    // Recursively collect all learnable parameters from this module
    // and all sub-modules. Returns a flat vector of GradTensor pointers.
    // This is what the Optimizer receives.
    std::vector<GradTensor*> parameters() {
        std::vector<GradTensor*> all_params;

        // Add this module's own parameters
        for (auto& p : params_) {
            all_params.push_back(p.second);
        }

        // Recursively add sub-modules' parameters
        for (auto& m : sub_modules_) {
            auto sub_params = m.second->parameters();
            all_params.insert(all_params.end(),
                              sub_params.begin(), sub_params.end());
        }

        return all_params;
    }

    // ---- train() / set_eval() ----
    // Switch between training and evaluation mode. This affects layers
    // like BatchNorm (uses batch stats in training, running stats in
    // inference) and Dropout (active in training, identity in inference).
    void train(bool mode = true) {
        training_ = mode;
        // Recursively set mode on sub-modules
        for (auto& m : sub_modules_) {
            m.second->train(mode);
        }
    }

    void set_eval() {
        train(false);
    }

    // ---- print() ----
    // Display the model architecture with parameter counts.
    // Output looks like:
    //   Sequential (
    //     (conv1): Conv2d(4, 16, kernel_size=3, padding=1) [params: 592]
    //     (bn1):   BatchNorm2d(16) [params: 32]
    //     (relu):  ReLU() [params: 0]
    //     (fc):    Linear(16, 10) [params: 170]
    //   )
    //   Total parameters: 794
    void print(int indent = 0) {
        std::string pad(indent * 2, ' ');

        // Count this module's own parameters
        int own_params = 0;
        for (auto& p : params_) {
            own_params += p.second->size;
        }

        if (sub_modules_.empty()) {
            // Leaf module — print on one line
            printf("%s%s [params: %d]\n", pad.c_str(), name_.c_str(), own_params);
        } else {
            // Container module — print children indented
            printf("%s%s (\n", pad.c_str(), name_.c_str());
            for (auto& m : sub_modules_) {
                printf("%s  (%s): ", pad.c_str(), m.first.c_str());
                // For leaf modules, print inline; for containers, recurse
                if (m.second->sub_modules_.empty()) {
                    int child_params = 0;
                    for (auto& p : m.second->params_) {
                        child_params += p.second->size;
                    }
                    printf("%s [params: %d]\n",
                           m.second->name_.c_str(), child_params);
                } else {
                    printf("\n");
                    m.second->print(indent + 2);
                }
            }
            printf("%s) [own params: %d]\n", pad.c_str(), own_params);
        }

        // If top-level call, print total
        if (indent == 0) {
            auto all = parameters();
            int total = 0;
            for (auto* p : all) total += p->size;
            printf("Total parameters: %d\n", total);
        }
    }
};

#endif // CUDALEARN_MODULE_CUH
