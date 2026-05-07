/*******************************************************************************
 * optimizer.cuh — Optimizers and Learning Rate Schedulers for cudalearn
 *
 * This file implements the optimizers that update model parameters during
 * training. Each optimizer receives a vector of GradTensor* (the model's
 * learnable parameters) and applies a specific update rule using CUDA kernels.
 *
 * Optimizers implemented:
 *   - SGD:  Stochastic Gradient Descent with momentum and weight decay
 *   - Adam: Adaptive Moment Estimation (the most popular optimizer)
 *
 * Learning rate schedulers:
 *   - CosineAnnealingLR: cosine decay from lr_max to eta_min over T_max epochs
 *
 * All state buffers (velocity, moments) live on the GPU for maximum throughput.
 * The optimizer kernels fuse multiple operations into a single kernel launch
 * to minimize kernel launch overhead.
 *
 * Usage (mirrors PyTorch):
 *   auto params = model->parameters();
 *   Adam optimizer(params, 0.001f);
 *   CosineAnnealingLR scheduler(&optimizer, 100);
 *
 *   for (int epoch = 0; epoch < 100; epoch++) {
 *       // ... forward, loss, backward ...
 *       optimizer.step();
 *       optimizer.zero_grad();
 *       scheduler.step(epoch);
 *   }
 ******************************************************************************/

#ifndef CUDALEARN_OPTIMIZER_CUH
#define CUDALEARN_OPTIMIZER_CUH

#include "module.cuh"

// =============================================================================
// CUDA Kernels for Optimizer Updates
// =============================================================================
// Each kernel performs a fused parameter update — reading the gradient, updating
// internal state (velocity/moments), and writing the new parameter value, all
// in a single pass over memory. This is important because these arrays can be
// large (millions of floats for a ResNet), and we want to minimize memory
// bandwidth usage.
// =============================================================================


// -----------------------------------------------------------------------------
// Kernel: SGD with Momentum and Weight Decay
// -----------------------------------------------------------------------------
// The SGD update rule with momentum and L2 weight decay:
//
//   v_t = momentum * v_{t-1} + grad + weight_decay * param
//   param = param - lr * v_t
//
// Where:
//   - v_t:          velocity buffer (accumulated gradient direction)
//   - momentum:     how much of the previous velocity to retain (typically 0.9)
//   - grad:         gradient of the loss w.r.t. this parameter
//   - weight_decay: L2 regularization coefficient (prevents overfitting)
//   - lr:           learning rate (step size)
//
// Without momentum (momentum=0), this reduces to vanilla SGD:
//   param = param - lr * (grad + weight_decay * param)
//
// Thread mapping: one thread per parameter element
// -----------------------------------------------------------------------------
__global__ void sgd_step_kernel(
    float* param,       // Parameter values (read + write)
    const float* grad,  // Gradient values (read only)
    float* velocity,    // Velocity buffer (read + write)
    float lr,           // Learning rate
    float momentum,     // Momentum coefficient (e.g., 0.9)
    float weight_decay, // L2 regularization coefficient
    int size)           // Number of elements in this parameter
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Read gradient and current parameter value
    float g = grad[idx];
    float p = param[idx];

    // Add L2 weight decay: effectively adds weight_decay * param to the gradient.
    // This penalizes large weights, encouraging the network to use smaller values.
    // Mathematically equivalent to adding (weight_decay/2) * ||param||^2 to the loss.
    g += weight_decay * p;

    // Update velocity with momentum:
    //   v = momentum * v_old + g
    // The velocity accumulates gradient information over time. With momentum=0.9,
    // the velocity is a weighted sum of the last ~10 gradients, which smooths out
    // noisy gradient estimates and helps navigate ravines in the loss landscape.
    float v = momentum * velocity[idx] + g;
    velocity[idx] = v;

    // Update parameter: param -= lr * velocity
    param[idx] = p - lr * v;
}


// -----------------------------------------------------------------------------
// Kernel: Adam (Adaptive Moment Estimation)
// -----------------------------------------------------------------------------
// Adam combines the ideas of momentum (first moment) and RMSProp (second moment)
// to adapt the learning rate per-parameter based on gradient statistics.
//
// Update rules:
//   m_t = beta1 * m_{t-1} + (1 - beta1) * grad          (first moment estimate)
//   v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2         (second moment estimate)
//   m_hat = m_t / (1 - beta1^t)                           (bias correction)
//   v_hat = v_t / (1 - beta2^t)                           (bias correction)
//   param = param - lr * m_hat / (sqrt(v_hat) + eps)
//
// Where:
//   - m_t:   exponential moving average of gradients (tracks direction)
//   - v_t:   exponential moving average of squared gradients (tracks magnitude)
//   - beta1: decay rate for first moment (typically 0.9)
//   - beta2: decay rate for second moment (typically 0.999)
//   - eps:   small constant to prevent division by zero (typically 1e-8)
//
// Bias correction is needed because m and v are initialized to zero, which
// biases them toward zero in the early steps. Dividing by (1 - beta^t)
// corrects for this initialization bias.
//
// Why Adam is popular:
//   1. Adaptive per-parameter learning rates (no manual tuning per layer)
//   2. Works well with sparse gradients (common in NLP)
//   3. Generally converges faster than SGD on many problems
//   4. Less sensitive to learning rate choice than vanilla SGD
//
// Thread mapping: one thread per parameter element
// -----------------------------------------------------------------------------
__global__ void adam_step_kernel(
    float* param,       // Parameter values (read + write)
    const float* grad,  // Gradient values (read only)
    float* m,           // First moment buffer (read + write)
    float* v,           // Second moment buffer (read + write)
    float lr,           // Learning rate
    float beta1,        // First moment decay rate (e.g., 0.9)
    float beta2,        // Second moment decay rate (e.g., 0.999)
    float eps,          // Numerical stability constant (e.g., 1e-8)
    float weight_decay, // L2 regularization coefficient
    float bc1,          // Bias correction factor for first moment: 1/(1 - beta1^t)
    float bc2,          // Bias correction factor for second moment: 1/(1 - beta2^t)
    int size)           // Number of elements in this parameter
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Read gradient and apply weight decay (AdamW-style decoupled weight decay)
    float g = grad[idx];
    float p = param[idx];

    // L2 regularization: add weight_decay * param to gradient
    // This is standard L2 regularization (not AdamW decoupled weight decay).
    // For true AdamW, you would subtract weight_decay * lr * param separately.
    g += weight_decay * p;

    // Update first moment (mean of gradients):
    //   m = beta1 * m + (1 - beta1) * g
    // This is an exponential moving average with decay rate beta1.
    // With beta1=0.9, recent gradients have more influence.
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    m[idx] = m_new;

    // Update second moment (mean of squared gradients):
    //   v = beta2 * v + (1 - beta2) * g^2
    // This tracks the magnitude of recent gradients, allowing Adam to
    // shrink the effective learning rate for parameters with large gradients
    // and increase it for parameters with small gradients.
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
    v[idx] = v_new;

    // Apply bias correction:
    //   m_hat = m / (1 - beta1^t)
    //   v_hat = v / (1 - beta2^t)
    // Without this, the moments would be biased toward zero in early steps
    // because they are initialized to zero.
    float m_hat = m_new * bc1;  // bc1 = 1/(1 - beta1^t)
    float v_hat = v_new * bc2;  // bc2 = 1/(1 - beta2^t)

    // Update parameter:
    //   param -= lr * m_hat / (sqrt(v_hat) + eps)
    // The denominator adapts the effective learning rate per-parameter.
    // Parameters with consistently large gradients get smaller updates;
    // parameters with small gradients get larger updates.
    param[idx] = p - lr * m_hat / (sqrtf(v_hat) + eps);
}


// =============================================================================
// SGD Optimizer
// =============================================================================
// Stochastic Gradient Descent with optional momentum and weight decay.
//
// This is the simplest optimizer. Despite its simplicity, SGD with momentum
// often achieves better generalization than Adam on image classification tasks,
// which is why it is still widely used for training CNNs and ResNets.
//
// Member variables:
//   - params_:      vector of GradTensor* to optimize
//   - lr_:          learning rate (step size)
//   - momentum_:    momentum coefficient
//   - weight_decay_: L2 regularization strength
//   - velocities_:  GPU-allocated velocity buffers (one per parameter)
// =============================================================================

class SGD {
public:
    std::vector<GradTensor*> params_;   // Parameters to optimize
    float lr_;                          // Current learning rate
    float momentum_;                    // Momentum coefficient (0 = no momentum)
    float weight_decay_;                // L2 regularization strength (0 = none)
    std::vector<float*> velocities_;    // GPU velocity buffers (one per parameter)

    // ---- Constructor ----
    // Allocates one GPU velocity buffer (initialized to zero) for each parameter.
    // The velocity buffer has the same size as the parameter tensor.
    //
    // Args:
    //   params:       model.parameters() — all learnable GradTensors
    //   lr:           learning rate (e.g., 0.01 for SGD)
    //   momentum:     momentum coefficient (e.g., 0.9)
    //   weight_decay: L2 regularization (e.g., 1e-4 for ResNet)
    SGD(std::vector<GradTensor*> params, float lr = 0.01f,
        float momentum = 0.0f, float weight_decay = 0.0f)
        : params_(params), lr_(lr), momentum_(momentum),
          weight_decay_(weight_decay)
    {
        // Allocate velocity buffers on the GPU — one per parameter tensor.
        // These store the "momentum" accumulation, which smooths the gradient
        // updates over time. Initialized to zero.
        for (auto* p : params_) {
            float* vel = nullptr;
            cudaMalloc(&vel, p->size * sizeof(float));
            cudaMemset(vel, 0, p->size * sizeof(float));
            velocities_.push_back(vel);
        }
    }

    // ---- Destructor ----
    // Free all GPU velocity buffers
    ~SGD() {
        for (auto* v : velocities_) {
            if (v) cudaFree(v);
        }
    }

    // ---- step() ----
    // Apply one SGD update to all parameters. This should be called after
    // loss.backward() has populated the gradients.
    //
    // For each parameter p with gradient g and velocity v:
    //   v = momentum * v + g + weight_decay * p
    //   p = p - lr * v
    //
    // Each parameter tensor gets its own kernel launch. This is fine for
    // typical networks (tens of parameter tensors). For extreme cases with
    // thousands of tiny tensors, you could fuse them into a single kernel.
    void step() {
        int threads = 256;
        for (size_t i = 0; i < params_.size(); i++) {
            GradTensor* p = params_[i];
            if (!p->grad) continue;  // Skip parameters without gradients

            int blocks = (p->size + threads - 1) / threads;
            sgd_step_kernel<<<blocks, threads>>>(
                p->data, p->grad, velocities_[i],
                lr_, momentum_, weight_decay_, p->size
            );
        }
        // Synchronize to ensure all updates are complete before the next forward pass
        cudaDeviceSynchronize();
    }

    // ---- zero_grad() ----
    // Reset all parameter gradients to zero. Must be called before each forward
    // pass to prevent gradient accumulation across mini-batches.
    //
    // In PyTorch, this is optimizer.zero_grad(). It zeroes the .grad field of
    // every parameter tensor. Without this, gradients from the previous batch
    // would be added to the current batch's gradients (which is sometimes
    // intentional for gradient accumulation, but usually a bug).
    void zero_grad() {
        for (auto* p : params_) {
            p->zero_grad();
        }
    }
};


// =============================================================================
// Adam Optimizer
// =============================================================================
// Adaptive Moment Estimation — the most popular optimizer for deep learning.
//
// Adam maintains two running averages per parameter:
//   - First moment (m):  mean of gradients     -> tracks direction
//   - Second moment (v): mean of grad squared   -> tracks magnitude
//
// These are used to compute an adaptive learning rate for each parameter.
// Parameters with large gradients get smaller effective learning rates,
// preventing overshooting. Parameters with small gradients get larger
// effective learning rates, speeding up convergence.
//
// Default hyperparameters (from the original paper, Kingma & Ba 2014):
//   beta1 = 0.9, beta2 = 0.999, eps = 1e-8
//
// Member variables:
//   - params_:      vector of GradTensor* to optimize
//   - lr_:          learning rate
//   - beta1_, beta2_: moment decay rates
//   - eps_:         numerical stability constant
//   - weight_decay_: L2 regularization
//   - m_, v_:       GPU-allocated moment buffers
//   - t_:           step counter (for bias correction)
// =============================================================================

class Adam {
public:
    std::vector<GradTensor*> params_;   // Parameters to optimize
    float lr_;                          // Learning rate (e.g., 0.001)
    float beta1_;                       // First moment decay (e.g., 0.9)
    float beta2_;                       // Second moment decay (e.g., 0.999)
    float eps_;                         // Numerical stability (e.g., 1e-8)
    float weight_decay_;                // L2 regularization strength
    std::vector<float*> m_;             // First moment buffers (GPU)
    std::vector<float*> v_;             // Second moment buffers (GPU)
    int t_;                             // Step counter (starts at 0)

    // ---- Constructor ----
    // Allocates first and second moment buffers on GPU for each parameter.
    // Both are initialized to zero, which is why bias correction is needed.
    //
    // Args:
    //   params:       model.parameters()
    //   lr:           learning rate (0.001 is a good default for Adam)
    //   beta1:        first moment decay (0.9 works well for most tasks)
    //   beta2:        second moment decay (0.999 works well for most tasks)
    //   eps:          prevents division by zero (1e-8 is standard)
    //   weight_decay: L2 regularization (0 = none)
    Adam(std::vector<GradTensor*> params, float lr = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f)
        : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2),
          eps_(eps), weight_decay_(weight_decay), t_(0)
    {
        // Allocate first moment (m) and second moment (v) buffers on GPU.
        // Each parameter gets its own pair of buffers, same size as the parameter.
        // m tracks the exponential moving average of the gradient (direction).
        // v tracks the exponential moving average of the squared gradient (scale).
        for (auto* p : params_) {
            float* m_buf = nullptr;
            float* v_buf = nullptr;
            cudaMalloc(&m_buf, p->size * sizeof(float));
            cudaMalloc(&v_buf, p->size * sizeof(float));
            cudaMemset(m_buf, 0, p->size * sizeof(float));
            cudaMemset(v_buf, 0, p->size * sizeof(float));
            m_.push_back(m_buf);
            v_.push_back(v_buf);
        }
    }

    // ---- Destructor ----
    // Free all GPU moment buffers
    ~Adam() {
        for (size_t i = 0; i < m_.size(); i++) {
            if (m_[i]) cudaFree(m_[i]);
            if (v_[i]) cudaFree(v_[i]);
        }
    }

    // ---- step() ----
    // Apply one Adam update to all parameters. Call after loss.backward().
    //
    // The update for each parameter p with gradient g:
    //   t += 1
    //   m = beta1 * m + (1 - beta1) * g          // update first moment
    //   v = beta2 * v + (1 - beta2) * g^2         // update second moment
    //   m_hat = m / (1 - beta1^t)                  // bias-corrected first moment
    //   v_hat = v / (1 - beta2^t)                  // bias-corrected second moment
    //   p = p - lr * m_hat / (sqrt(v_hat) + eps)   // parameter update
    //
    // The bias correction factors (1 - beta^t) are precomputed on the CPU and
    // passed to the kernel, since they are the same for all parameters.
    void step() {
        // Increment step counter (starts at 1 for the first step)
        t_++;

        // Precompute bias correction factors on CPU.
        // These correct for the zero-initialization of m and v.
        // bc1 = 1 / (1 - beta1^t), bc2 = 1 / (1 - beta2^t)
        //
        // At t=1: bc1 = 1/(1-0.9) = 10, bc2 = 1/(1-0.999) = 1000
        //   -> large correction because m and v have only seen one gradient
        // At t=100: bc1 ~= 1.0, bc2 ~= 1.0
        //   -> correction is negligible because m and v have converged
        float bc1 = 1.0f / (1.0f - powf(beta1_, (float)t_));
        float bc2 = 1.0f / (1.0f - powf(beta2_, (float)t_));

        int threads = 256;
        for (size_t i = 0; i < params_.size(); i++) {
            GradTensor* p = params_[i];
            if (!p->grad) continue;  // Skip parameters without gradients

            int blocks = (p->size + threads - 1) / threads;
            adam_step_kernel<<<blocks, threads>>>(
                p->data, p->grad, m_[i], v_[i],
                lr_, beta1_, beta2_, eps_, weight_decay_,
                bc1, bc2, p->size
            );
        }
        cudaDeviceSynchronize();
    }

    // ---- zero_grad() ----
    // Reset all parameter gradients to zero before the next forward pass.
    // See SGD::zero_grad() for explanation.
    void zero_grad() {
        for (auto* p : params_) {
            p->zero_grad();
        }
    }
};


// =============================================================================
// Learning Rate Scheduler: CosineAnnealingLR
// =============================================================================
// Decays the learning rate following a cosine curve from lr_max down to eta_min
// over T_max epochs, then optionally restarts (warm restarts).
//
// Formula:
//   lr(epoch) = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(pi * epoch / T_max))
//
// At epoch 0:   lr = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(0))
//             = eta_min + (lr_max - eta_min) * 1
//             = lr_max                                     (full learning rate)
//
// At epoch T_max/2: lr = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(pi/2))
//                 = eta_min + 0.5 * (lr_max - eta_min) * 1
//                 = midpoint between lr_max and eta_min     (half decay)
//
// At epoch T_max: lr = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(pi))
//               = eta_min + 0                               (minimum learning rate)
//
// Why cosine annealing?
//   1. Smooth decay avoids sudden drops that can destabilize training
//   2. Starts with aggressive updates, then fine-tunes with small steps
//   3. Empirically matches or beats step-decay schedules on many benchmarks
//   4. Pairs well with warm restarts (SGDR paper, Loshchilov & Hutter 2017)
//
// This scheduler works with any optimizer that has a public lr_ member.
// We template it so it can work with both SGD and Adam.
// =============================================================================

// We use a simple approach: the scheduler holds a pointer to the optimizer's
// learning rate. This avoids templates and works with both SGD and Adam since
// both expose lr_ as a public float member.
//
// For a cleaner design, you could use an Optimizer base class, but that adds
// complexity we don't need for this educational library.

class CosineAnnealingLR {
public:
    float* lr_ptr_;     // Pointer to the optimizer's learning rate field
    float lr_max_;      // Initial (maximum) learning rate
    float eta_min_;     // Minimum learning rate at the end of the schedule
    int T_max_;         // Total number of epochs for one cosine cycle

    // ---- Constructor ----
    // Takes a pointer to the optimizer's lr_ field, plus schedule parameters.
    //
    // Args:
    //   lr_ptr:  pointer to optimizer.lr_ (modified in-place by step())
    //   T_max:   number of epochs in one cosine half-cycle
    //   eta_min: minimum learning rate (default 0 = full decay to zero)
    //
    // Usage with SGD:
    //   SGD optimizer(params, 0.1);
    //   CosineAnnealingLR scheduler(&optimizer.lr_, 200, 1e-4);
    //
    // Usage with Adam:
    //   Adam optimizer(params, 0.001);
    //   CosineAnnealingLR scheduler(&optimizer.lr_, 100, 1e-5);
    CosineAnnealingLR(float* lr_ptr, int T_max, float eta_min = 0.0f)
        : lr_ptr_(lr_ptr), T_max_(T_max), eta_min_(eta_min)
    {
        // Save the initial learning rate as the maximum
        lr_max_ = *lr_ptr_;
    }

    // ---- step() ----
    // Update the learning rate based on the current epoch.
    //
    // This directly modifies the optimizer's lr_ field through the pointer.
    // The optimizer will use the new learning rate on its next step() call.
    //
    // Args:
    //   epoch: current epoch number (0-indexed)
    //
    // The formula:
    //   lr = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(pi * epoch / T_max))
    //
    // This traces out one half of a cosine wave:
    //   epoch=0:      cos(0) = 1       -> lr = lr_max
    //   epoch=T/4:    cos(pi/4) ~= 0.7 -> lr ~= 0.85 * lr_max
    //   epoch=T/2:    cos(pi/2) = 0    -> lr = midpoint
    //   epoch=3T/4:   cos(3pi/4) ~=-0.7-> lr ~= 0.15 * lr_max
    //   epoch=T:      cos(pi) = -1     -> lr = eta_min
    void step(int epoch) {
        // Compute cosine-annealed learning rate
        float cos_val = cosf(M_PI * (float)epoch / (float)T_max_);
        *lr_ptr_ = eta_min_ + 0.5f * (lr_max_ - eta_min_) * (1.0f + cos_val);
    }
};


#endif // CUDALEARN_OPTIMIZER_CUH
