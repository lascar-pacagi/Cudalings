/*******************************************************************************
 * loss.cuh — Loss Functions for cudalearn
 *
 * Loss functions measure how far the model's predictions are from the true
 * labels. They produce a scalar GradTensor that, when backward() is called,
 * propagates gradients through the entire computation graph.
 *
 * Loss functions implemented:
 *   - CrossEntropyLoss:  softmax + negative log-likelihood (classification)
 *   - MSELoss:           mean squared error (regression)
 *
 * Both functions follow the same pattern:
 *   1. forward(predictions, targets) -> scalar GradTensor*
 *   2. The returned tensor has a backward_fn that computes d(loss)/d(predictions)
 *   3. Calling loss->backward() triggers the full chain of gradient computation
 *
 * IMPORTANT: These functions create new GradTensors that must be deleted by
 * the caller to avoid memory leaks. In a production library, you would use
 * smart pointers or a memory pool.
 ******************************************************************************/

#ifndef CUDALEARN_LOSS_CUH
#define CUDALEARN_LOSS_CUH

#include "module.cuh"


// =============================================================================
// CUDA Kernels for Loss Functions
// =============================================================================


// -----------------------------------------------------------------------------
// Kernel: Find the maximum value per sample (for numerically stable softmax)
// -----------------------------------------------------------------------------
// For each sample n in the batch, find max over all classes:
//   max_vals[n] = max_c logits[n * num_classes + c]
//
// This is Step 1 of the log-sum-exp trick. Without it, exp(logits) can overflow
// to infinity for large logit values. By subtracting the max first, the largest
// exponent becomes exp(0) = 1, and all others are <= 1.
//
// The trick relies on the mathematical identity:
//   softmax(x_i) = exp(x_i - max) / sum_j exp(x_j - max)
//
// Thread mapping: one thread per sample in the batch
// -----------------------------------------------------------------------------
__global__ void find_max_kernel(
    const float* logits,    // [N, C] — raw model outputs (unnormalized log-probabilities)
    float* max_vals,        // [N]    — per-sample maximum
    int N,                  // Batch size
    int C)                  // Number of classes
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Linear scan to find the maximum logit for this sample
    float max_val = logits[n * C];
    for (int c = 1; c < C; c++) {
        float val = logits[n * C + c];
        if (val > max_val) max_val = val;
    }
    max_vals[n] = max_val;
}


// -----------------------------------------------------------------------------
// Kernel: Compute softmax probabilities (numerically stable)
// -----------------------------------------------------------------------------
// For each sample n:
//   sum_exp[n] = sum_c exp(logits[n][c] - max_vals[n])
//   probs[n][c] = exp(logits[n][c] - max_vals[n]) / sum_exp[n]
//
// The softmax function converts raw logits into a probability distribution:
//   - All outputs are in [0, 1]
//   - They sum to 1 across classes
//   - The class with the largest logit gets the highest probability
//
// Mathematically:
//   softmax(x_i) = exp(x_i) / sum_j exp(x_j)
//
// But we use the numerically stable version:
//   softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
//
// This avoids overflow because the largest exponent is exp(0) = 1.
//
// Thread mapping: one thread per sample in the batch
// -----------------------------------------------------------------------------
__global__ void softmax_kernel(
    const float* logits,    // [N, C] — raw model outputs
    const float* max_vals,  // [N]    — per-sample maximum (from find_max_kernel)
    float* probs,           // [N, C] — output softmax probabilities
    int N,                  // Batch size
    int C)                  // Number of classes
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float max_val = max_vals[n];

    // Step 1: Compute sum of exponentials (denominator of softmax)
    float sum_exp = 0.0f;
    for (int c = 0; c < C; c++) {
        sum_exp += expf(logits[n * C + c] - max_val);
    }

    // Step 2: Compute softmax probabilities
    // Each probability = exp(logit - max) / sum_exp
    for (int c = 0; c < C; c++) {
        probs[n * C + c] = expf(logits[n * C + c] - max_val) / sum_exp;
    }
}


// -----------------------------------------------------------------------------
// Kernel: Compute cross-entropy loss (negative log-likelihood of softmax)
// -----------------------------------------------------------------------------
// The cross-entropy loss for a single sample with true class y is:
//   loss_n = -log(probs[n][y])
//
// The total loss is the mean over the batch:
//   loss = (1/N) * sum_n loss_n = -(1/N) * sum_n log(probs[n][labels[n]])
//
// Why cross-entropy?
//   - It is the natural loss for classification (derived from maximum likelihood)
//   - It heavily penalizes confident wrong predictions (log(p) -> -inf as p -> 0)
//   - Its gradient has a beautiful simplicity: d_logits = probs - one_hot
//
// Thread mapping: single thread (reduces N per-sample losses into one scalar)
// For large batches, a parallel reduction would be faster, but this is simpler.
// -----------------------------------------------------------------------------
__global__ void cross_entropy_loss_kernel(
    const float* probs,     // [N, C] — softmax probabilities
    const int* labels,      // [N]    — true class indices (0 to C-1)
    float* loss,            // [1]    — scalar output loss
    int N,                  // Batch size
    int C)                  // Number of classes
{
    // Single thread computes the average negative log-likelihood
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float total = 0.0f;
    for (int n = 0; n < N; n++) {
        // Get the probability the model assigned to the true class
        float p = probs[n * C + labels[n]];

        // Clamp to avoid log(0) which gives -infinity
        // log(1e-7) ~= -16, which is a large but finite penalty
        if (p < 1e-7f) p = 1e-7f;

        // Negative log-likelihood: -log(p)
        // When p=1 (perfect prediction): loss = 0
        // When p=0 (completely wrong):   loss = very large
        total += -logf(p);
    }

    // Average over the batch
    *loss = total / (float)N;
}


// -----------------------------------------------------------------------------
// Kernel: Cross-entropy backward (gradient w.r.t. logits)
// -----------------------------------------------------------------------------
// The gradient of cross-entropy loss w.r.t. the input logits has an elegant form:
//
//   d_logits[n][c] = (1/N) * (probs[n][c] - one_hot[n][c])
//
// Where one_hot[n][c] = 1 if c == labels[n], else 0.
//
// This means:
//   - For the true class: gradient = (1/N) * (prob - 1)  (negative, pushes logit up)
//   - For wrong classes:  gradient = (1/N) * prob         (positive, pushes logit down)
//
// Derivation:
//   loss = -(1/N) * sum_n log(softmax(x)_{labels[n]})
//   d_loss/d_x_i = (1/N) * (softmax(x)_i - 1_{i=label})
//
// This beautiful formula is one reason softmax + cross-entropy is so popular:
//   the gradient is just "predicted probability minus true probability" divided
//   by batch size. Simple, numerically stable, and efficient.
//
// Thread mapping: one thread per element in the [N, C] logits tensor
// -----------------------------------------------------------------------------
__global__ void cross_entropy_backward_kernel(
    const float* probs,     // [N, C] — softmax probabilities from forward pass
    const int* labels,      // [N]    — true class indices
    float* d_logits,        // [N, C] — gradient to write (accumulated into)
    int N,                  // Batch size
    int C)                  // Number of classes
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;

    int c = idx % C;        // Which class
    int n = idx / C;        // Which sample in the batch

    // Gradient = (1/N) * (softmax_probability - one_hot_label)
    // The one_hot term is 1 only for the true class, 0 elsewhere
    float one_hot = (c == labels[n]) ? 1.0f : 0.0f;
    d_logits[idx] += (probs[idx] - one_hot) / (float)N;
}


// -----------------------------------------------------------------------------
// Kernel: MSE forward (mean squared error)
// -----------------------------------------------------------------------------
// MSE loss for regression:
//   loss = (1/N) * sum_n (predictions[n] - targets[n])^2
//
// This is the simplest loss for regression tasks. It heavily penalizes large
// errors (quadratic penalty) and produces smooth gradients.
//
// Thread mapping: single thread (simple reduction)
// -----------------------------------------------------------------------------
__global__ void mse_loss_kernel(
    const float* predictions,   // [N] — model outputs
    const float* targets,       // [N] — true values
    float* loss,                // [1] — scalar output
    int N)                      // Number of elements
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float total = 0.0f;
    for (int i = 0; i < N; i++) {
        float diff = predictions[i] - targets[i];
        total += diff * diff;
    }
    *loss = total / (float)N;
}


// -----------------------------------------------------------------------------
// Kernel: MSE backward
// -----------------------------------------------------------------------------
// Gradient of MSE loss w.r.t. predictions:
//   d_pred[n] = (2/N) * (predictions[n] - targets[n])
//
// Derivation:
//   loss = (1/N) * sum_n (pred_n - target_n)^2
//   d_loss/d_pred_n = (2/N) * (pred_n - target_n)
//
// The gradient is proportional to the error: large errors get large gradients,
// which pushes the model harder to fix big mistakes.
//
// Thread mapping: one thread per element
// -----------------------------------------------------------------------------
__global__ void mse_backward_kernel(
    const float* predictions,   // [N] — model outputs
    const float* targets,       // [N] — true values
    float* d_predictions,       // [N] — gradient to write
    int N)                      // Number of elements
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // d_loss/d_pred = 2 * (pred - target) / N
    d_predictions[idx] += 2.0f * (predictions[idx] - targets[idx]) / (float)N;
}


// =============================================================================
// CrossEntropyLoss
// =============================================================================
// Combines softmax and negative log-likelihood into a single loss function.
//
// Input:   logits [N, C] — raw model outputs (NOT softmax probabilities)
//          labels [N]    — integer class labels (0 to C-1), stored on GPU
//
// Output:  scalar GradTensor (size 1) containing the loss value
//
// Backward: computes d(loss)/d(logits) = (1/N) * (softmax(logits) - one_hot(labels))
//
// Why combine softmax + NLL?
//   1. Numerically stable: we use the log-sum-exp trick on the raw logits
//   2. Efficient backward: the gradient is just softmax - one_hot
//   3. Matches PyTorch's nn.CrossEntropyLoss exactly
//
// IMPORTANT: The input should be raw logits, not softmax probabilities.
// Applying softmax before CrossEntropyLoss would double-softmax and give
// wrong results.
// =============================================================================

class CrossEntropyLoss {
public:
    // ---- Temporary GPU buffers ----
    // These are allocated in forward() and freed in the destructor.
    // They store intermediate results needed by the backward pass.
    float* probs_;      // [N, C] — softmax probabilities (kept for backward)
    float* max_vals_;   // [N]    — per-sample max logits (for numerical stability)
    int* labels_buf_;   // [N]    — copy of labels (kept for backward)
    int last_N_;        // Batch size from the last forward call
    int last_C_;        // Number of classes from the last forward call

    CrossEntropyLoss()
        : probs_(nullptr), max_vals_(nullptr), labels_buf_(nullptr),
          last_N_(0), last_C_(0) {}

    ~CrossEntropyLoss() {
        if (probs_) cudaFree(probs_);
        if (max_vals_) cudaFree(max_vals_);
        if (labels_buf_) cudaFree(labels_buf_);
    }

    // ---- forward() ----
    // Compute the cross-entropy loss given raw logits and integer labels.
    //
    // Args:
    //   logits: GradTensor of shape [N, C] — raw model outputs
    //   labels: GPU int array of shape [N] — true class labels (0 to C-1)
    //
    // Returns:
    //   GradTensor* of size 1 — the scalar loss value with backward_fn set up
    //
    // The forward pass does:
    //   1. Find max per sample (numerical stability)
    //   2. Compute softmax probabilities
    //   3. Compute -log(prob[true_class]) averaged over the batch
    //
    // The backward pass (triggered by loss->backward()) does:
    //   d_logits[n][c] = (1/N) * (softmax[n][c] - one_hot[n][c])
    GradTensor* forward(GradTensor* logits, int* labels) {
        int N = logits->dims[0];    // Batch size
        int C = logits->dims[1];    // Number of classes
        last_N_ = N;
        last_C_ = C;

        // --- Allocate/reallocate temporary buffers ---

        // Free old buffers if they exist (in case batch size changed)
        if (probs_) cudaFree(probs_);
        if (max_vals_) cudaFree(max_vals_);
        if (labels_buf_) cudaFree(labels_buf_);

        // Allocate softmax probability buffer [N, C]
        cudaMalloc(&probs_, N * C * sizeof(float));

        // Allocate per-sample max buffer [N]
        cudaMalloc(&max_vals_, N * sizeof(float));

        // Copy labels to a buffer we own (so they persist until backward)
        cudaMalloc(&labels_buf_, N * sizeof(int));
        cudaMemcpy(labels_buf_, labels, N * sizeof(int), cudaMemcpyDeviceToDevice);

        int threads = 256;

        // --- Step 1: Find per-sample max (for numerical stability) ---
        // max_vals[n] = max_c logits[n][c]
        int blocks_n = (N + threads - 1) / threads;
        find_max_kernel<<<blocks_n, threads>>>(
            logits->data, max_vals_, N, C
        );

        // --- Step 2: Compute softmax probabilities ---
        // probs[n][c] = exp(logits[n][c] - max[n]) / sum_j exp(logits[n][j] - max[n])
        softmax_kernel<<<blocks_n, threads>>>(
            logits->data, max_vals_, probs_, N, C
        );

        // --- Step 3: Compute the scalar loss ---
        // loss = -(1/N) * sum_n log(probs[n][labels[n]])
        GradTensor* loss = new GradTensor(1, 1, 1, 1, false);
        // Allocate grad for the loss tensor (will be seeded to 1.0 by backward())
        cudaMalloc(&loss->grad, sizeof(float));
        cudaMemset(loss->grad, 0, sizeof(float));

        cross_entropy_loss_kernel<<<1, 1>>>(
            probs_, labels_buf_, loss->data, N, C
        );

        // --- Set up backward function ---
        // The backward pass computes the gradient of the loss w.r.t. the input logits.
        // This gradient is: d_logits = (1/N) * (softmax - one_hot)
        //
        // We need to propagate this gradient into logits->grad, which will then
        // be propagated further back through the network by the autograd system.
        loss->parents = {logits};

        // Capture the buffers needed for backward by value (pointers)
        float* probs = probs_;
        int* lbls = labels_buf_;

        loss->backward_fn = [loss, logits, probs, lbls, N, C]() {
            // Ensure the logits tensor has gradient storage
            if (!logits->grad) {
                cudaMalloc(&logits->grad, N * C * sizeof(float));
                cudaMemset(logits->grad, 0, N * C * sizeof(float));
            }

            // Compute gradient: d_logits = (1/N) * (softmax - one_hot)
            int total = N * C;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            cross_entropy_backward_kernel<<<blocks, threads>>>(
                probs, lbls, logits->grad, N, C
            );
        };

        return loss;
    }
};


// =============================================================================
// MSELoss
// =============================================================================
// Mean Squared Error loss for regression tasks.
//
// Input:   predictions [N] — model outputs
//          targets     [N] — true values (GPU float array)
//
// Output:  scalar GradTensor (size 1) containing the loss value
//
// Forward:  loss = (1/N) * sum_n (pred_n - target_n)^2
// Backward: d_pred_n = (2/N) * (pred_n - target_n)
//
// Properties:
//   - Always non-negative (squared terms)
//   - Zero when predictions exactly match targets
//   - Smooth and differentiable everywhere
//   - Gradient magnitude proportional to error magnitude
//   - Sensitive to outliers (quadratic penalty)
//
// For tasks where outlier robustness matters, consider Huber loss (L1+L2 hybrid).
// =============================================================================

class MSELoss {
public:
    // We store targets for the backward pass
    float* targets_buf_;
    int last_N_;

    MSELoss() : targets_buf_(nullptr), last_N_(0) {}

    ~MSELoss() {
        if (targets_buf_) cudaFree(targets_buf_);
    }

    // ---- forward() ----
    // Compute MSE loss given predictions and target values.
    //
    // Args:
    //   predictions: GradTensor of shape [N] — model outputs
    //   targets:     GPU float array of shape [N] — true values
    //
    // Returns:
    //   GradTensor* of size 1 — scalar loss with backward_fn
    GradTensor* forward(GradTensor* predictions, float* targets) {
        int N = predictions->size;
        last_N_ = N;

        // Copy targets to our buffer (persist for backward)
        if (targets_buf_) cudaFree(targets_buf_);
        cudaMalloc(&targets_buf_, N * sizeof(float));
        cudaMemcpy(targets_buf_, targets, N * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        // Compute loss: (1/N) * sum (pred - target)^2
        GradTensor* loss = new GradTensor(1, 1, 1, 1, false);
        cudaMalloc(&loss->grad, sizeof(float));
        cudaMemset(loss->grad, 0, sizeof(float));

        mse_loss_kernel<<<1, 1>>>(
            predictions->data, targets_buf_, loss->data, N
        );

        // Set up backward
        loss->parents = {predictions};
        float* tgt = targets_buf_;

        loss->backward_fn = [loss, predictions, tgt, N]() {
            if (!predictions->grad) {
                cudaMalloc(&predictions->grad, N * sizeof(float));
                cudaMemset(predictions->grad, 0, N * sizeof(float));
            }

            int threads = 256;
            int blocks = (N + threads - 1) / threads;
            mse_backward_kernel<<<blocks, threads>>>(
                predictions->data, tgt, predictions->grad, N
            );
        };

        return loss;
    }
};


#endif // CUDALEARN_LOSS_CUH
