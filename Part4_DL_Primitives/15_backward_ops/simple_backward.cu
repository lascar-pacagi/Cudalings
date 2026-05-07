// ===========================================================================
// Chapter 15: simple_backward.cu -- Backward Pass for Simple Operations
// ===========================================================================
//
// Implements backward (gradient) kernels for five "simpler" operations:
//
//   1. relu_backward         -- element-wise gate
//   2. linear_backward       -- matrix multiplications + reduction
//   3. gap_backward          -- Global Average Pooling inverse
//   4. cross_entropy_backward -- softmax minus one-hot
//   5. residual_backward     -- copy to both branches
//
// Each operation includes:
//   (a) CUDA kernel for the backward pass
//   (b) CPU reference implementation (for verification)
//   (c) Numerical gradient checking via finite differences
//   (d) Chain rule math in comments
//
// This file is the "easy wins" of backpropagation -- every gradient here
// is simpler than Conv2D or BatchNorm, but together they form the glue
// that holds a neural network's backward pass together.
//
// ===========================================================================
//
//  NOTATION USED THROUGHOUT:
//
//    dL/dy  = "grad_output" = upstream gradient (what we RECEIVE)
//    dL/dx  = "grad_input"  = downstream gradient (what we PRODUCE)
//    dL/dW  = "grad_weight" = parameter gradient (what we ACCUMULATE)
//
//    The chain rule says:
//
//      dL/dx = dL/dy * dy/dx
//              ^^^^    ^^^^
//              |       local Jacobian of this operation
//              upstream gradient from the layer above
//
//    We always receive dL/dy and must produce dL/dx (and dL/d(params)).
//
// ===========================================================================

#include "../13_tensor_class/tensor.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cfloat>
#include <vector>

// ===========================================================================
//  1. RELU BACKWARD
// ===========================================================================
//
//  Forward:  y = max(0, x)
//
//  The derivative of ReLU is a step function:
//
//    dy/dx = { 1   if x > 0
//            { 0   if x <= 0
//
//  Chain rule:
//
//    dL/dx = dL/dy * dy/dx
//          = dL/dy * (x > 0)        <-- element-wise multiply by binary mask
//
//  ASCII DIAGRAM:
//
//    input x:      [-2.0,  3.0, -1.0,  5.0,  0.0,  7.0]
//    mask (x>0):   [ 0,    1,    0,    1,    0,    1  ]
//    grad_output:  [ 0.5, -0.3,  0.1,  0.8, -0.2,  0.4]
//    grad_input:   [ 0.0, -0.3,  0.0,  0.8,  0.0,  0.4]
//                    ^^^          ^^^          ^^^
//                    killed       killed       killed
//                    (x <= 0)     (x <= 0)     (x == 0)
//
//  Key insight: ReLU backward is a GATE. Where the input was positive,
//  the gradient flows through unchanged. Where it was zero or negative,
//  the gradient is killed (set to zero). This is why "dying ReLU" is
//  a problem: if a neuron's input is always negative, its gradient is
//  always zero, and it can never recover.
//
//  No parameters: ReLU has no learnable parameters, so there is no
//  grad_weight or grad_bias to compute.
//
// ===========================================================================

__global__ void relu_backward_kernel(
    const float* __restrict__ grad_output,  // dL/dy, shape (N,)
    const float* __restrict__ input,        // x (saved from forward), shape (N,)
    float* __restrict__ grad_input,         // dL/dx, shape (N,)
    int N
) {
    // Grid-stride loop: each thread handles multiple elements if needed.
    // This is the standard CUDA pattern for element-wise operations.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        // The entire backward pass is ONE line:
        //   grad_input = grad_output * (input > 0 ? 1 : 0)
        //
        // We use a float cast of the boolean comparison. This compiles to
        // a predicated move on the GPU (no branch divergence).
        grad_input[i] = grad_output[i] * (input[i] > 0.0f ? 1.0f : 0.0f);
    }
}

// Host wrapper
Tensor<float> relu_backward(
    const Tensor<float>& grad_output,
    const Tensor<float>& input
) {
    int N = input.size_;
    Tensor<float> grad_input({N}, Device::GPU);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr(), input.data_ptr(), grad_input.data_ptr(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_input;
}

// CPU reference: ReLU forward (needed for finite differences)
void relu_forward_cpu(const float* input, float* output, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

// CPU reference: ReLU backward
void relu_backward_cpu(
    const float* grad_output, const float* input,
    float* grad_input, int N
) {
    for (int i = 0; i < N; i++) {
        grad_input[i] = grad_output[i] * (input[i] > 0.0f ? 1.0f : 0.0f);
    }
}

// ===========================================================================
//  2. LINEAR (FULLY CONNECTED) BACKWARD
// ===========================================================================
//
//  Forward:  y = x @ W^T + b
//
//    x is (B, K)  -- batch of input vectors
//    W is (J, K)  -- weight matrix (J outputs, K inputs)
//    b is (J,)    -- bias vector
//    y is (B, J)  -- batch of output vectors
//
//  The forward pass for a single batch element:
//
//    y[b][j] = SUM_{k=0}^{K-1} x[b][k] * W[j][k] + b[j]
//
//  ---- CHAIN RULE DERIVATIONS ----
//
//  (a) grad_input: dL/dx  shape (B, K)
//
//    dL/dx[b][k] = SUM_{j} dL/dy[b][j] * dy[b][j] / dx[b][k]
//                = SUM_{j} dL/dy[b][j] * W[j][k]
//
//    In matrix notation:  dL/dx = dL/dy @ W       ... shape (B,J) @ (J,K) = (B,K)
//
//  (b) grad_weight: dL/dW  shape (J, K)
//
//    dL/dW[j][k] = SUM_{b} dL/dy[b][j] * dy[b][j] / dW[j][k]
//                = SUM_{b} dL/dy[b][j] * x[b][k]
//
//    In matrix notation:  dL/dW = (dL/dy)^T @ x   ... shape (J,B) @ (B,K) = (J,K)
//
//  (c) grad_bias: dL/db  shape (J,)
//
//    dL/db[j] = SUM_{b} dL/dy[b][j]              ... just sum over batch dim
//
//  DIAGRAM:
//
//    x (B, K)          W^T (K, J)         b (J,)
//    +--------+        +--------+         +---+
//    | x[0,:] | --@--> |        | --+--> | b | ---> y (B, J)
//    | x[1,:] |        | W^T    |   |    +---+
//    |  ...   |        |        |   |
//    +--------+        +--------+   |
//                                   v
//    BACKWARD:                    y = x @ W^T + b
//
//    grad_input  = grad_output @ W         (undo the transpose)
//    grad_weight = grad_output^T @ x       (correlation of grad and input)
//    grad_bias   = sum(grad_output, dim=0)  (reduction over batch)
//
// ===========================================================================

// Kernel (a): grad_input = grad_output @ W
// Each thread computes one element of the (B, K) result.
__global__ void linear_backward_input_kernel(
    const float* __restrict__ grad_output,  // (B, J)
    const float* __restrict__ weight,       // (J, K)
    float* __restrict__ grad_input,         // (B, K)
    int B, int J, int K
) {
    int total = B * K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int k = i % K;       // which input feature
        int b = i / K;       // which batch element

        // dL/dx[b][k] = SUM_{j=0}^{J-1} dL/dy[b][j] * W[j][k]
        //
        // This is a dot product: row b of grad_output with column k of W.
        // (Equivalently, this is matmul: grad_output @ W.)
        float sum = 0.0f;
        for (int j = 0; j < J; j++) {
            sum += grad_output[b * J + j] * weight[j * K + k];
        }
        grad_input[i] = sum;
    }
}

// Kernel (b): grad_weight = grad_output^T @ input
// Each thread computes one element of the (J, K) result.
__global__ void linear_backward_weight_kernel(
    const float* __restrict__ grad_output,  // (B, J)
    const float* __restrict__ input,        // (B, K)
    float* __restrict__ grad_weight,        // (J, K)
    int B, int J, int K
) {
    int total = J * K;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int k = i % K;       // which input feature
        int j = i / K;       // which output feature

        // dL/dW[j][k] = SUM_{b=0}^{B-1} dL/dy[b][j] * x[b][k]
        //
        // This is a dot product: column j of grad_output^T with column k of input.
        // (Equivalently, this is matmul: grad_output^T @ input.)
        float sum = 0.0f;
        for (int b = 0; b < B; b++) {
            sum += grad_output[b * J + j] * input[b * K + k];
        }
        grad_weight[i] = sum;
    }
}

// Kernel (c): grad_bias = sum(grad_output, dim=0)
// Each thread computes one element of the (J,) result.
__global__ void linear_backward_bias_kernel(
    const float* __restrict__ grad_output,  // (B, J)
    float* __restrict__ grad_bias,          // (J,)
    int B, int J
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= J) return;

    // dL/db[j] = SUM_{b=0}^{B-1} dL/dy[b][j]
    //
    // Simple reduction over the batch dimension. In a production implementation
    // you'd use shared-memory reduction for large B, but for typical batch sizes
    // (8-128) a serial loop per thread is fine.
    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        sum += grad_output[b * J + j];
    }
    grad_bias[j] = sum;
}

// Result struct for linear backward (mirrors the three gradients)
struct LinearBackwardResult {
    Tensor<float> grad_input;   // (B, K)
    Tensor<float> grad_weight;  // (J, K)
    Tensor<float> grad_bias;    // (J,)
};

// Host wrapper
LinearBackwardResult linear_backward(
    const Tensor<float>& grad_output,  // (B, J)
    const Tensor<float>& input,        // (B, K)
    const Tensor<float>& weight,       // (J, K)
    int B, int J, int K
) {
    Tensor<float> grad_input({B, K}, Device::GPU);
    Tensor<float> grad_weight({J, K}, Device::GPU);
    Tensor<float> grad_bias({J}, Device::GPU);

    int threads = 256;

    // (a) grad_input = grad_output @ W
    {
        int total = B * K;
        int blocks = (total + threads - 1) / threads;
        linear_backward_input_kernel<<<blocks, threads>>>(
            grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr(),
            B, J, K);
        CUDA_CHECK(cudaGetLastError());
    }

    // (b) grad_weight = grad_output^T @ input
    {
        int total = J * K;
        int blocks = (total + threads - 1) / threads;
        linear_backward_weight_kernel<<<blocks, threads>>>(
            grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr(),
            B, J, K);
        CUDA_CHECK(cudaGetLastError());
    }

    // (c) grad_bias = sum(grad_output, dim=0)
    {
        int blocks = (J + threads - 1) / threads;
        linear_backward_bias_kernel<<<blocks, threads>>>(
            grad_output.data_ptr(), grad_bias.data_ptr(), B, J);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return {grad_input, grad_weight, grad_bias};
}

// CPU reference: Linear forward
void linear_forward_cpu(
    const float* x, const float* W, const float* b,
    float* y, int B, int J, int K
) {
    for (int batch = 0; batch < B; batch++) {
        for (int j = 0; j < J; j++) {
            float sum = b[j];
            for (int k = 0; k < K; k++) {
                sum += x[batch * K + k] * W[j * K + k];
            }
            y[batch * J + j] = sum;
        }
    }
}

// CPU reference: Linear backward
void linear_backward_cpu(
    const float* grad_output, const float* x, const float* W,
    float* grad_input, float* grad_weight, float* grad_bias,
    int B, int J, int K
) {
    // (a) grad_input = grad_output @ W
    for (int batch = 0; batch < B; batch++)
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int j = 0; j < J; j++)
                sum += grad_output[batch * J + j] * W[j * K + k];
            grad_input[batch * K + k] = sum;
        }

    // (b) grad_weight = grad_output^T @ x
    for (int j = 0; j < J; j++)
        for (int k = 0; k < K; k++) {
            float sum = 0.0f;
            for (int batch = 0; batch < B; batch++)
                sum += grad_output[batch * J + j] * x[batch * K + k];
            grad_weight[j * K + k] = sum;
        }

    // (c) grad_bias = sum(grad_output, dim=0)
    for (int j = 0; j < J; j++) {
        float sum = 0.0f;
        for (int batch = 0; batch < B; batch++)
            sum += grad_output[batch * J + j];
        grad_bias[j] = sum;
    }
}

// ===========================================================================
//  3. GLOBAL AVERAGE POOLING (GAP) BACKWARD
// ===========================================================================
//
//  Forward:  y[b][c] = (1 / (H * W)) * SUM_{h,w} x[b][c][h][w]
//
//    x is (B, C, H, W)  -- spatial feature map
//    y is (B, C)         -- one value per channel per batch element
//
//  The forward pass averages over all spatial positions within each channel.
//
//  ---- CHAIN RULE ----
//
//  Each x[b][c][h][w] contributes equally to y[b][c]:
//
//    dy[b][c] / dx[b][c][h][w] = 1 / (H * W)
//
//  Therefore:
//
//    dL/dx[b][c][h][w] = dL/dy[b][c] * 1 / (H * W)
//
//  The gradient is UNIFORMLY DISTRIBUTED to all spatial positions.
//  Every position in the HxW grid gets the SAME gradient value.
//
//  DIAGRAM (H=2, W=2, so HW=4):
//
//    grad_output (after GAP):  [0.8]  (one scalar per channel)
//
//    grad_input (before GAP):
//    +------+------+
//    | 0.2  | 0.2  |    Each position gets 0.8 / 4 = 0.2
//    +------+------+
//    | 0.2  | 0.2  |
//    +------+------+
//
//  Intuition: in the forward pass, every spatial position contributed
//  equally (1/HW) to the output. In the backward pass, every position
//  receives equal blame (1/HW of the upstream gradient).
//
//  No parameters: GAP has no learnable parameters.
//
// ===========================================================================

__global__ void gap_backward_kernel(
    const float* __restrict__ grad_output,  // (B, C) -- upstream gradient
    float* __restrict__ grad_input,         // (B, C, H, W) -- gradient to compute
    int B, int C, int H, int W
) {
    int total = B * C * H * W;
    int HW = H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float inv_HW = 1.0f / (float)(HW);

    for (int i = idx; i < total; i += stride) {
        // Decompose flat index i -> (b, c, h, w) in NCHW layout
        //
        // i = b * (C*H*W) + c * (H*W) + h * W + w
        //
        // We only need (b, c) to index grad_output, so:
        int bc = i / HW;    // = b * C + c
        // int b = bc / C;
        // int c = bc % C;

        // dL/dx[b][c][h][w] = dL/dy[b][c] / (H * W)
        //
        // Every spatial position within channel (b, c) gets the same value.
        grad_input[i] = grad_output[bc] * inv_HW;
    }
}

// Host wrapper
Tensor<float> gap_backward(
    const Tensor<float>& grad_output,  // (B, C)
    int B, int C, int H, int W
) {
    Tensor<float> grad_input({B, C, H, W}, Device::GPU);

    int total = B * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    gap_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr(), grad_input.data_ptr(), B, C, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_input;
}

// CPU reference: GAP forward
void gap_forward_cpu(const float* input, float* output, int B, int C, int H, int W) {
    int HW = H * W;
    for (int b = 0; b < B; b++)
        for (int c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int s = 0; s < HW; s++)
                sum += input[b * C * HW + c * HW + s];
            output[b * C + c] = sum / (float)HW;
        }
}

// CPU reference: GAP backward
void gap_backward_cpu(
    const float* grad_output, float* grad_input,
    int B, int C, int H, int W
) {
    int HW = H * W;
    float inv_HW = 1.0f / (float)HW;
    for (int b = 0; b < B; b++)
        for (int c = 0; c < C; c++)
            for (int s = 0; s < HW; s++)
                grad_input[b * C * HW + c * HW + s] = grad_output[b * C + c] * inv_HW;
}

// ===========================================================================
//  4. CROSS-ENTROPY LOSS BACKWARD
// ===========================================================================
//
//  Forward:
//    softmax(z)[j] = exp(z[j]) / SUM_k exp(z[k])     (numerically stable version)
//    L = -(1/B) * SUM_b log(softmax(logits[b])[target[b]])
//
//  The gradient of cross-entropy w.r.t. the logits has a beautifully
//  simple form. Let's derive it step by step.
//
//  For a single sample b (dropping the b index for clarity):
//
//    L_b = -log(softmax(z)[t])       where t = target label
//
//  Let p_j = softmax(z)[j] = exp(z_j) / SUM_k exp(z_k).
//
//    dL_b / dz_j = ?
//
//  Case 1: j != t
//    L_b = -log(p_t) = -z_t + log(SUM exp(z_k))
//    dL_b/dz_j = 0 + exp(z_j) / SUM exp(z_k) = p_j
//
//  Case 2: j == t
//    dL_b/dz_j = -1 + exp(z_j) / SUM exp(z_k) = -1 + p_j = p_j - 1
//
//  Combining:
//    dL_b / dz_j = p_j - (j == t ? 1 : 0)
//                = softmax(z)[j] - one_hot(t)[j]
//
//  For batch mean:
//    dL/dz[b][j] = (softmax(z[b])[j] - one_hot(target[b])[j]) / B
//
//  DIAGRAM:
//
//    logits z:    [2.0,  1.0,  0.1]
//    softmax p:   [0.659, 0.242, 0.099]     (sums to 1)
//    target = 0:  one_hot = [1, 0, 0]
//    grad:        [0.659-1, 0.242-0, 0.099-0]  =  [-0.341, 0.242, 0.099]
//    (divided by B for mean)
//
//  NUMERICAL STABILITY:
//  ====================
//  Naive softmax: exp(z) can overflow for large z. Solution: subtract max(z):
//    softmax(z)[j] = exp(z[j] - max(z)) / SUM_k exp(z[k] - max(z))
//  This is mathematically equivalent but numerically stable.
//
//  No parameters: the loss function has no learnable parameters.
//
// ===========================================================================

__global__ void cross_entropy_backward_kernel(
    const float* __restrict__ logits,       // (B, num_classes)
    const int* __restrict__ targets,        // (B,) -- integer class labels
    float* __restrict__ grad_logits,        // (B, num_classes) -- output gradient
    int B, int num_classes
) {
    // Each thread handles one batch element (computes all num_classes gradients)
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* z = logits + b * num_classes;
    float* g = grad_logits + b * num_classes;
    int target = targets[b];

    // Step 1: Find max logit for numerical stability
    //
    // Without this, exp(z[j]) can overflow to infinity for large logits.
    // Subtracting the max makes the largest exponent 0 (exp(0) = 1).
    float max_z = z[0];
    for (int j = 1; j < num_classes; j++) {
        if (z[j] > max_z) max_z = z[j];
    }

    // Step 2: Compute sum of exp(z[j] - max_z)
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        sum_exp += expf(z[j] - max_z);
    }

    // Step 3: Compute softmax and gradient in one pass
    //
    //   grad[b][j] = (softmax(z)[j] - one_hot(target)[j]) / B
    //
    // The division by B gives us the gradient of the MEAN cross-entropy loss.
    float inv_B = 1.0f / (float)B;
    for (int j = 0; j < num_classes; j++) {
        float softmax_j = expf(z[j] - max_z) / sum_exp;
        float one_hot_j = (j == target) ? 1.0f : 0.0f;
        g[j] = (softmax_j - one_hot_j) * inv_B;
    }
}

// Host wrapper
Tensor<float> cross_entropy_backward(
    const Tensor<float>& logits,    // (B, num_classes) on GPU
    const Tensor<float>& targets,   // (B,) int targets on GPU -- stored as float, cast to int
    int B, int num_classes
) {
    Tensor<float> grad_logits({B, num_classes}, Device::GPU);

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    // We need integer targets on GPU. The targets tensor stores floats
    // but we interpret them as ints inside the kernel by casting.
    // For a clean interface, we allocate int targets and copy.
    int* d_targets;
    CUDA_CHECK(cudaMalloc(&d_targets, B * sizeof(int)));

    // Copy float targets to CPU, convert to int, copy int to GPU
    Tensor<float> targets_cpu = targets.to_cpu();
    std::vector<int> h_targets(B);
    for (int b = 0; b < B; b++) {
        h_targets[b] = (int)targets_cpu.data_ptr()[b];
    }
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), B * sizeof(int),
                           cudaMemcpyHostToDevice));

    cross_entropy_backward_kernel<<<blocks, threads>>>(
        logits.data_ptr(), d_targets, grad_logits.data_ptr(), B, num_classes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_targets));
    return grad_logits;
}

// CPU reference: Cross-entropy forward (returns scalar loss)
float cross_entropy_forward_cpu(
    const float* logits, const int* targets,
    int B, int num_classes
) {
    float total_loss = 0.0f;
    for (int b = 0; b < B; b++) {
        const float* z = logits + b * num_classes;
        int t = targets[b];

        // Numerically stable log-softmax
        float max_z = z[0];
        for (int j = 1; j < num_classes; j++)
            if (z[j] > max_z) max_z = z[j];

        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++)
            sum_exp += expf(z[j] - max_z);

        float log_softmax_t = (z[t] - max_z) - logf(sum_exp);
        total_loss += -log_softmax_t;
    }
    return total_loss / (float)B;
}

// CPU reference: Cross-entropy backward
void cross_entropy_backward_cpu(
    const float* logits, const int* targets,
    float* grad_logits, int B, int num_classes
) {
    float inv_B = 1.0f / (float)B;
    for (int b = 0; b < B; b++) {
        const float* z = logits + b * num_classes;
        float* g = grad_logits + b * num_classes;
        int t = targets[b];

        float max_z = z[0];
        for (int j = 1; j < num_classes; j++)
            if (z[j] > max_z) max_z = z[j];

        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++)
            sum_exp += expf(z[j] - max_z);

        for (int j = 0; j < num_classes; j++) {
            float p_j = expf(z[j] - max_z) / sum_exp;
            g[j] = (p_j - (j == t ? 1.0f : 0.0f)) * inv_B;
        }
    }
}

// ===========================================================================
//  5. RESIDUAL (SKIP CONNECTION) BACKWARD
// ===========================================================================
//
//  Forward:  y = a + b     (element-wise addition)
//
//    a is (N,)  -- output of one branch (e.g., conv path)
//    b is (N,)  -- output of other branch (e.g., skip/identity path)
//    y is (N,)  -- sum of both branches
//
//  ---- CHAIN RULE ----
//
//  Since y = a + b, the partial derivatives are trivially:
//
//    dy/da = 1     (element-wise)
//    dy/db = 1     (element-wise)
//
//  Therefore:
//
//    dL/da = dL/dy * 1 = dL/dy    (gradient is COPIED to branch a)
//    dL/db = dL/dy * 1 = dL/dy    (gradient is COPIED to branch b)
//
//  DIAGRAM:
//
//                 +--> branch a (conv path) --+
//    input -------|                            +--> (+) ---> output
//                 +--> branch b (skip path) --+
//
//                          BACKWARD:
//
//                 +-- grad_a = grad_output <--+
//    grad_in <----|                            +--- grad_output
//                 +-- grad_b = grad_output <--+
//
//  At the branch SPLIT POINT, the gradients from both branches are SUMMED:
//    dL/d(input) = dL/da + dL/db = 2 * dL/dy
//    (if both branches originate from the same input)
//
//  This is the FUNDAMENTAL INSIGHT of ResNets: the skip connection provides
//  a "gradient highway." Even if the conv path has vanishing gradients
//  (due to many multiplications < 1), the identity path guarantees that
//  dL/d(input) >= dL/d(output). The gradient can always flow.
//
//  Implementation note: residual_backward is trivially just a copy.
//  In practice, you don't even need a kernel -- you just pass the same
//  gradient pointer to both branches. But we implement it explicitly
//  for clarity and to match the structure of the other backward ops.
//
//  No parameters: addition has no learnable parameters.
//
// ===========================================================================

__global__ void residual_backward_kernel(
    const float* __restrict__ grad_output,  // dL/dy, shape (N,)
    float* __restrict__ grad_a,             // dL/da = dL/dy, shape (N,)
    float* __restrict__ grad_b,             // dL/db = dL/dy, shape (N,)
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        // Both branches receive an identical copy of the upstream gradient.
        // This is the entire kernel: two copies.
        float g = grad_output[i];
        grad_a[i] = g;
        grad_b[i] = g;
    }
}

// Host wrapper
void residual_backward(
    const Tensor<float>& grad_output,
    Tensor<float>& grad_a,
    Tensor<float>& grad_b
) {
    int N = grad_output.size_;

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    residual_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr(), grad_a.data_ptr(), grad_b.data_ptr(), N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ===========================================================================
//  NUMERICAL GRADIENT CHECKING UTILITIES
// ===========================================================================
//
// The gold standard for verifying backward-pass correctness:
//
//   numerical_grad[i] = (f(x + eps) - f(x - eps)) / (2 * eps)
//
// This central difference approximation has O(eps^2) error, which for
// eps = 1e-3 gives ~1e-6 accuracy in float64 (but only ~1e-3 in float32
// due to limited mantissa precision).
//
// We compare analytical gradients (from our backward kernels) against
// numerical gradients. If max relative error < 1e-3, we're good.
//
// ===========================================================================

// Helper: compute scalar loss = sum(output) for a generic forward function
// (used in finite-difference checking)

// ===========================================================================
//  TEST MAIN
// ===========================================================================

int main() {
    printf("=== Chapter 15: Simple Backward Ops Test ===\n\n");

    // =====================================================================
    //  TEST 1: ReLU Backward
    // =====================================================================
    {
        printf("Test 1: ReLU Backward\n");

        int N = 64;
        Tensor<float> input_cpu = Tensor<float>::randn({N}, Device::CPU);

        // Scale to get a mix of positive and negative values
        for (int i = 0; i < N; i++)
            input_cpu.data_ptr()[i] *= 2.0f;

        Tensor<float> input_gpu = input_cpu.to_gpu();

        // Use grad_output = all 1s (gradient of sum(relu(x)) w.r.t. relu(x))
        Tensor<float> grad_output_gpu = Tensor<float>::ones({N}, Device::GPU);

        // --- Analytical backward ---
        Tensor<float> grad_input_gpu = relu_backward(grad_output_gpu, input_gpu);
        Tensor<float> grad_input_cpu = grad_input_gpu.to_cpu();

        // --- Numerical gradient (finite differences) ---
        // Loss = sum(relu(x))
        // dL/dx[i] = (sum(relu(x + eps*e_i)) - sum(relu(x - eps*e_i))) / (2*eps)
        float eps = 1e-3f;
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;

        for (int i = 0; i < N; i++) {
            float orig = input_cpu.data_ptr()[i];

            // f(x + eps)
            input_cpu.data_ptr()[i] = orig + eps;
            std::vector<float> out_p(N);
            relu_forward_cpu(input_cpu.data_ptr(), out_p.data(), N);
            float loss_p = 0.0f;
            for (int j = 0; j < N; j++) loss_p += out_p[j];

            // f(x - eps)
            input_cpu.data_ptr()[i] = orig - eps;
            std::vector<float> out_m(N);
            relu_forward_cpu(input_cpu.data_ptr(), out_m.data(), N);
            float loss_m = 0.0f;
            for (int j = 0; j < N; j++) loss_m += out_m[j];

            // Restore
            input_cpu.data_ptr()[i] = orig;

            float numerical = (loss_p - loss_m) / (2.0f * eps);
            float analytical = grad_input_cpu.data_ptr()[i];

            float abs_err = fabsf(numerical - analytical);
            float denom = fmaxf(fabsf(numerical) + fabsf(analytical), 1e-8f);
            float rel_err = abs_err / denom;

            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }

        printf("  Checked %d elements\n", N);
        printf("  Max absolute error: %.6e\n", max_abs_err);
        printf("  Max relative error: %.6e\n", max_rel_err);
        printf("  Result: %s\n\n",
               max_rel_err < 1e-3 ? "PASS" : "FAIL");
    }

    // =====================================================================
    //  TEST 2: Linear Backward
    // =====================================================================
    {
        printf("Test 2: Linear Backward\n");

        int B = 4, K = 8, J = 3;  // 4 samples, 8 input features, 3 outputs

        // Create random data
        Tensor<float> x_cpu = Tensor<float>::randn({B, K}, Device::CPU);
        Tensor<float> W_cpu = Tensor<float>::randn({J, K}, Device::CPU);
        Tensor<float> b_cpu = Tensor<float>::randn({J}, Device::CPU);

        // Scale down for numerical stability
        for (int i = 0; i < x_cpu.size_; i++) x_cpu.data_ptr()[i] *= 0.5f;
        for (int i = 0; i < W_cpu.size_; i++) W_cpu.data_ptr()[i] *= 0.5f;
        for (int i = 0; i < b_cpu.size_; i++) b_cpu.data_ptr()[i] *= 0.5f;

        // GPU forward + backward
        Tensor<float> x_gpu = x_cpu.to_gpu();
        Tensor<float> W_gpu = W_cpu.to_gpu();
        Tensor<float> b_gpu = b_cpu.to_gpu();
        Tensor<float> grad_output_gpu = Tensor<float>::ones({B, J}, Device::GPU);

        LinearBackwardResult bwd = linear_backward(grad_output_gpu, x_gpu, W_gpu, B, J, K);
        Tensor<float> gi_cpu = bwd.grad_input.to_cpu();
        Tensor<float> gw_cpu = bwd.grad_weight.to_cpu();
        Tensor<float> gb_cpu = bwd.grad_bias.to_cpu();

        // --- Numerical gradient: grad_input ---
        // Loss = sum(linear(x, W, b))
        float eps = 1e-3f;

        // Helper lambda: compute loss = sum(y) on CPU
        auto compute_linear_loss = [&](const float* x_data, const float* W_data,
                                       const float* b_data) -> float {
            std::vector<float> y(B * J);
            linear_forward_cpu(x_data, W_data, b_data, y.data(), B, J, K);
            float loss = 0.0f;
            for (int i = 0; i < B * J; i++) loss += y[i];
            return loss;
        };

        // (a) Check grad_input
        {
            float max_rel = 0.0f;
            for (int i = 0; i < B * K; i++) {
                float orig = x_cpu.data_ptr()[i];
                x_cpu.data_ptr()[i] = orig + eps;
                float lp = compute_linear_loss(x_cpu.data_ptr(), W_cpu.data_ptr(), b_cpu.data_ptr());
                x_cpu.data_ptr()[i] = orig - eps;
                float lm = compute_linear_loss(x_cpu.data_ptr(), W_cpu.data_ptr(), b_cpu.data_ptr());
                x_cpu.data_ptr()[i] = orig;

                float num = (lp - lm) / (2.0f * eps);
                float ana = gi_cpu.data_ptr()[i];
                float rel = fabsf(num - ana) / fmaxf(fabsf(num) + fabsf(ana), 1e-8f);
                if (rel > max_rel) max_rel = rel;
            }
            printf("  grad_input  max rel error: %.6e -- %s\n",
                   max_rel, max_rel < 2e-3 ? "PASS" : "FAIL");
        }

        // (b) Check grad_weight
        {
            float max_rel = 0.0f;
            for (int i = 0; i < J * K; i++) {
                float orig = W_cpu.data_ptr()[i];
                W_cpu.data_ptr()[i] = orig + eps;
                float lp = compute_linear_loss(x_cpu.data_ptr(), W_cpu.data_ptr(), b_cpu.data_ptr());
                W_cpu.data_ptr()[i] = orig - eps;
                float lm = compute_linear_loss(x_cpu.data_ptr(), W_cpu.data_ptr(), b_cpu.data_ptr());
                W_cpu.data_ptr()[i] = orig;

                float num = (lp - lm) / (2.0f * eps);
                float ana = gw_cpu.data_ptr()[i];
                float rel = fabsf(num - ana) / fmaxf(fabsf(num) + fabsf(ana), 1e-8f);
                if (rel > max_rel) max_rel = rel;
            }
            printf("  grad_weight max rel error: %.6e -- %s\n",
                   max_rel, max_rel < 2e-3 ? "PASS" : "FAIL");
        }

        // (c) Check grad_bias
        {
            float max_rel = 0.0f;
            for (int j = 0; j < J; j++) {
                float orig = b_cpu.data_ptr()[j];
                b_cpu.data_ptr()[j] = orig + eps;
                float lp = compute_linear_loss(x_cpu.data_ptr(), W_cpu.data_ptr(), b_cpu.data_ptr());
                b_cpu.data_ptr()[j] = orig - eps;
                float lm = compute_linear_loss(x_cpu.data_ptr(), W_cpu.data_ptr(), b_cpu.data_ptr());
                b_cpu.data_ptr()[j] = orig;

                float num = (lp - lm) / (2.0f * eps);
                float ana = gb_cpu.data_ptr()[j];
                float rel = fabsf(num - ana) / fmaxf(fabsf(num) + fabsf(ana), 1e-8f);
                if (rel > max_rel) max_rel = rel;
            }
            printf("  grad_bias   max rel error: %.6e -- %s\n",
                   max_rel, max_rel < 2e-3 ? "PASS" : "FAIL");
        }

        // Sanity check: grad_bias with all-ones upstream should equal B
        {
            bool pass = true;
            for (int j = 0; j < J; j++) {
                if (fabsf(gb_cpu.data_ptr()[j] - (float)B) > 1e-5f) pass = false;
            }
            printf("  grad_bias = B = %d sanity: %s\n\n", B, pass ? "PASS" : "FAIL");
        }
    }

    // =====================================================================
    //  TEST 3: Global Average Pooling Backward
    // =====================================================================
    {
        printf("Test 3: GAP Backward\n");

        int B = 2, C = 3, H = 4, W = 4;
        int total = B * C * H * W;

        Tensor<float> input_cpu = Tensor<float>::randn({B, C, H, W}, Device::CPU);
        for (int i = 0; i < total; i++)
            input_cpu.data_ptr()[i] *= 0.5f;

        // grad_output = all 1s (loss = sum(gap(x)))
        Tensor<float> grad_output_gpu = Tensor<float>::ones({B, C}, Device::GPU);

        // Analytical backward
        Tensor<float> grad_input_gpu = gap_backward(grad_output_gpu, B, C, H, W);
        Tensor<float> grad_input_cpu = grad_input_gpu.to_cpu();

        // Numerical gradient
        float eps = 1e-3f;
        float max_rel = 0.0f;

        for (int i = 0; i < total; i++) {
            float orig = input_cpu.data_ptr()[i];

            // f(x + eps)
            input_cpu.data_ptr()[i] = orig + eps;
            std::vector<float> gap_out_p(B * C);
            gap_forward_cpu(input_cpu.data_ptr(), gap_out_p.data(), B, C, H, W);
            float lp = 0.0f;
            for (int j = 0; j < B * C; j++) lp += gap_out_p[j];

            // f(x - eps)
            input_cpu.data_ptr()[i] = orig - eps;
            std::vector<float> gap_out_m(B * C);
            gap_forward_cpu(input_cpu.data_ptr(), gap_out_m.data(), B, C, H, W);
            float lm = 0.0f;
            for (int j = 0; j < B * C; j++) lm += gap_out_m[j];

            input_cpu.data_ptr()[i] = orig;

            float num = (lp - lm) / (2.0f * eps);
            float ana = grad_input_cpu.data_ptr()[i];
            float rel = fabsf(num - ana) / fmaxf(fabsf(num) + fabsf(ana), 1e-8f);
            if (rel > max_rel) max_rel = rel;
        }

        printf("  Checked %d elements\n", total);
        printf("  Max relative error: %.6e\n", max_rel);
        printf("  Result: %s\n", max_rel < 2e-3 ? "PASS" : "FAIL");

        // Sanity: each element should be 1/(H*W) = 1/16 = 0.0625
        float expected = 1.0f / (float)(H * W);
        bool sanity = true;
        for (int i = 0; i < total; i++) {
            if (fabsf(grad_input_cpu.data_ptr()[i] - expected) > 1e-6f)
                sanity = false;
        }
        printf("  All grads = 1/HW = %.4f sanity: %s\n\n",
               expected, sanity ? "PASS" : "FAIL");
    }

    // =====================================================================
    //  TEST 4: Cross-Entropy Backward
    // =====================================================================
    {
        printf("Test 4: Cross-Entropy Backward\n");

        int B = 4, num_classes = 5;

        // Random logits
        Tensor<float> logits_cpu = Tensor<float>::randn({B, num_classes}, Device::CPU);
        for (int i = 0; i < B * num_classes; i++)
            logits_cpu.data_ptr()[i] *= 0.5f;

        // Random targets (integers 0..num_classes-1)
        int h_targets[] = {2, 0, 4, 1};

        // Store targets as float for our Tensor class
        float h_targets_f[4];
        for (int b = 0; b < B; b++) h_targets_f[b] = (float)h_targets[b];

        Tensor<float> targets_gpu({B}, h_targets_f, Device::GPU);
        Tensor<float> logits_gpu = logits_cpu.to_gpu();

        // Analytical backward
        Tensor<float> grad_logits_gpu = cross_entropy_backward(
            logits_gpu, targets_gpu, B, num_classes);
        Tensor<float> grad_logits_cpu = grad_logits_gpu.to_cpu();

        // Numerical gradient
        float eps = 1e-3f;
        float max_rel = 0.0f;

        for (int i = 0; i < B * num_classes; i++) {
            float orig = logits_cpu.data_ptr()[i];

            logits_cpu.data_ptr()[i] = orig + eps;
            float lp = cross_entropy_forward_cpu(logits_cpu.data_ptr(), h_targets,
                                                  B, num_classes);

            logits_cpu.data_ptr()[i] = orig - eps;
            float lm = cross_entropy_forward_cpu(logits_cpu.data_ptr(), h_targets,
                                                  B, num_classes);

            logits_cpu.data_ptr()[i] = orig;

            float num = (lp - lm) / (2.0f * eps);
            float ana = grad_logits_cpu.data_ptr()[i];
            float rel = fabsf(num - ana) / fmaxf(fabsf(num) + fabsf(ana), 1e-8f);
            if (rel > max_rel) max_rel = rel;
        }

        printf("  Checked %d elements\n", B * num_classes);
        printf("  Max relative error: %.6e\n", max_rel);
        printf("  Result: %s\n", max_rel < 2e-3 ? "PASS" : "FAIL");

        // Sanity: gradients per sample should sum to 0
        // (softmax sums to 1, one_hot sums to 1, so their difference sums to 0)
        bool sum_zero = true;
        for (int b = 0; b < B; b++) {
            float row_sum = 0.0f;
            for (int j = 0; j < num_classes; j++)
                row_sum += grad_logits_cpu.data_ptr()[b * num_classes + j];
            if (fabsf(row_sum) > 1e-5f) sum_zero = false;
        }
        printf("  Row sums ~ 0 (softmax property): %s\n\n",
               sum_zero ? "PASS" : "FAIL");
    }

    // =====================================================================
    //  TEST 5: Residual Backward
    // =====================================================================
    {
        printf("Test 5: Residual Backward\n");

        int N = 32;

        Tensor<float> grad_output_gpu = Tensor<float>::randn({N}, Device::GPU);
        Tensor<float> grad_a_gpu({N}, Device::GPU);
        Tensor<float> grad_b_gpu({N}, Device::GPU);

        residual_backward(grad_output_gpu, grad_a_gpu, grad_b_gpu);

        Tensor<float> go_cpu = grad_output_gpu.to_cpu();
        Tensor<float> ga_cpu = grad_a_gpu.to_cpu();
        Tensor<float> gb_cpu = grad_b_gpu.to_cpu();

        // Both grad_a and grad_b should be exact copies of grad_output
        bool pass_a = true, pass_b = true;
        float max_err_a = 0.0f, max_err_b = 0.0f;

        for (int i = 0; i < N; i++) {
            float err_a = fabsf(ga_cpu.data_ptr()[i] - go_cpu.data_ptr()[i]);
            float err_b = fabsf(gb_cpu.data_ptr()[i] - go_cpu.data_ptr()[i]);
            if (err_a > 1e-7f) pass_a = false;
            if (err_b > 1e-7f) pass_b = false;
            if (err_a > max_err_a) max_err_a = err_a;
            if (err_b > max_err_b) max_err_b = err_b;
        }

        printf("  grad_a == grad_output: %s (max err: %.2e)\n",
               pass_a ? "PASS" : "FAIL", max_err_a);
        printf("  grad_b == grad_output: %s (max err: %.2e)\n",
               pass_b ? "PASS" : "FAIL", max_err_b);

        // Numerical check: y = a + b, loss = sum(y) = sum(a) + sum(b)
        // dL/da[i] = 1 for all i (since grad_output = dL/dy = 1 for loss = sum)
        // But we used random grad_output, so we just verify the copy.

        // Also verify: if we add the two branch gradients at the split point,
        // we get 2x the original gradient (both branches contribute)
        float sum_combined = 0.0f;
        float sum_original = 0.0f;
        for (int i = 0; i < N; i++) {
            sum_combined += ga_cpu.data_ptr()[i] + gb_cpu.data_ptr()[i];
            sum_original += go_cpu.data_ptr()[i];
        }
        bool double_check = fabsf(sum_combined - 2.0f * sum_original) < 1e-4f;
        printf("  sum(grad_a + grad_b) == 2 * sum(grad_output): %s\n\n",
               double_check ? "PASS" : "FAIL");
    }

    printf("=== Simple Backward Ops Tests Complete ===\n");
    return 0;
}
