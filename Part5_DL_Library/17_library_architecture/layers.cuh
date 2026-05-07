/*******************************************************************************
 * layers.cuh — Neural Network Layers for cudalearn
 *
 * Every layer inherits from Module and implements forward().
 * Each forward() builds the autograd computation graph by creating new
 * GradTensors with backward_fn lambdas that propagate gradients.
 *
 * Layers implemented:
 *   - Conv2d:         2D convolution with optional bias, padding, stride
 *   - BatchNorm2d:    batch normalization with running statistics
 *   - ReLU:           rectified linear unit (stateless)
 *   - Linear:         fully connected layer
 *   - GlobalAvgPool2d: global average pooling (stateless)
 *   - Sequential:     chain of modules
 *
 * All CUDA kernels are defined inline (self-contained chapter).
 *
 * Weight initialization:
 *   - Conv2d, Linear: Kaiming (He) initialization
 *   - BatchNorm2d:    gamma=1, beta=0
 ******************************************************************************/

#ifndef CUDALEARN_LAYERS_CUH
#define CUDALEARN_LAYERS_CUH

#include "module.cuh"
#include <curand.h>
#include <curand_kernel.h>

// =============================================================================
// CUDA Kernels for Layer Operations
// =============================================================================
// These kernels implement the forward and backward passes for each layer.
// They are simple, correct implementations — not optimized for peak throughput
// but clear enough to understand what each layer does mathematically.
// =============================================================================

// -----------------------------------------------------------------------------
// Kernel: Conv2d forward
// -----------------------------------------------------------------------------
// For each output element (n, co, oh, ow):
//   out[n][co][oh][ow] = sum over ci, kh, kw of
//       input[n][ci][oh*stride + kh - pad][ow*stride + kw - pad]
//       * weight[co][ci][kh][kw]
//   + bias[co]  (if bias is not null)
//
// Thread mapping: one thread per output element.
// Total threads: N * C_out * H_out * W_out
// -----------------------------------------------------------------------------
__global__ void conv2d_forward_kernel(
    const float* input,     // [N, C_in, H_in, W_in]
    const float* weight,    // [C_out, C_in, K, K]
    const float* bias,      // [C_out] or nullptr
    float* output,          // [N, C_out, H_out, W_out]
    int N, int C_in, int H_in, int W_in,
    int C_out, int K, int pad, int stride,
    int H_out, int W_out)
{
    // Global thread index — each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    // Decompose linear index into (n, co, oh, ow)
    int ow = idx % W_out;
    int oh = (idx / W_out) % H_out;
    int co = (idx / (W_out * H_out)) % C_out;
    int n  = idx / (W_out * H_out * C_out);

    // Accumulate the convolution sum
    float sum = 0.0f;
    for (int ci = 0; ci < C_in; ci++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                // Input coordinates with padding
                int ih = oh * stride + kh - pad;
                int iw = ow * stride + kw - pad;

                // Zero-padding: skip if outside input bounds
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                    int input_idx = ((n * C_in + ci) * H_in + ih) * W_in + iw;
                    int weight_idx = ((co * C_in + ci) * K + kh) * K + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    // Add bias if present
    if (bias) sum += bias[co];

    output[idx] = sum;
}

// -----------------------------------------------------------------------------
// Kernel: Conv2d backward w.r.t. input (data gradient)
// -----------------------------------------------------------------------------
// For each input element (n, ci, ih, iw), accumulate contributions from all
// output elements that used this input element in the forward pass.
//
// d_input[n][ci][ih][iw] = sum over co, kh, kw of
//     d_output[n][co][(ih+pad-kh)/stride][(iw+pad-kw)/stride]
//     * weight[co][ci][kh][kw]
//   (only when (ih+pad-kh) and (iw+pad-kw) are divisible by stride and in range)
// -----------------------------------------------------------------------------
__global__ void conv2d_backward_input_kernel(
    const float* d_output,  // [N, C_out, H_out, W_out]
    const float* weight,    // [C_out, C_in, K, K]
    float* d_input,         // [N, C_in, H_in, W_in]
    int N, int C_in, int H_in, int W_in,
    int C_out, int K, int pad, int stride,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_in * H_in * W_in;
    if (idx >= total) return;

    int iw = idx % W_in;
    int ih = (idx / W_in) % H_in;
    int ci = (idx / (W_in * H_in)) % C_in;
    int n  = idx / (W_in * H_in * C_in);

    float sum = 0.0f;
    for (int co = 0; co < C_out; co++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                // Which output element used this input with this kernel position?
                int oh_num = ih + pad - kh;
                int ow_num = iw + pad - kw;

                // Check divisibility by stride and range
                if (oh_num % stride == 0 && ow_num % stride == 0) {
                    int oh = oh_num / stride;
                    int ow = ow_num / stride;
                    if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                        int d_out_idx = ((n * C_out + co) * H_out + oh) * W_out + ow;
                        int w_idx = ((co * C_in + ci) * K + kh) * K + kw;
                        sum += d_output[d_out_idx] * weight[w_idx];
                    }
                }
            }
        }
    }
    d_input[idx] += sum;  // += because gradients accumulate
}

// -----------------------------------------------------------------------------
// Kernel: Conv2d backward w.r.t. weight (weight gradient)
// -----------------------------------------------------------------------------
// For each weight element (co, ci, kh, kw):
//   d_weight[co][ci][kh][kw] = sum over n, oh, ow of
//       d_output[n][co][oh][ow] * input[n][ci][oh*stride+kh-pad][ow*stride+kw-pad]
// -----------------------------------------------------------------------------
__global__ void conv2d_backward_weight_kernel(
    const float* d_output,  // [N, C_out, H_out, W_out]
    const float* input,     // [N, C_in, H_in, W_in]
    float* d_weight,        // [C_out, C_in, K, K]
    int N, int C_in, int H_in, int W_in,
    int C_out, int K, int pad, int stride,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C_out * C_in * K * K;
    if (idx >= total) return;

    int kw = idx % K;
    int kh = (idx / K) % K;
    int ci = (idx / (K * K)) % C_in;
    int co = idx / (K * K * C_in);

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                int ih = oh * stride + kh - pad;
                int iw = ow * stride + kw - pad;
                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                    int d_out_idx = ((n * C_out + co) * H_out + oh) * W_out + ow;
                    int in_idx = ((n * C_in + ci) * H_in + ih) * W_in + iw;
                    sum += d_output[d_out_idx] * input[in_idx];
                }
            }
        }
    }
    d_weight[idx] += sum;
}

// -----------------------------------------------------------------------------
// Kernel: Conv2d backward w.r.t. bias
// -----------------------------------------------------------------------------
// d_bias[co] = sum over n, oh, ow of d_output[n][co][oh][ow]
// -----------------------------------------------------------------------------
__global__ void conv2d_backward_bias_kernel(
    const float* d_output,  // [N, C_out, H_out, W_out]
    float* d_bias,          // [C_out]
    int N, int C_out, int H_out, int W_out)
{
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= C_out) return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                sum += d_output[((n * C_out + co) * H_out + oh) * W_out + ow];
            }
        }
    }
    d_bias[co] += sum;
}


// -----------------------------------------------------------------------------
// Kernel: BatchNorm2d forward (training mode)
// -----------------------------------------------------------------------------
// For each channel c:
//   mean[c] = average of all (n, h, w) values in channel c
//   var[c]  = variance of all (n, h, w) values in channel c
//   x_hat[n][c][h][w] = (x[n][c][h][w] - mean[c]) / sqrt(var[c] + eps)
//   out[n][c][h][w]   = gamma[c] * x_hat[n][c][h][w] + beta[c]
//
// Also updates running_mean and running_var with exponential moving average.
// -----------------------------------------------------------------------------

// Step 1: Compute per-channel mean
__global__ void batchnorm_mean_kernel(
    const float* input,     // [N, C, H, W]
    float* mean,            // [C]
    int N, int C, int H, int W)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    int count = N * H * W;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += input[((n * C + c) * H + h) * W + w];
            }
        }
    }
    mean[c] = sum / count;
}

// Step 2: Compute per-channel variance
__global__ void batchnorm_var_kernel(
    const float* input,     // [N, C, H, W]
    const float* mean,      // [C]
    float* var,             // [C]
    int N, int C, int H, int W)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float sum = 0.0f;
    int count = N * H * W;
    float m = mean[c];
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float diff = input[((n * C + c) * H + h) * W + w] - m;
                sum += diff * diff;
            }
        }
    }
    var[c] = sum / count;
}

// Step 3: Normalize and apply affine transform (gamma, beta)
__global__ void batchnorm_normalize_kernel(
    const float* input,     // [N, C, H, W]
    const float* mean,      // [C]
    const float* var,       // [C]
    const float* gamma,     // [C]
    const float* beta,      // [C]
    float* output,          // [N, C, H, W]
    float* x_hat,           // [N, C, H, W] — saved for backward
    float eps,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // Decompose index to find which channel this element belongs to
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;

    // Normalize: x_hat = (x - mean) / sqrt(var + eps)
    float inv_std = 1.0f / sqrtf(var[c] + eps);
    float xh = (input[idx] - mean[c]) * inv_std;
    x_hat[idx] = xh;

    // Affine: y = gamma * x_hat + beta
    output[idx] = gamma[c] * xh + beta[c];
}

// Update running statistics: running = momentum * running + (1-momentum) * batch
__global__ void batchnorm_update_running_kernel(
    float* running,         // [C] running_mean or running_var
    const float* batch,     // [C] batch mean or variance
    float momentum,
    int C)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    running[c] = momentum * running[c] + (1.0f - momentum) * batch[c];
}

// BatchNorm backward w.r.t. input, gamma, beta
// This is the most complex backward pass — see the BatchNorm paper for derivation.
//
// Given d_output (upstream gradient), compute:
//   d_gamma[c] = sum_{n,h,w} d_output[n,c,h,w] * x_hat[n,c,h,w]
//   d_beta[c]  = sum_{n,h,w} d_output[n,c,h,w]
//   d_input:    follows from chain rule through mean and variance
__global__ void batchnorm_backward_gamma_beta_kernel(
    const float* d_output,  // [N, C, H, W]
    const float* x_hat,     // [N, C, H, W]
    float* d_gamma,         // [C]
    float* d_beta,          // [C]
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
    d_gamma[c] += dg;
    d_beta[c] += db;
}

__global__ void batchnorm_backward_input_kernel(
    const float* d_output,  // [N, C, H, W]
    const float* x_hat,     // [N, C, H, W]
    const float* var,       // [C]
    const float* gamma,     // [C]
    float* d_input,         // [N, C, H, W]
    float eps,
    int N, int C, int H, int W)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    int M = N * H * W;  // Number of elements per channel
    float inv_std = 1.0f / sqrtf(var[c] + eps);

    // Compute intermediate sums needed for the backward formula
    float sum_dy = 0.0f;       // sum of d_output over (n,h,w)
    float sum_dy_xhat = 0.0f;  // sum of d_output * x_hat over (n,h,w)

    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = ((n * C + c) * H + h) * W + w;
                sum_dy += d_output[idx];
                sum_dy_xhat += d_output[idx] * x_hat[idx];
            }
        }
    }

    // Compute d_input for each element in this channel
    // Formula: d_input = gamma * inv_std / M * (M * dy - sum_dy - x_hat * sum_dy_xhat)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = ((n * C + c) * H + h) * W + w;
                d_input[idx] += gamma[c] * inv_std / M *
                    (M * d_output[idx] - sum_dy - x_hat[idx] * sum_dy_xhat);
            }
        }
    }
}


// -----------------------------------------------------------------------------
// Kernel: ReLU forward and backward
// -----------------------------------------------------------------------------
// Forward: out = max(0, x)
// Backward: d_input = d_output * (x > 0 ? 1 : 0)
// -----------------------------------------------------------------------------
__global__ void relu_forward_kernel(
    const float* input,
    float* output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = fmaxf(0.0f, input[idx]);
}

__global__ void relu_backward_kernel(
    const float* d_output,  // upstream gradient
    const float* input,     // original input (to check sign)
    float* d_input,         // gradient to accumulate into
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    // ReLU derivative: 1 if input > 0, else 0
    d_input[idx] += (input[idx] > 0.0f) ? d_output[idx] : 0.0f;
}


// -----------------------------------------------------------------------------
// Kernel: Linear (fully connected) forward and backward
// -----------------------------------------------------------------------------
// Forward: output = input * weight^T + bias
//   input:  [N, in_features]
//   weight: [out_features, in_features]
//   bias:   [out_features]
//   output: [N, out_features]
//
// This is a matrix multiply: output[n][o] = sum_i input[n][i] * weight[o][i] + bias[o]
// -----------------------------------------------------------------------------
__global__ void linear_forward_kernel(
    const float* input,     // [N, in_features]
    const float* weight,    // [out_features, in_features]
    const float* bias,      // [out_features] or nullptr
    float* output,          // [N, out_features]
    int N, int in_f, int out_f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_f;
    if (idx >= total) return;

    int o = idx % out_f;
    int n = idx / out_f;

    float sum = 0.0f;
    for (int i = 0; i < in_f; i++) {
        sum += input[n * in_f + i] * weight[o * in_f + i];
    }
    if (bias) sum += bias[o];
    output[idx] = sum;
}

// Backward w.r.t. input: d_input = d_output * weight
__global__ void linear_backward_input_kernel(
    const float* d_output,  // [N, out_features]
    const float* weight,    // [out_features, in_features]
    float* d_input,         // [N, in_features]
    int N, int in_f, int out_f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * in_f;
    if (idx >= total) return;

    int i = idx % in_f;
    int n = idx / in_f;

    float sum = 0.0f;
    for (int o = 0; o < out_f; o++) {
        sum += d_output[n * out_f + o] * weight[o * in_f + i];
    }
    d_input[idx] += sum;
}

// Backward w.r.t. weight: d_weight[o][i] = sum_n d_output[n][o] * input[n][i]
__global__ void linear_backward_weight_kernel(
    const float* d_output,  // [N, out_features]
    const float* input,     // [N, in_features]
    float* d_weight,        // [out_features, in_features]
    int N, int in_f, int out_f)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_f * in_f;
    if (idx >= total) return;

    int i = idx % in_f;
    int o = idx / in_f;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += d_output[n * out_f + o] * input[n * in_f + i];
    }
    d_weight[idx] += sum;
}

// Backward w.r.t. bias: d_bias[o] = sum_n d_output[n][o]
__global__ void linear_backward_bias_kernel(
    const float* d_output,  // [N, out_features]
    float* d_bias,          // [out_features]
    int N, int out_f)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_f) return;

    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        sum += d_output[n * out_f + o];
    }
    d_bias[o] += sum;
}


// -----------------------------------------------------------------------------
// Kernel: Global Average Pooling 2D
// -----------------------------------------------------------------------------
// Forward: for each (n, c), output = mean over (h, w) of input[n][c][h][w]
//   input:  [N, C, H, W]
//   output: [N, C]  (or equivalently [N, C, 1, 1])
//
// Backward: d_input[n][c][h][w] = d_output[n][c] / (H * W)
//   (gradient is uniformly distributed because average is a uniform sum)
// -----------------------------------------------------------------------------
__global__ void global_avg_pool_forward_kernel(
    const float* input,     // [N, C, H, W]
    float* output,          // [N, C]
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;

    int c = idx % C;
    int n = idx / C;

    float sum = 0.0f;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            sum += input[((n * C + c) * H + h) * W + w];
        }
    }
    output[idx] = sum / (H * W);
}

__global__ void global_avg_pool_backward_kernel(
    const float* d_output,  // [N, C]
    float* d_input,         // [N, C, H, W]
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    // Each input element gets an equal share of the output gradient
    d_input[idx] += d_output[n * C + c] / (H * W);
}


// -----------------------------------------------------------------------------
// Kernel: Kaiming (He) initialization
// -----------------------------------------------------------------------------
// Fills a tensor with values drawn from N(0, sqrt(2/fan_in)).
// fan_in = number of input connections per neuron.
// For Conv2d: fan_in = C_in * K * K
// For Linear: fan_in = in_features
// -----------------------------------------------------------------------------
__global__ void kaiming_init_kernel(
    float* data,
    int size,
    float std_dev,
    unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // Use cuRAND device API for random normal generation
    curandState state;
    curand_init(seed, idx, 0, &state);
    data[idx] = curand_normal(&state) * std_dev;
}


// =============================================================================
// Layer: Conv2d
// =============================================================================
// 2D convolution layer.
//
// Parameters:
//   - weight: [C_out, C_in, K, K] — learnable convolution filters
//   - bias:   [C_out]             — optional learnable bias (one per output channel)
//
// Hyperparameters:
//   - kernel_size (K): size of the square convolution kernel
//   - padding:         zero-padding added to input borders
//   - stride:          step size of the convolution
//
// Initialization: Kaiming (He) — std = sqrt(2 / (C_in * K * K))
// =============================================================================

class Conv2d : public Module {
public:
    int in_channels_, out_channels_, kernel_size_, padding_, stride_;
    GradTensor* weight_;    // [C_out, C_in, K, K]
    GradTensor* bias_;      // [C_out] or nullptr
    bool use_bias_;

    Conv2d(int in_channels, int out_channels, int kernel_size,
           int padding = 0, int stride = 1, bool use_bias = true)
        : in_channels_(in_channels), out_channels_(out_channels),
          kernel_size_(kernel_size), padding_(padding), stride_(stride),
          use_bias_(use_bias)
    {
        // Set descriptive name
        char buf[128];
        snprintf(buf, sizeof(buf), "Conv2d(%d, %d, kernel_size=%d, padding=%d, stride=%d)",
                 in_channels, out_channels, kernel_size, padding, stride);
        name_ = buf;

        // Create weight parameter: [C_out, C_in, K, K]
        weight_ = new GradTensor(out_channels, in_channels, kernel_size, kernel_size, true);
        register_parameter("weight", weight_);

        // Kaiming initialization: std = sqrt(2 / fan_in)
        int fan_in = in_channels * kernel_size * kernel_size;
        float std_dev = sqrtf(2.0f / fan_in);
        int blocks = (weight_->size + 255) / 256;
        kaiming_init_kernel<<<blocks, 256>>>(weight_->data, weight_->size, std_dev, 42);

        // Optional bias
        if (use_bias) {
            bias_ = new GradTensor(out_channels, 1, 1, 1, true);
            register_parameter("bias", bias_);
            // Bias initialized to zero (already done by GradTensor constructor)
        } else {
            bias_ = nullptr;
        }

        cudaDeviceSynchronize();
    }

    ~Conv2d() {
        delete weight_;
        if (bias_) delete bias_;
    }

    // forward() builds the computation graph and returns the output tensor.
    // The output tensor's backward_fn, when called, computes gradients for
    // weight, bias, and input — and propagates them.
    GradTensor* forward(GradTensor* input) override {
        int N = input->dims[0];
        int C_in = input->dims[1];
        int H_in = input->dims[2];
        int W_in = input->dims[3];
        int K = kernel_size_;

        // Compute output spatial dimensions
        int H_out = (H_in + 2 * padding_ - K) / stride_ + 1;
        int W_out = (W_in + 2 * padding_ - K) / stride_ + 1;

        // Allocate output tensor (no requires_grad — it gets grad from backward)
        GradTensor* output = new GradTensor(N, out_channels_, H_out, W_out, false);

        // Allocate gradient storage for output (needed for backward)
        cudaMalloc(&output->grad, output->size * sizeof(float));
        cudaMemset(output->grad, 0, output->size * sizeof(float));

        // Launch forward kernel
        int total = N * out_channels_ * H_out * W_out;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        conv2d_forward_kernel<<<blocks, threads>>>(
            input->data, weight_->data, use_bias_ ? bias_->data : nullptr,
            output->data,
            N, C_in, H_in, W_in, out_channels_, K, padding_, stride_,
            H_out, W_out
        );

        // ---------- Set up backward function ----------
        // Capture all needed pointers and dimensions by value (they're ints/ptrs)
        output->parents = {input, weight_};
        if (bias_) output->parents.push_back(bias_);

        // We need the input data for weight gradient computation
        // In a production library, we'd save intermediate tensors properly.
        // Here we capture the pointer (input must outlive backward).
        GradTensor* w = weight_;
        GradTensor* b = bias_;
        int pad = padding_;
        int str = stride_;
        int cout = out_channels_;

        output->backward_fn = [output, input, w, b, N, C_in, H_in, W_in,
                                cout, K, pad, str, H_out, W_out]() {
            int threads = 256;

            // 1. Gradient w.r.t. input (if input needs grad or has parents)
            if (input->grad) {
                int total_in = N * C_in * H_in * W_in;
                int blocks_in = (total_in + threads - 1) / threads;
                conv2d_backward_input_kernel<<<blocks_in, threads>>>(
                    output->grad, w->data, input->grad,
                    N, C_in, H_in, W_in, cout, K, pad, str, H_out, W_out
                );
            }

            // 2. Gradient w.r.t. weight (always — weight is a parameter)
            {
                int total_w = cout * C_in * K * K;
                int blocks_w = (total_w + threads - 1) / threads;
                conv2d_backward_weight_kernel<<<blocks_w, threads>>>(
                    output->grad, input->data, w->grad,
                    N, C_in, H_in, W_in, cout, K, pad, str, H_out, W_out
                );
            }

            // 3. Gradient w.r.t. bias (if bias exists)
            if (b) {
                int blocks_b = (cout + threads - 1) / threads;
                conv2d_backward_bias_kernel<<<blocks_b, threads>>>(
                    output->grad, b->grad, N, cout, H_out, W_out
                );
            }
        };

        return output;
    }
};


// =============================================================================
// Layer: BatchNorm2d
// =============================================================================
// Batch normalization for 2D inputs (4D tensors: N, C, H, W).
//
// Parameters (learnable):
//   - gamma: [C] — scale factor (initialized to 1)
//   - beta:  [C] — shift factor (initialized to 0)
//
// Buffers (non-learnable, updated during training):
//   - running_mean: [C] — exponential moving average of batch means
//   - running_var:  [C] — exponential moving average of batch variances
//
// In training mode: normalize using batch statistics, update running stats.
// In inference mode: normalize using running statistics.
//
// Hyperparameters:
//   - eps:      small constant for numerical stability (default 1e-5)
//   - momentum: weight for running stat update (default 0.1)
// =============================================================================

class BatchNorm2d : public Module {
public:
    int num_features_;
    float eps_, momentum_;

    GradTensor* gamma_;         // [C] — learnable scale (parameter)
    GradTensor* beta_;          // [C] — learnable shift (parameter)

    float* running_mean_;       // [C] — buffer (non-learnable)
    float* running_var_;        // [C] — buffer (non-learnable)

    // Saved for backward pass:
    float* batch_mean_;         // [C] — mean computed during forward
    float* batch_var_;          // [C] — variance computed during forward
    float* x_hat_;              // [N*C*H*W] — normalized input, saved for backward
    int x_hat_size_;            // Size of x_hat_ allocation

    BatchNorm2d(int num_features, float eps = 1e-5f, float momentum = 0.1f)
        : num_features_(num_features), eps_(eps), momentum_(momentum),
          x_hat_(nullptr), x_hat_size_(0)
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "BatchNorm2d(%d)", num_features);
        name_ = buf;

        // Learnable parameters
        gamma_ = new GradTensor(num_features, 1, 1, 1, true);
        beta_  = new GradTensor(num_features, 1, 1, 1, true);
        register_parameter("gamma", gamma_);
        register_parameter("beta", beta_);

        // Initialize gamma to 1, beta to 0
        std::vector<float> ones(num_features, 1.0f);
        cudaMemcpy(gamma_->data, ones.data(), num_features * sizeof(float),
                   cudaMemcpyHostToDevice);
        // beta is already zero from GradTensor constructor

        // Non-learnable buffers (running statistics)
        cudaMalloc(&running_mean_, num_features * sizeof(float));
        cudaMalloc(&running_var_, num_features * sizeof(float));
        cudaMemset(running_mean_, 0, num_features * sizeof(float));
        // Initialize running_var to 1
        cudaMemcpy(running_var_, ones.data(), num_features * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Temporary storage for batch statistics
        cudaMalloc(&batch_mean_, num_features * sizeof(float));
        cudaMalloc(&batch_var_, num_features * sizeof(float));

        cudaDeviceSynchronize();
    }

    ~BatchNorm2d() {
        delete gamma_;
        delete beta_;
        cudaFree(running_mean_);
        cudaFree(running_var_);
        cudaFree(batch_mean_);
        cudaFree(batch_var_);
        if (x_hat_) cudaFree(x_hat_);
    }

    GradTensor* forward(GradTensor* input) override {
        int N = input->dims[0];
        int C = input->dims[1];
        int H = input->dims[2];
        int W = input->dims[3];
        int total = N * C * H * W;

        // Allocate/reallocate x_hat if needed (saved for backward)
        if (x_hat_size_ < total) {
            if (x_hat_) cudaFree(x_hat_);
            cudaMalloc(&x_hat_, total * sizeof(float));
            x_hat_size_ = total;
        }

        GradTensor* output = new GradTensor(N, C, H, W, false);
        cudaMalloc(&output->grad, output->size * sizeof(float));
        cudaMemset(output->grad, 0, output->size * sizeof(float));

        int threads = 256;

        if (training_) {
            // Training mode: compute batch statistics
            int blocks_c = (C + threads - 1) / threads;

            // Step 1: Compute per-channel mean
            batchnorm_mean_kernel<<<blocks_c, threads>>>(
                input->data, batch_mean_, N, C, H, W
            );

            // Step 2: Compute per-channel variance
            batchnorm_var_kernel<<<blocks_c, threads>>>(
                input->data, batch_mean_, batch_var_, N, C, H, W
            );

            // Step 3: Normalize + affine transform
            int blocks_all = (total + threads - 1) / threads;
            batchnorm_normalize_kernel<<<blocks_all, threads>>>(
                input->data, batch_mean_, batch_var_,
                gamma_->data, beta_->data,
                output->data, x_hat_, eps_, N, C, H, W
            );

            // Step 4: Update running statistics
            batchnorm_update_running_kernel<<<blocks_c, threads>>>(
                running_mean_, batch_mean_, momentum_, C
            );
            batchnorm_update_running_kernel<<<blocks_c, threads>>>(
                running_var_, batch_var_, momentum_, C
            );
        } else {
            // Inference mode: use running statistics (no batch stats)
            int blocks_all = (total + threads - 1) / threads;
            batchnorm_normalize_kernel<<<blocks_all, threads>>>(
                input->data, running_mean_, running_var_,
                gamma_->data, beta_->data,
                output->data, x_hat_, eps_, N, C, H, W
            );
        }

        // ---------- Backward function ----------
        output->parents = {input, gamma_, beta_};

        GradTensor* g = gamma_;
        GradTensor* b = beta_;
        float* bvar = batch_var_;
        float* xh = x_hat_;
        float eps = eps_;

        output->backward_fn = [output, input, g, b, bvar, xh, eps,
                                N, C, H, W]() {
            int threads = 256;
            int blocks_c = (C + threads - 1) / threads;

            // Gradient w.r.t. gamma and beta
            batchnorm_backward_gamma_beta_kernel<<<blocks_c, threads>>>(
                output->grad, xh, g->grad, b->grad, N, C, H, W
            );

            // Gradient w.r.t. input
            if (input->grad) {
                batchnorm_backward_input_kernel<<<blocks_c, threads>>>(
                    output->grad, xh, bvar, g->data, input->grad,
                    eps, N, C, H, W
                );
            }
        };

        return output;
    }
};


// =============================================================================
// Layer: ReLU
// =============================================================================
// Rectified Linear Unit — a stateless activation function.
// No learnable parameters.
//
// Forward:  out = max(0, x)
// Backward: d_input = d_output * (x > 0 ? 1 : 0)
// =============================================================================

class ReLU : public Module {
public:
    ReLU() {
        name_ = "ReLU()";
    }

    GradTensor* forward(GradTensor* input) override {
        int total = input->size;

        GradTensor* output = new GradTensor(
            input->dims[0], input->dims[1], input->dims[2], input->dims[3], false
        );
        cudaMalloc(&output->grad, output->size * sizeof(float));
        cudaMemset(output->grad, 0, output->size * sizeof(float));

        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        relu_forward_kernel<<<blocks, threads>>>(input->data, output->data, total);

        // Backward: need original input to check sign
        output->parents = {input};

        output->backward_fn = [output, input, total]() {
            if (input->grad) {
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                relu_backward_kernel<<<blocks, threads>>>(
                    output->grad, input->data, input->grad, total
                );
            }
        };

        return output;
    }
};


// =============================================================================
// Layer: Linear (Fully Connected)
// =============================================================================
// Applies a linear transformation: y = x * W^T + b
//
// Parameters:
//   - weight: [out_features, in_features]
//   - bias:   [out_features] (optional)
//
// Input:  [N, in_features]
// Output: [N, out_features]
//
// Initialization: Kaiming — std = sqrt(2 / in_features)
// =============================================================================

class Linear : public Module {
public:
    int in_features_, out_features_;
    GradTensor* weight_;
    GradTensor* bias_;
    bool use_bias_;

    Linear(int in_features, int out_features, bool use_bias = true)
        : in_features_(in_features), out_features_(out_features),
          use_bias_(use_bias)
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "Linear(%d, %d)", in_features, out_features);
        name_ = buf;

        // Weight: [out_features, in_features]
        weight_ = new GradTensor(out_features, in_features, 1, 1, true);
        register_parameter("weight", weight_);

        // Kaiming initialization
        float std_dev = sqrtf(2.0f / in_features);
        int blocks = (weight_->size + 255) / 256;
        kaiming_init_kernel<<<blocks, 256>>>(weight_->data, weight_->size, std_dev, 123);

        if (use_bias) {
            bias_ = new GradTensor(out_features, 1, 1, 1, true);
            register_parameter("bias", bias_);
        } else {
            bias_ = nullptr;
        }

        cudaDeviceSynchronize();
    }

    ~Linear() {
        delete weight_;
        if (bias_) delete bias_;
    }

    GradTensor* forward(GradTensor* input) override {
        int N = input->dims[0];
        int in_f = in_features_;
        int out_f = out_features_;

        GradTensor* output = new GradTensor(N, out_f, 1, 1, false);
        cudaMalloc(&output->grad, output->size * sizeof(float));
        cudaMemset(output->grad, 0, output->size * sizeof(float));

        int total = N * out_f;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        linear_forward_kernel<<<blocks, threads>>>(
            input->data, weight_->data, use_bias_ ? bias_->data : nullptr,
            output->data, N, in_f, out_f
        );

        // Backward
        output->parents = {input, weight_};
        if (bias_) output->parents.push_back(bias_);

        GradTensor* w = weight_;
        GradTensor* b = bias_;

        output->backward_fn = [output, input, w, b, N, in_f, out_f]() {
            int threads = 256;

            // d_input
            if (input->grad) {
                int total_in = N * in_f;
                int blocks_in = (total_in + threads - 1) / threads;
                linear_backward_input_kernel<<<blocks_in, threads>>>(
                    output->grad, w->data, input->grad, N, in_f, out_f
                );
            }

            // d_weight
            {
                int total_w = out_f * in_f;
                int blocks_w = (total_w + threads - 1) / threads;
                linear_backward_weight_kernel<<<blocks_w, threads>>>(
                    output->grad, input->data, w->grad, N, in_f, out_f
                );
            }

            // d_bias
            if (b) {
                int blocks_b = (out_f + threads - 1) / threads;
                linear_backward_bias_kernel<<<blocks_b, threads>>>(
                    output->grad, b->grad, N, out_f
                );
            }
        };

        return output;
    }
};


// =============================================================================
// Layer: GlobalAvgPool2d
// =============================================================================
// Global average pooling: takes [N, C, H, W] and produces [N, C].
// Each output is the mean of all spatial positions in a channel.
// No learnable parameters.
//
// This is the modern alternative to flattening + large FC layers.
// Used in ResNet, GoogLeNet, etc.
// =============================================================================

class GlobalAvgPool2d : public Module {
public:
    GlobalAvgPool2d() {
        name_ = "GlobalAvgPool2d()";
    }

    GradTensor* forward(GradTensor* input) override {
        int N = input->dims[0];
        int C = input->dims[1];
        int H = input->dims[2];
        int W = input->dims[3];

        // Output shape: [N, C] (stored as [N, C, 1, 1] for consistency)
        GradTensor* output = new GradTensor(N, C, 1, 1, false);
        cudaMalloc(&output->grad, output->size * sizeof(float));
        cudaMemset(output->grad, 0, output->size * sizeof(float));

        int total_out = N * C;
        int threads = 256;
        int blocks = (total_out + threads - 1) / threads;

        global_avg_pool_forward_kernel<<<blocks, threads>>>(
            input->data, output->data, N, C, H, W
        );

        // Backward
        output->parents = {input};

        output->backward_fn = [output, input, N, C, H, W]() {
            if (input->grad) {
                int total_in = N * C * H * W;
                int threads = 256;
                int blocks = (total_in + threads - 1) / threads;
                global_avg_pool_backward_kernel<<<blocks, threads>>>(
                    output->grad, input->grad, N, C, H, W
                );
            }
        };

        return output;
    }
};


// =============================================================================
// Layer: Sequential
// =============================================================================
// A container that chains modules together, passing each module's output
// as input to the next. Equivalent to PyTorch's nn.Sequential.
//
// Usage:
//   Sequential* model = new Sequential();
//   model->add("conv1", new Conv2d(4, 16, 3, 1));
//   model->add("bn1",   new BatchNorm2d(16));
//   model->add("relu",  new ReLU());
//   model->add("fc",    new Linear(16, 10));
//
//   GradTensor* output = model->forward(input);
// =============================================================================

class Sequential : public Module {
public:
    // Ordered list of modules — forward() runs them in sequence
    std::vector<Module*> layers_;

    Sequential() {
        name_ = "Sequential";
    }

    // Add a named layer to the sequence
    void add(const std::string& name, Module* layer) {
        layers_.push_back(layer);
        register_module(name, layer);
    }

    // Forward: chain outputs through each layer
    GradTensor* forward(GradTensor* input) override {
        GradTensor* x = input;
        for (auto* layer : layers_) {
            x = layer->forward(x);
        }
        return x;
    }
};


#endif // CUDALEARN_LAYERS_CUH
