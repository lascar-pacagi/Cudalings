// ===========================================================================
// Chapter 16: autograd_ops.cuh -- Autograd-Wrapped GPU Operations
// ===========================================================================
//
// This header provides the autograd-enabled versions of all operations
// needed to build and train a ResNet. Each function:
//
//   1. Computes the forward pass using a CUDA kernel
//   2. Creates a new GradTensor for the output
//   3. Sets up a backward_fn closure that:
//      - Reads the output's .grad (upstream gradient)
//      - Computes gradients for each input using CUDA backward kernels
//      - ACCUMULATES (+=) gradients into input .grad fields
//
// OPERATIONS IMPLEMENTED:
//
//   autograd::add            -- element-wise addition (residual connections)
//   autograd::relu           -- element-wise rectified linear unit
//   autograd::linear         -- fully connected layer: y = x @ W^T + b
//   autograd::conv2d         -- 2D convolution (NCHW, stride=1)
//   autograd::batchnorm      -- batch normalization (per-channel)
//   autograd::global_avg_pool-- global average pooling (spatial dims -> 1x1)
//   autograd::cross_entropy  -- cross-entropy loss with softmax
//   autograd::mul_scalar     -- multiply tensor by scalar (for testing)
//   autograd::add_scalar     -- add scalar to tensor (for testing)
//   autograd::square         -- element-wise square (for testing)
//   autograd::sum            -- sum all elements to scalar (for testing)
//
// ALL CUDA KERNELS ARE SELF-CONTAINED -- no dependencies on other chapters.
//
// CHAIN RULE NOTATION:
//   dL/dy  = "grad_output" = upstream gradient (what this node's .grad holds)
//   dL/dx  = "grad_input"  = downstream gradient (what we propagate to children)
//   dL/dW  = "grad_weight" = parameter gradient (accumulated into weight.grad)
//
// ===========================================================================

#pragma once

#include "grad_tensor.cuh"

namespace autograd {

// ===========================================================================
// HELPER: grid-stride loop block/grid calculation
// ===========================================================================
inline void get_launch_config(int N, int& blocks, int& threads) {
    threads = 256;
    blocks = (N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
}

// ===========================================================================
//  1. ELEMENT-WISE ADDITION: z = a + b
// ===========================================================================
//
//  Forward:  z[i] = a[i] + b[i]
//
//  Backward (chain rule):
//    dL/da[i] = dL/dz[i] * dz/da[i] = dL/dz[i] * 1 = dL/dz[i]
//    dL/db[i] = dL/dz[i] * dz/db[i] = dL/dz[i] * 1 = dL/dz[i]
//
//  Both inputs receive the upstream gradient unchanged.
//  This is why residual connections provide a "gradient highway" --
//  the gradient flows through the addition without any scaling.
//
// ===========================================================================

__global__ void add_forward_kernel(const float* a, const float* b,
                                    float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
}

// add_backward does not need a kernel -- it just copies the upstream gradient
// to both inputs. We use accumulate_kernel from grad_tensor.cuh.

inline GradTensorPtr add(GradTensorPtr a, GradTensorPtr b) {
    // Verify shapes match
    assert(a->size == b->size && "add: tensors must have same size");

    // Create output tensor (inherits grad tracking if either input needs it)
    bool needs_grad = a->requires_grad || b->requires_grad;
    auto out = make_grad_tensor(a->shape, needs_grad, "add_out");
    out->op_name = "add";
    out->children = {a, b};

    // Forward pass: z = a + b
    int blocks, threads;
    get_launch_config(a->size, blocks, threads);
    add_forward_kernel<<<blocks, threads>>>(a->data, b->data, out->data, a->size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Set up backward function
    // Captures a, b, out by shared_ptr (prevents dangling pointers)
    if (needs_grad) {
        out->backward_fn = [a, b, out]() {
            int blk, thr;
            get_launch_config(out->size, blk, thr);

            // dL/da += dL/dz (gradient flows through unchanged)
            if (a->requires_grad) {
                a->ensure_grad();
                accumulate_kernel<<<blk, thr>>>(a->grad, out->grad, out->size);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // dL/db += dL/dz (gradient flows through unchanged)
            if (b->requires_grad) {
                b->ensure_grad();
                accumulate_kernel<<<blk, thr>>>(b->grad, out->grad, out->size);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        };
    }

    return out;
}

// ===========================================================================
//  2. RELU: y = max(0, x)
// ===========================================================================
//
//  Forward:  y[i] = max(0, x[i])
//
//  Backward (chain rule):
//    dL/dx[i] = dL/dy[i] * dy/dx[i]
//             = dL/dy[i] * (x[i] > 0 ? 1 : 0)
//
//  ReLU backward is a binary gate: gradients pass through where input > 0,
//  and are killed where input <= 0.
//
// ===========================================================================

__global__ void relu_forward_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

__global__ void relu_backward_kernel(const float* grad_output,
                                      const float* input,
                                      float* grad_input,
                                      int N) {
    // grad_input[i] += grad_output[i] * (input[i] > 0)
    // Note: we ACCUMULATE (+=) to handle gradient accumulation correctly.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_input[idx] += grad_output[idx] * (input[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

inline GradTensorPtr relu(GradTensorPtr input) {
    auto out = make_grad_tensor(input->shape, input->requires_grad, "relu_out");
    out->op_name = "relu";
    out->children = {input};

    // Forward: y = max(0, x)
    int blocks, threads;
    get_launch_config(input->size, blocks, threads);
    relu_forward_kernel<<<blocks, threads>>>(input->data, out->data, input->size);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (input->requires_grad) {
        // Backward closure captures input (for the mask) and out (for grad)
        out->backward_fn = [input, out]() {
            input->ensure_grad();
            int blk, thr;
            get_launch_config(input->size, blk, thr);
            relu_backward_kernel<<<blk, thr>>>(
                out->grad, input->data, input->grad, input->size);
            CUDA_CHECK(cudaDeviceSynchronize());
        };
    }

    return out;
}

// ===========================================================================
//  3. LINEAR: y = x @ W^T + b
// ===========================================================================
//
//  Shapes:
//    x:      (B, in_features)
//    W:      (out_features, in_features)
//    b:      (out_features,)
//    output: (B, out_features)
//
//  Forward:
//    y[b][j] = SUM_k x[b][k] * W[j][k] + b[j]
//
//  Backward:
//    dL/dx[b][k] = SUM_j dL/dy[b][j] * W[j][k]       (matmul with W)
//    dL/dW[j][k] = SUM_b dL/dy[b][j] * x[b][k]       (outer product sum)
//    dL/db[j]    = SUM_b dL/dy[b][j]                  (sum over batch)
//
// ===========================================================================

// Forward kernel: each thread computes one element of the output matrix
__global__ void linear_forward_kernel(
    const float* __restrict__ x,      // (B, in_f)
    const float* __restrict__ W,      // (out_f, in_f)
    const float* __restrict__ bias,   // (out_f,)
    float* __restrict__ output,       // (B, out_f)
    int B, int in_f, int out_f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * out_f;
    if (idx < total) {
        int b = idx / out_f;
        int j = idx % out_f;

        float sum = bias[j];
        for (int k = 0; k < in_f; k++) {
            sum += x[b * in_f + k] * W[j * in_f + k];
        }
        output[idx] = sum;
    }
}

// Backward kernel: gradient w.r.t. input x
//   dL/dx[b][k] = SUM_j dL/dy[b][j] * W[j][k]
__global__ void linear_backward_input_kernel(
    const float* __restrict__ grad_output,  // (B, out_f)
    const float* __restrict__ W,            // (out_f, in_f)
    float* __restrict__ grad_input,         // (B, in_f)
    int B, int in_f, int out_f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * in_f;
    if (idx < total) {
        int b = idx / in_f;
        int k = idx % in_f;

        float sum = 0.0f;
        for (int j = 0; j < out_f; j++) {
            sum += grad_output[b * out_f + j] * W[j * in_f + k];
        }
        grad_input[idx] += sum;  // Accumulate!
    }
}

// Backward kernel: gradient w.r.t. weight W
//   dL/dW[j][k] = SUM_b dL/dy[b][j] * x[b][k]
__global__ void linear_backward_weight_kernel(
    const float* __restrict__ grad_output,  // (B, out_f)
    const float* __restrict__ x,            // (B, in_f)
    float* __restrict__ grad_weight,        // (out_f, in_f)
    int B, int in_f, int out_f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_f * in_f;
    if (idx < total) {
        int j = idx / in_f;
        int k = idx % in_f;

        float sum = 0.0f;
        for (int b = 0; b < B; b++) {
            sum += grad_output[b * out_f + j] * x[b * in_f + k];
        }
        grad_weight[idx] += sum;  // Accumulate!
    }
}

// Backward kernel: gradient w.r.t. bias
//   dL/db[j] = SUM_b dL/dy[b][j]
__global__ void linear_backward_bias_kernel(
    const float* __restrict__ grad_output,  // (B, out_f)
    float* __restrict__ grad_bias,          // (out_f,)
    int B, int out_f
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < out_f) {
        float sum = 0.0f;
        for (int b = 0; b < B; b++) {
            sum += grad_output[b * out_f + j];
        }
        grad_bias[j] += sum;  // Accumulate!
    }
}

inline GradTensorPtr linear(GradTensorPtr input, GradTensorPtr weight,
                             GradTensorPtr bias) {
    // input: (B, in_f), weight: (out_f, in_f), bias: (out_f,)
    int B = input->shape[0];
    int in_f = input->shape[1];
    int out_f = weight->shape[0];

    auto out = make_grad_tensor({B, out_f}, true, "linear_out");
    out->op_name = "linear";
    out->children = {input, weight, bias};

    // Forward pass
    int blocks, threads;
    get_launch_config(B * out_f, blocks, threads);
    linear_forward_kernel<<<blocks, threads>>>(
        input->data, weight->data, bias->data, out->data,
        B, in_f, out_f);
    CUDA_CHECK(cudaDeviceSynchronize());

    out->backward_fn = [input, weight, bias, out, B, in_f, out_f]() {
        int blk, thr;

        // dL/dx
        if (input->requires_grad) {
            input->ensure_grad();
            get_launch_config(B * in_f, blk, thr);
            linear_backward_input_kernel<<<blk, thr>>>(
                out->grad, weight->data, input->grad, B, in_f, out_f);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // dL/dW
        if (weight->requires_grad) {
            weight->ensure_grad();
            get_launch_config(out_f * in_f, blk, thr);
            linear_backward_weight_kernel<<<blk, thr>>>(
                out->grad, input->data, weight->grad, B, in_f, out_f);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // dL/db
        if (bias->requires_grad) {
            bias->ensure_grad();
            get_launch_config(out_f, blk, thr);
            linear_backward_bias_kernel<<<blk, thr>>>(
                out->grad, bias->grad, B, out_f);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    };

    return out;
}

// ===========================================================================
//  4. CONV2D: 2D Convolution (NCHW, stride=1)
// ===========================================================================
//
//  Forward:
//    output[b][oc][oh][ow] =
//      SUM_{ic,kh,kw} input[b][ic][oh+kh-pad][ow+kw-pad] * weight[oc][ic][kh][kw]
//
//  Shapes:
//    input:  (B, IC, IH, IW)
//    weight: (OC, IC, KH, KW)
//    output: (B, OC, OH, OW)   where OH = IH + 2*pad - KH + 1
//
//  Backward w.r.t. input (stride=1):
//    dL/d(input[b][ic][ih][iw]) =
//      SUM_{oc,kh,kw} weight[oc][ic][kh][kw]
//                      * grad_output[b][oc][ih-kh+pad][iw-kw+pad]
//                                    (when indices are in bounds)
//
//    This is equivalent to convolving grad_output with the 180-rotated weight.
//
//  Backward w.r.t. weight:
//    dL/d(weight[oc][ic][kh][kw]) =
//      SUM_{b,oh,ow} grad_output[b][oc][oh][ow]
//                     * input[b][ic][oh+kh-pad][ow+kw-pad]
//
// ===========================================================================

__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,   // (B, IC, IH, IW)
    const float* __restrict__ weight,  // (OC, IC, KH, KW)
    float* __restrict__ output,        // (B, OC, OH, OW)
    int B, int IC, int IH, int IW,
    int OC, int KH, int KW,
    int OH, int OW, int pad
) {
    // Each thread computes one output element: output[b][oc][oh][ow]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * OH * OW;
    if (idx >= total) return;

    // Decompose linear index into (b, oc, oh, ow)
    int ow = idx % OW;
    int oh = (idx / OW) % OH;
    int oc = (idx / (OW * OH)) % OC;
    int b  = idx / (OW * OH * OC);

    float sum = 0.0f;
    for (int ic = 0; ic < IC; ic++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                int ih = oh + kh - pad;
                int iw = ow + kw - pad;
                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                    float in_val = input[b * IC * IH * IW + ic * IH * IW + ih * IW + iw];
                    float w_val  = weight[oc * IC * KH * KW + ic * KH * KW + kh * KW + kw];
                    sum += in_val * w_val;
                }
            }
        }
    }
    output[idx] = sum;
}

// Conv2D backward w.r.t. input
__global__ void conv2d_backward_input_kernel(
    const float* __restrict__ grad_output,  // (B, OC, OH, OW)
    const float* __restrict__ weight,       // (OC, IC, KH, KW)
    float* __restrict__ grad_input,         // (B, IC, IH, IW)
    int B, int IC, int IH, int IW,
    int OC, int KH, int KW,
    int OH, int OW, int pad
) {
    // Each thread computes grad for one input element: grad_input[b][ic][ih][iw]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * IC * IH * IW;
    if (idx >= total) return;

    int iw = idx % IW;
    int ih = (idx / IW) % IH;
    int ic = (idx / (IW * IH)) % IC;
    int b  = idx / (IW * IH * IC);

    float sum = 0.0f;
    for (int oc = 0; oc < OC; oc++) {
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                // oh = ih - kh + pad (from the forward: ih = oh + kh - pad)
                int oh = ih - kh + pad;
                int ow_val = iw - kw + pad;
                if (oh >= 0 && oh < OH && ow_val >= 0 && ow_val < OW) {
                    float go = grad_output[b * OC * OH * OW + oc * OH * OW + oh * OW + ow_val];
                    float w  = weight[oc * IC * KH * KW + ic * KH * KW + kh * KW + kw];
                    sum += go * w;
                }
            }
        }
    }
    grad_input[idx] += sum;  // Accumulate!
}

// Conv2D backward w.r.t. weight
__global__ void conv2d_backward_weight_kernel(
    const float* __restrict__ grad_output,  // (B, OC, OH, OW)
    const float* __restrict__ input,        // (B, IC, IH, IW)
    float* __restrict__ grad_weight,        // (OC, IC, KH, KW)
    int B, int IC, int IH, int IW,
    int OC, int KH, int KW,
    int OH, int OW, int pad
) {
    // Each thread computes grad for one weight element: grad_weight[oc][ic][kh][kw]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OC * IC * KH * KW;
    if (idx >= total) return;

    int kw = idx % KW;
    int kh = (idx / KW) % KH;
    int ic = (idx / (KW * KH)) % IC;
    int oc = idx / (KW * KH * IC);

    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int ih = oh + kh - pad;
                int iw = ow + kw - pad;
                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                    float go = grad_output[b * OC * OH * OW + oc * OH * OW + oh * OW + ow];
                    float iv = input[b * IC * IH * IW + ic * IH * IW + ih * IW + iw];
                    sum += go * iv;
                }
            }
        }
    }
    grad_weight[idx] += sum;  // Accumulate!
}

inline GradTensorPtr conv2d(GradTensorPtr input, GradTensorPtr weight, int padding) {
    // Extract dimensions
    int B  = input->shape[0];
    int IC = input->shape[1];
    int IH = input->shape[2];
    int IW = input->shape[3];
    int OC = weight->shape[0];
    int KH = weight->shape[2];
    int KW = weight->shape[3];
    int OH = IH + 2 * padding - KH + 1;
    int OW = IW + 2 * padding - KW + 1;

    auto out = make_grad_tensor({B, OC, OH, OW}, true, "conv2d_out");
    out->op_name = "conv2d";
    out->children = {input, weight};

    // Forward pass
    int total = B * OC * OH * OW;
    int blocks, threads;
    get_launch_config(total, blocks, threads);
    conv2d_forward_kernel<<<blocks, threads>>>(
        input->data, weight->data, out->data,
        B, IC, IH, IW, OC, KH, KW, OH, OW, padding);
    CUDA_CHECK(cudaDeviceSynchronize());

    out->backward_fn = [input, weight, out, B, IC, IH, IW, OC, KH, KW, OH, OW, padding]() {
        int blk, thr;

        // dL/d(input)
        if (input->requires_grad) {
            input->ensure_grad();
            get_launch_config(B * IC * IH * IW, blk, thr);
            conv2d_backward_input_kernel<<<blk, thr>>>(
                out->grad, weight->data, input->grad,
                B, IC, IH, IW, OC, KH, KW, OH, OW, padding);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // dL/d(weight)
        if (weight->requires_grad) {
            weight->ensure_grad();
            get_launch_config(OC * IC * KH * KW, blk, thr);
            conv2d_backward_weight_kernel<<<blk, thr>>>(
                out->grad, input->data, weight->grad,
                B, IC, IH, IW, OC, KH, KW, OH, OW, padding);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    };

    return out;
}

// ===========================================================================
//  5. BATCHNORM: Batch Normalization (per-channel, NCHW)
// ===========================================================================
//
//  Forward (training mode):
//    For each channel c:
//      mean[c]     = (1/N_spatial) * SUM_{b,h,w} x[b][c][h][w]
//      var[c]      = (1/N_spatial) * SUM_{b,h,w} (x[b][c][h][w] - mean[c])^2
//      x_hat[b][c][h][w] = (x[b][c][h][w] - mean[c]) / sqrt(var[c] + eps)
//      y[b][c][h][w]     = gamma[c] * x_hat[b][c][h][w] + beta[c]
//
//    where N_spatial = B * H * W (number of elements per channel)
//
//  Backward:
//    dL/dgamma[c] = SUM_{b,h,w} dL/dy[b][c][h][w] * x_hat[b][c][h][w]
//    dL/dbeta[c]  = SUM_{b,h,w} dL/dy[b][c][h][w]
//    dL/dx = (gamma / sqrt(var+eps)) * (dL/dy - mean(dL/dy) - x_hat * mean(dL/dy * x_hat))
//
//    This is the full batch norm gradient including the contribution
//    through the mean and variance (the "hard" part of batchnorm backward).
//
// ===========================================================================

// BN forward: compute mean and variance per channel
__global__ void bn_compute_stats_kernel(
    const float* __restrict__ input,   // (B, C, H, W)
    float* __restrict__ mean,          // (C,)
    float* __restrict__ var,           // (C,)
    int B, int C, int H, int W
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    int N_spatial = B * H * W;
    float sum = 0.0f;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += input[b * C * H * W + c * H * W + h * W + w];
            }
        }
    }
    mean[c] = sum / N_spatial;

    float var_sum = 0.0f;
    float m = mean[c];
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float diff = input[b * C * H * W + c * H * W + h * W + w] - m;
                var_sum += diff * diff;
            }
        }
    }
    var[c] = var_sum / N_spatial;
}

// BN forward: normalize and scale
__global__ void bn_normalize_kernel(
    const float* __restrict__ input,   // (B, C, H, W)
    const float* __restrict__ mean,    // (C,)
    const float* __restrict__ var,     // (C,)
    const float* __restrict__ gamma,   // (C,)
    const float* __restrict__ beta,    // (C,)
    float* __restrict__ output,        // (B, C, H, W)
    float* __restrict__ x_hat,         // (B, C, H, W) -- saved for backward
    int B, int C, int H, int W, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;

    int c = (idx / (W * H)) % C;

    float inv_std = rsqrtf(var[c] + eps);
    float xh = (input[idx] - mean[c]) * inv_std;
    x_hat[idx] = xh;
    output[idx] = gamma[c] * xh + beta[c];
}

// BN backward: compute gradients for gamma, beta, and input
__global__ void bn_backward_kernel(
    const float* __restrict__ grad_output,  // (B, C, H, W)
    const float* __restrict__ x_hat,        // (B, C, H, W) saved from forward
    const float* __restrict__ gamma,        // (C,)
    const float* __restrict__ var,          // (C,)
    float* __restrict__ grad_input,         // (B, C, H, W)
    float* __restrict__ grad_gamma,         // (C,)
    float* __restrict__ grad_beta,          // (C,)
    int B, int C, int H, int W, float eps
) {
    // One thread per channel -- computes stats then applies to all elements.
    // Not optimal for large inputs, but correct and clear.
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    int N_spatial = B * H * W;
    float inv_std = rsqrtf(var[c] + eps);

    // Compute dL/dgamma and dL/dbeta (reductions over spatial+batch dims)
    float dgamma = 0.0f;
    float dbeta = 0.0f;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = b * C * H * W + c * H * W + h * W + w;
                dgamma += grad_output[idx] * x_hat[idx];
                dbeta  += grad_output[idx];
            }
        }
    }
    grad_gamma[c] += dgamma;  // Accumulate!
    grad_beta[c]  += dbeta;

    // Compute mean of (grad_output) and mean of (grad_output * x_hat) per channel
    // These are needed for the full batchnorm gradient through mean/var
    float mean_dy = dbeta / N_spatial;
    float mean_dy_xhat = dgamma / N_spatial;

    // dL/dx = (gamma / std) * (dL/dy - mean(dL/dy) - x_hat * mean(dL/dy * x_hat))
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = b * C * H * W + c * H * W + h * W + w;
                float dy = grad_output[idx];
                grad_input[idx] += gamma[c] * inv_std *
                    (dy - mean_dy - x_hat[idx] * mean_dy_xhat);
            }
        }
    }
}

inline GradTensorPtr batchnorm(GradTensorPtr input, GradTensorPtr gamma,
                                GradTensorPtr beta,
                                GradTensorPtr running_mean,
                                GradTensorPtr running_var,
                                bool training) {
    int B = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    float eps = 1e-5f;

    auto out = make_grad_tensor({B, C, H, W}, true, "bn_out");
    out->op_name = "batchnorm";
    out->children = {input, gamma, beta};

    // Saved tensors for backward pass
    auto x_hat = make_grad_tensor({B, C, H, W}, false, "bn_xhat");
    auto batch_mean = make_grad_tensor({C}, false, "bn_mean");
    auto batch_var  = make_grad_tensor({C}, false, "bn_var");

    if (training) {
        // Compute batch statistics
        int blk, thr;
        get_launch_config(C, blk, thr);
        bn_compute_stats_kernel<<<blk, thr>>>(
            input->data, batch_mean->data, batch_var->data, B, C, H, W);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update running statistics (exponential moving average, momentum=0.1)
        // running_mean = 0.9 * running_mean + 0.1 * batch_mean
        // (done on CPU for simplicity -- small array)
        std::vector<float> h_rm(C), h_rv(C), h_bm(C), h_bv(C);
        running_mean->get_data_to_host(h_rm.data());
        running_var->get_data_to_host(h_rv.data());
        batch_mean->get_data_to_host(h_bm.data());
        batch_var->get_data_to_host(h_bv.data());
        for (int c = 0; c < C; c++) {
            h_rm[c] = 0.9f * h_rm[c] + 0.1f * h_bm[c];
            h_rv[c] = 0.9f * h_rv[c] + 0.1f * h_bv[c];
        }
        running_mean->set_data_from_host(h_rm.data());
        running_var->set_data_from_host(h_rv.data());
    } else {
        // Use running statistics for inference
        CUDA_CHECK(cudaMemcpy(batch_mean->data, running_mean->data,
                              C * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(batch_var->data, running_var->data,
                              C * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Normalize and scale
    int total = B * C * H * W;
    int blocks, threads;
    get_launch_config(total, blocks, threads);
    bn_normalize_kernel<<<blocks, threads>>>(
        input->data, batch_mean->data, batch_var->data,
        gamma->data, beta->data, out->data, x_hat->data,
        B, C, H, W, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    out->backward_fn = [input, gamma, beta, out, x_hat, batch_var,
                         B, C, H, W, eps]() {
        int blk, thr;
        get_launch_config(C, blk, thr);

        // Allocate/ensure grad buffers
        if (input->requires_grad) input->ensure_grad();
        if (gamma->requires_grad) gamma->ensure_grad();
        if (beta->requires_grad)  beta->ensure_grad();

        // The backward kernel handles all three gradients (input, gamma, beta)
        // in one pass per channel.
        if (input->requires_grad || gamma->requires_grad || beta->requires_grad) {
            // We need temporary grad buffers if input doesn't require grad
            // but params do. For simplicity, always allocate grad_input temp.
            float* gi = input->grad;
            float* gg = gamma->grad;
            float* gb = beta->grad;

            // If input doesn't require grad, use a dummy buffer
            float* dummy_gi = nullptr;
            if (!input->requires_grad) {
                CUDA_CHECK(cudaMalloc(&dummy_gi, input->size * sizeof(float)));
                int b2, t2;
                get_launch_config(input->size, b2, t2);
                fill_kernel<<<b2, t2>>>(dummy_gi, 0.0f, input->size);
                gi = dummy_gi;
            }
            float* dummy_gg = nullptr;
            if (!gamma->requires_grad) {
                CUDA_CHECK(cudaMalloc(&dummy_gg, C * sizeof(float)));
                fill_kernel<<<1, C>>>(dummy_gg, 0.0f, C);
                gg = dummy_gg;
            }
            float* dummy_gb = nullptr;
            if (!beta->requires_grad) {
                CUDA_CHECK(cudaMalloc(&dummy_gb, C * sizeof(float)));
                fill_kernel<<<1, C>>>(dummy_gb, 0.0f, C);
                gb = dummy_gb;
            }

            bn_backward_kernel<<<blk, thr>>>(
                out->grad, x_hat->data, gamma->data, batch_var->data,
                gi, gg, gb, B, C, H, W, eps);
            CUDA_CHECK(cudaDeviceSynchronize());

            if (dummy_gi) cudaFree(dummy_gi);
            if (dummy_gg) cudaFree(dummy_gg);
            if (dummy_gb) cudaFree(dummy_gb);
        }
    };

    return out;
}

// ===========================================================================
//  6. GLOBAL AVERAGE POOLING: (B, C, H, W) -> (B, C)
// ===========================================================================
//
//  Forward:
//    output[b][c] = (1 / (H*W)) * SUM_{h,w} input[b][c][h][w]
//
//  Backward:
//    dL/d(input[b][c][h][w]) = dL/d(output[b][c]) / (H*W)
//
//    Each spatial element contributes equally to the average, so the
//    gradient is uniformly distributed back to all spatial positions.
//
// ===========================================================================

__global__ void gap_forward_kernel(
    const float* __restrict__ input,   // (B, C, H, W)
    float* __restrict__ output,        // (B, C)
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    if (idx >= total) return;

    int c = idx % C;
    int b = idx / C;

    float sum = 0.0f;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            sum += input[b * C * H * W + c * H * W + h * W + w];
        }
    }
    output[idx] = sum / (H * W);
}

__global__ void gap_backward_kernel(
    const float* __restrict__ grad_output,  // (B, C)
    float* __restrict__ grad_input,         // (B, C, H, W)
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;

    int c = (idx / (W * H)) % C;
    int b = idx / (W * H * C);

    // Each spatial position gets 1/(H*W) of the upstream gradient
    grad_input[idx] += grad_output[b * C + c] / (H * W);
}

inline GradTensorPtr global_avg_pool(GradTensorPtr input) {
    int B = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];

    auto out = make_grad_tensor({B, C}, true, "gap_out");
    out->op_name = "global_avg_pool";
    out->children = {input};

    // Forward
    int blocks, threads;
    get_launch_config(B * C, blocks, threads);
    gap_forward_kernel<<<blocks, threads>>>(input->data, out->data, B, C, H, W);
    CUDA_CHECK(cudaDeviceSynchronize());

    out->backward_fn = [input, out, B, C, H, W]() {
        if (input->requires_grad) {
            input->ensure_grad();
            int blk, thr;
            get_launch_config(B * C * H * W, blk, thr);
            gap_backward_kernel<<<blk, thr>>>(
                out->grad, input->grad, B, C, H, W);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    };

    return out;
}

// ===========================================================================
//  7. CROSS-ENTROPY LOSS with SOFTMAX
// ===========================================================================
//
//  Combined softmax + cross-entropy for numerical stability.
//
//  Forward:
//    For each sample b:
//      max_logit = max_j logits[b][j]
//      log_sum_exp = log(SUM_j exp(logits[b][j] - max_logit)) + max_logit
//      loss = -logits[b][label[b]] + log_sum_exp
//    total_loss = (1/B) * SUM_b loss[b]
//
//  Backward:
//    dL/d(logits[b][j]) = (1/B) * (softmax(logits[b])[j] - 1{j == label[b]})
//
//    The softmax-minus-one-hot formula is one of the most elegant results
//    in deep learning: the gradient of cross-entropy w.r.t. logits is simply
//    the softmax probabilities with -1 subtracted at the correct class.
//
// ===========================================================================

// Forward: compute per-sample loss and total loss
// We do this on CPU for simplicity (B is typically small)
__global__ void cross_entropy_backward_kernel(
    const float* __restrict__ logits,    // (B, num_classes)
    const int* __restrict__ labels,      // (B,)
    float* __restrict__ grad_logits,     // (B, num_classes)
    const float* __restrict__ grad_loss, // scalar (upstream gradient)
    int B, int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * num_classes;
    if (idx >= total) return;

    int j = idx % num_classes;
    int b = idx / num_classes;

    // Compute softmax for this sample (numerically stable)
    // First find max for this sample
    float max_val = -1e30f;
    for (int k = 0; k < num_classes; k++) {
        float v = logits[b * num_classes + k];
        if (v > max_val) max_val = v;
    }

    // Compute exp(logit - max) and sum
    float sum_exp = 0.0f;
    for (int k = 0; k < num_classes; k++) {
        sum_exp += expf(logits[b * num_classes + k] - max_val);
    }

    // softmax probability for class j
    float prob = expf(logits[b * num_classes + j] - max_val) / sum_exp;

    // Gradient: (1/B) * (prob - 1{j == label})
    float indicator = (j == labels[b]) ? 1.0f : 0.0f;
    grad_logits[idx] += grad_loss[0] * (prob - indicator) / B;
}

inline GradTensorPtr cross_entropy(GradTensorPtr logits, GradTensorPtr labels_tensor) {
    int B = logits->shape[0];
    int num_classes = logits->shape[1];

    // Output is a scalar (shape {1})
    auto loss = make_grad_tensor({1}, true, "ce_loss");
    loss->op_name = "cross_entropy";
    loss->children = {logits};

    // --- Forward pass (compute loss on CPU for clarity) ---
    std::vector<float> h_logits(logits->size);
    std::vector<int> h_labels(B);
    logits->get_data_to_host(h_logits.data());

    // Labels are stored as float on GPU but represent integers
    std::vector<float> h_labels_f(B);
    labels_tensor->get_data_to_host(h_labels_f.data());
    for (int i = 0; i < B; i++) h_labels[i] = (int)h_labels_f[i];

    float total_loss = 0.0f;
    for (int b = 0; b < B; b++) {
        // Numerically stable log-sum-exp
        float max_val = -1e30f;
        for (int j = 0; j < num_classes; j++) {
            float v = h_logits[b * num_classes + j];
            if (v > max_val) max_val = v;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) {
            sum_exp += expf(h_logits[b * num_classes + j] - max_val);
        }
        float log_sum_exp = logf(sum_exp) + max_val;
        total_loss += -h_logits[b * num_classes + h_labels[b]] + log_sum_exp;
    }
    total_loss /= B;

    float h_loss = total_loss;
    loss->set_data_from_host(&h_loss);

    // Copy integer labels to GPU for backward kernel
    int* d_labels;
    CUDA_CHECK(cudaMalloc(&d_labels, B * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), B * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Backward closure
    // We need to capture d_labels (GPU pointer) and free it when done.
    // Using a shared_ptr with custom deleter to manage the GPU memory.
    auto labels_gpu = std::shared_ptr<int>(d_labels, [](int* p) { cudaFree(p); });

    loss->backward_fn = [logits, loss, labels_gpu, B, num_classes]() {
        if (logits->requires_grad) {
            logits->ensure_grad();
            int blk, thr;
            get_launch_config(B * num_classes, blk, thr);
            cross_entropy_backward_kernel<<<blk, thr>>>(
                logits->data, labels_gpu.get(), logits->grad,
                loss->grad, B, num_classes);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    };

    return loss;
}

// ===========================================================================
//  8. SIMPLE SCALAR/ELEMENT-WISE OPS (for testing the autograd engine)
// ===========================================================================

// --- Multiply by scalar: y = a * x ---
//
//  dL/dx[i] = dL/dy[i] * a
//
__global__ void mul_scalar_kernel(const float* x, float a, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = x[idx] * a;
}

__global__ void mul_scalar_backward_kernel(const float* grad_out, float a,
                                            float* grad_in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) grad_in[idx] += grad_out[idx] * a;
}

inline GradTensorPtr mul_scalar(GradTensorPtr x, float a) {
    auto out = make_grad_tensor(x->shape, x->requires_grad, "mul_s_out");
    out->op_name = "mul_scalar";
    out->children = {x};

    int blk, thr;
    get_launch_config(x->size, blk, thr);
    mul_scalar_kernel<<<blk, thr>>>(x->data, a, out->data, x->size);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (x->requires_grad) {
        out->backward_fn = [x, out, a]() {
            x->ensure_grad();
            int b, t;
            get_launch_config(x->size, b, t);
            mul_scalar_backward_kernel<<<b, t>>>(out->grad, a, x->grad, x->size);
            CUDA_CHECK(cudaDeviceSynchronize());
        };
    }
    return out;
}

// --- Add scalar: y = x + c ---
//
//  dL/dx[i] = dL/dy[i] * 1 = dL/dy[i]
//
inline GradTensorPtr add_scalar(GradTensorPtr x, float c) {
    auto out = make_grad_tensor(x->shape, x->requires_grad, "add_s_out");
    out->op_name = "add_scalar";
    out->children = {x};

    // Forward: just add c to each element
    std::vector<float> h(x->size);
    x->get_data_to_host(h.data());
    for (int i = 0; i < x->size; i++) h[i] += c;
    out->set_data_from_host(h.data());

    if (x->requires_grad) {
        out->backward_fn = [x, out]() {
            x->ensure_grad();
            int b, t;
            get_launch_config(x->size, b, t);
            accumulate_kernel<<<b, t>>>(x->grad, out->grad, x->size);
            CUDA_CHECK(cudaDeviceSynchronize());
        };
    }
    return out;
}

// --- Element-wise square: y = x^2 ---
//
//  dL/dx[i] = dL/dy[i] * 2 * x[i]
//
__global__ void square_kernel(const float* x, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = x[idx] * x[idx];
}

__global__ void square_backward_kernel(const float* grad_out, const float* x,
                                        float* grad_in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) grad_in[idx] += grad_out[idx] * 2.0f * x[idx];
}

inline GradTensorPtr square(GradTensorPtr x) {
    auto out = make_grad_tensor(x->shape, x->requires_grad, "sq_out");
    out->op_name = "square";
    out->children = {x};

    int blk, thr;
    get_launch_config(x->size, blk, thr);
    square_kernel<<<blk, thr>>>(x->data, out->data, x->size);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (x->requires_grad) {
        out->backward_fn = [x, out]() {
            x->ensure_grad();
            int b, t;
            get_launch_config(x->size, b, t);
            square_backward_kernel<<<b, t>>>(out->grad, x->data, x->grad, x->size);
            CUDA_CHECK(cudaDeviceSynchronize());
        };
    }
    return out;
}

// --- Sum all elements to scalar: y = SUM_i x[i] ---
//
//  dL/dx[i] = dL/dy * 1
//
//  Every element's gradient equals the upstream scalar gradient.
//
__global__ void sum_backward_kernel(const float* grad_out, float* grad_in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) grad_in[idx] += grad_out[0];  // grad_out is scalar
}

inline GradTensorPtr sum(GradTensorPtr x) {
    auto out = make_grad_tensor({1}, x->requires_grad, "sum_out");
    out->op_name = "sum";
    out->children = {x};

    // Forward: sum on CPU (small enough for testing)
    std::vector<float> h(x->size);
    x->get_data_to_host(h.data());
    float s = 0.0f;
    for (int i = 0; i < x->size; i++) s += h[i];
    out->set_data_from_host(&s);

    if (x->requires_grad) {
        out->backward_fn = [x, out]() {
            x->ensure_grad();
            int b, t;
            get_launch_config(x->size, b, t);
            sum_backward_kernel<<<b, t>>>(out->grad, x->grad, x->size);
            CUDA_CHECK(cudaDeviceSynchronize());
        };
    }
    return out;
}

// --- Element-wise multiply: z = a * b ---
//
//  dL/da[i] = dL/dz[i] * b[i]
//  dL/db[i] = dL/dz[i] * a[i]
//
__global__ void mul_elem_kernel(const float* a, const float* b, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = a[idx] * b[idx];
}

__global__ void mul_elem_backward_kernel(const float* grad_out,
                                          const float* other,
                                          float* grad_self, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) grad_self[idx] += grad_out[idx] * other[idx];
}

inline GradTensorPtr mul(GradTensorPtr a, GradTensorPtr b) {
    assert(a->size == b->size);
    bool needs_grad = a->requires_grad || b->requires_grad;
    auto out = make_grad_tensor(a->shape, needs_grad, "mul_out");
    out->op_name = "mul";
    out->children = {a, b};

    int blk, thr;
    get_launch_config(a->size, blk, thr);
    mul_elem_kernel<<<blk, thr>>>(a->data, b->data, out->data, a->size);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (needs_grad) {
        out->backward_fn = [a, b, out]() {
            int bl, th;
            get_launch_config(a->size, bl, th);
            if (a->requires_grad) {
                a->ensure_grad();
                mul_elem_backward_kernel<<<bl, th>>>(out->grad, b->data, a->grad, a->size);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            if (b->requires_grad) {
                b->ensure_grad();
                mul_elem_backward_kernel<<<bl, th>>>(out->grad, a->data, b->grad, b->size);
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        };
    }
    return out;
}

} // namespace autograd
