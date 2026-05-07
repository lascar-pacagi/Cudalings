// ===========================================================================
// Chapter 15: conv2d_backward.cu -- Conv2D Backward Pass Kernels
// ===========================================================================
//
// Implements the two gradient computations for 2D convolution:
//
//   Kernel 1: conv2d_backward_input  -- dL/d(input)
//   Kernel 2: conv2d_backward_weight -- dL/d(weight)
//
// Both are verified against numerical gradients (finite differences).
//
// ===========================================================================
//
//  RECAP: FORWARD PASS
//  ====================
//
//  output[b][oc][oh][ow] =
//      SUM_{ic,kh,kw} input[b][ic][oh*s+kh-pad][ow*s+kw-pad] * weight[oc][ic][kh][kw]
//
//  where:
//    - input  is (B, IC, IH, IW)   -- NCHW
//    - weight is (OC, IC, KH, KW)
//    - output is (B, OC, OH, OW)
//    - OH = (IH + 2*pad - KH) / stride + 1
//    - OW = (IW + 2*pad - KW) / stride + 1
//
// ===========================================================================
//
//  BACKWARD PASS: GRADIENT W.R.T. INPUT
//  =====================================
//
//  Given:  dL/d(output)   shape (B, OC, OH, OW)   -- upstream gradient
//  Wanted: dL/d(input)    shape (B, IC, IH, IW)
//
//  From the forward equation, input[b][ic][ih][iw] contributes to every
//  output element where ih = oh*s + kh - pad, i.e., oh = (ih - kh + pad) / s.
//
//  For stride = 1:
//
//    dL/d(input[b][ic][ih][iw]) =
//      SUM_{oc=0}^{OC-1}
//        SUM_{kh=0}^{KH-1}
//          SUM_{kw=0}^{KW-1}
//            weight[oc][ic][kh][kw] * grad_output[b][oc][ih - kh + pad][iw - kw + pad]
//
//  where we check bounds: 0 <= (ih - kh + pad) < OH  and  0 <= (iw - kw + pad) < OW
//
//  GEOMETRIC INTERPRETATION:
//  =========================
//
//  This is equivalent to convolving grad_output with the weight filter
//  rotated by 180 degrees, with padding = (KH-1-pad):
//
//    Original filter:                  Rotated filter (180 deg):
//    +----+----+----+                  +----+----+----+
//    | w00| w01| w02|                  | w22| w21| w20|
//    +----+----+----+                  +----+----+----+
//    | w10| w11| w12|        -->       | w12| w11| w10|
//    +----+----+----+                  +----+----+----+
//    | w20| w21| w22|                  | w02| w01| w00|
//    +----+----+----+                  +----+----+----+
//
//  INDEX MAPPING DIAGRAM (stride=1, pad=1, KH=KW=3, IH=IW=OH=OW=3):
//
//  For input position (ih=1, iw=1), which output positions contribute?
//
//    kh=0: oh = 1-0+1 = 2   valid (0<=2<3)  -> grad_out[b][oc][2][?] * w[oc][ic][0][?]
//    kh=1: oh = 1-1+1 = 1   valid           -> grad_out[b][oc][1][?] * w[oc][ic][1][?]
//    kh=2: oh = 1-2+1 = 0   valid           -> grad_out[b][oc][0][?] * w[oc][ic][2][?]
//
//    Similarly for kw dimension.
//
//    The key: we iterate kh from 0 to KH-1, and compute oh = ih - kh + pad.
//    If oh is in [0, OH), we include that term.
//
// ===========================================================================
//
//  BACKWARD PASS: GRADIENT W.R.T. WEIGHT
//  ======================================
//
//  Given:  dL/d(output)   shape (B, OC, OH, OW)
//  Wanted: dL/d(weight)   shape (OC, IC, KH, KW)
//
//  Each weight[oc][ic][kh][kw] appears in the forward pass for every (b, oh, ow):
//
//    dL/d(weight[oc][ic][kh][kw]) =
//      SUM_{b=0}^{B-1}
//        SUM_{oh=0}^{OH-1}
//          SUM_{ow=0}^{OW-1}
//            grad_output[b][oc][oh][ow]
//            * input[b][ic][oh*s + kh - pad][ow*s + kw - pad]
//
//  This is a CORRELATION of the input with grad_output (not a convolution).
//
// ===========================================================================

#include "../13_tensor_class/tensor.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>

// ===========================================================================
// Forward pass kernel (needed for finite-difference gradient checking)
// ===========================================================================

__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int B, int IC, int IH, int IW,
    int OC, int OH, int OW,
    int KH, int KW,
    int stride, int pad
) {
    int total = B * OC * OH * OW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += grid_stride) {
        int ow = i % OW;
        int tmp = i / OW;
        int oh = tmp % OH;
        tmp = tmp / OH;
        int oc = tmp % OC;
        int b  = tmp / OC;

        float sum = 0.0f;
        for (int ic = 0; ic < IC; ic++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    int ih = oh * stride + kh - pad;
                    int iw = ow * stride + kw - pad;
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        int input_idx = ((b * IC + ic) * IH + ih) * IW + iw;
                        int weight_idx = ((oc * IC + ic) * KH + kh) * KW + kw;
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        output[i] = sum;
    }
}

// Host wrapper for forward pass
Tensor<float> conv2d_forward(
    const Tensor<float>& input, const Tensor<float>& weight,
    int stride, int pad
) {
    int B = input.shape_[0], IC = input.shape_[1];
    int IH = input.shape_[2], IW = input.shape_[3];
    int OC = weight.shape_[0], KH = weight.shape_[2], KW = weight.shape_[3];
    int OH = (IH + 2 * pad - KH) / stride + 1;
    int OW = (IW + 2 * pad - KW) / stride + 1;

    Tensor<float> output({B, OC, OH, OW}, Device::GPU);
    int total = B * OC * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr(), weight.data_ptr(), output.data_ptr(),
        B, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}

// ===========================================================================
// CUDA Kernel: Conv2D Backward w.r.t. Input
// ===========================================================================
//
// Each thread computes dL/d(input[b][ic][ih][iw]) for one input element.
//
// The chain rule gives:
//
//   dL/d(input[b][ic][ih][iw]) =
//     SUM_{oc} SUM_{kh} SUM_{kw}
//       weight[oc][ic][kh][kw] * grad_output[b][oc][oh][ow]
//
//   where oh = ih - kh + pad   (for stride=1)
//         ow = iw - kw + pad
//
//   Bounds: 0 <= oh < OH, 0 <= ow < OW
//
// For general stride, the relationship is:
//   ih = oh * stride + kh - pad
//   oh = (ih - kh + pad) / stride   (must be integer and in bounds)
//
// ===========================================================================

__global__ void conv2d_backward_input_kernel(
    const float* __restrict__ grad_output,  // (B, OC, OH, OW) -- upstream gradient
    const float* __restrict__ weight,       // (OC, IC, KH, KW) -- filter weights
    float* __restrict__ grad_input,         // (B, IC, IH, IW) -- gradient to compute
    int B, int IC, int IH, int IW,
    int OC, int OH, int OW,
    int KH, int KW,
    int stride, int pad
) {
    // Total number of input elements: B * IC * IH * IW
    int total = B * IC * IH * IW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += grid_stride) {
        // ---------------------------------------------------------------
        // Decompose flat index i -> (b, ic, ih, iw) for input tensor
        //
        // Memory layout is NCHW:
        //   i = b * (IC*IH*IW) + ic * (IH*IW) + ih * IW + iw
        // ---------------------------------------------------------------
        int iw = i % IW;
        int tmp = i / IW;
        int ih = tmp % IH;
        tmp = tmp / IH;
        int ic = tmp % IC;
        int b  = tmp / IC;

        // ---------------------------------------------------------------
        // Accumulate dL/d(input[b][ic][ih][iw])
        //
        // We need to find all (oc, oh, ow) output positions that used
        // this input element in the forward pass:
        //
        //   Forward: oh*stride + kh - pad = ih
        //            ow*stride + kw - pad = iw
        //
        //   Rearranging: oh = (ih - kh + pad) / stride
        //                ow = (iw - kw + pad) / stride
        //
        //   For these to be valid:
        //     (ih - kh + pad) must be >= 0, divisible by stride, and oh < OH
        //     (iw - kw + pad) must be >= 0, divisible by stride, and ow < OW
        //
        // For stride=1, this simplifies to iterating kh, kw and computing
        // oh = ih - kh + pad, ow = iw - kw + pad directly.
        // ---------------------------------------------------------------
        float sum = 0.0f;

        for (int oc = 0; oc < OC; oc++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    // Compute which output position used this input element
                    // with this particular kernel offset (kh, kw)
                    int oh_times_s = ih - kh + pad;
                    int ow_times_s = iw - kw + pad;

                    // For stride > 1, check divisibility
                    if (oh_times_s % stride != 0 || ow_times_s % stride != 0)
                        continue;

                    int oh = oh_times_s / stride;
                    int ow = ow_times_s / stride;

                    // Bounds check on output position
                    if (oh >= 0 && oh < OH && ow >= 0 && ow < OW) {
                        // grad_output index: b*OC*OH*OW + oc*OH*OW + oh*OW + ow
                        int go_idx = ((b * OC + oc) * OH + oh) * OW + ow;

                        // weight index: oc*IC*KH*KW + ic*KH*KW + kh*KW + kw
                        int w_idx = ((oc * IC + ic) * KH + kh) * KW + kw;

                        sum += grad_output[go_idx] * weight[w_idx];
                    }
                }
            }
        }

        grad_input[i] = sum;
    }
}

// ===========================================================================
// CUDA Kernel: Conv2D Backward w.r.t. Weight
// ===========================================================================
//
// Each thread computes dL/d(weight[oc][ic][kh][kw]) for one weight element.
//
// The chain rule gives:
//
//   dL/d(weight[oc][ic][kh][kw]) =
//     SUM_{b} SUM_{oh} SUM_{ow}
//       grad_output[b][oc][oh][ow] * input[b][ic][oh*s + kh - pad][ow*s + kw - pad]
//
//   Bounds: input position must be in [0, IH) x [0, IW)
//
// This is a correlation: for each weight element, we slide across all
// spatial positions and batch elements, accumulating the product of the
// upstream gradient and the corresponding input value.
//
// ===========================================================================

__global__ void conv2d_backward_weight_kernel(
    const float* __restrict__ grad_output,  // (B, OC, OH, OW) -- upstream gradient
    const float* __restrict__ input,        // (B, IC, IH, IW) -- saved from forward
    float* __restrict__ grad_weight,        // (OC, IC, KH, KW) -- gradient to compute
    int B, int IC, int IH, int IW,
    int OC, int OH, int OW,
    int KH, int KW,
    int stride, int pad
) {
    // Total number of weight elements: OC * IC * KH * KW
    int total = OC * IC * KH * KW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += grid_stride) {
        // ---------------------------------------------------------------
        // Decompose flat index i -> (oc, ic, kh, kw) for weight tensor
        //
        // Memory layout: (OC, IC, KH, KW) row-major
        //   i = oc * (IC*KH*KW) + ic * (KH*KW) + kh * KW + kw
        // ---------------------------------------------------------------
        int kw = i % KW;
        int tmp = i / KW;
        int kh = tmp % KH;
        tmp = tmp / KH;
        int ic = tmp % IC;
        int oc = tmp / IC;

        // ---------------------------------------------------------------
        // Accumulate dL/d(weight[oc][ic][kh][kw])
        //
        // Sum over all batch elements and spatial output positions:
        //
        //   SUM_{b, oh, ow} grad_output[b][oc][oh][ow]
        //                   * input[b][ic][oh*s + kh - pad][ow*s + kw - pad]
        //
        // The input position must be in bounds (zero-padding in forward
        // means the input value was 0, so those terms contribute nothing).
        // ---------------------------------------------------------------
        float sum = 0.0f;

        for (int b = 0; b < B; b++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    // Compute the input spatial position
                    int ih = oh * stride + kh - pad;
                    int iw = ow * stride + kw - pad;

                    // Skip if out of bounds (corresponds to zero-padded region)
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        // grad_output index
                        int go_idx = ((b * OC + oc) * OH + oh) * OW + ow;

                        // input index
                        int in_idx = ((b * IC + ic) * IH + ih) * IW + iw;

                        sum += grad_output[go_idx] * input[in_idx];
                    }
                }
            }
        }

        grad_weight[i] = sum;
    }
}

// ===========================================================================
// Host Wrapper: conv2d_backward_input
// ===========================================================================
// Computes dL/d(input) given dL/d(output) and the weights.
//
//   grad_output: (B, OC, OH, OW) on GPU
//   weight:      (OC, IC, KH, KW) on GPU
//   stride, pad: same as forward pass
//   IH, IW:      original input spatial dimensions
//
// Returns: grad_input (B, IC, IH, IW) on GPU
// ===========================================================================

Tensor<float> conv2d_backward_input(
    const Tensor<float>& grad_output,
    const Tensor<float>& weight,
    int B, int IC, int IH, int IW,
    int stride, int pad
) {
    int OC = weight.shape_[0];
    int KH = weight.shape_[2];
    int KW = weight.shape_[3];
    int OH = grad_output.shape_[2];
    int OW = grad_output.shape_[3];

    Tensor<float> grad_input({B, IC, IH, IW}, Device::GPU);

    int total = B * IC * IH * IW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_backward_input_kernel<<<blocks, threads>>>(
        grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr(),
        B, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_input;
}

// ===========================================================================
// Host Wrapper: conv2d_backward_weight
// ===========================================================================
// Computes dL/d(weight) given dL/d(output) and the saved input.
//
//   grad_output: (B, OC, OH, OW) on GPU
//   input:       (B, IC, IH, IW) on GPU  -- saved from forward pass
//   stride, pad: same as forward pass
//
// Returns: grad_weight (OC, IC, KH, KW) on GPU
// ===========================================================================

Tensor<float> conv2d_backward_weight(
    const Tensor<float>& grad_output,
    const Tensor<float>& input,
    int OC, int KH, int KW,
    int stride, int pad
) {
    int B  = input.shape_[0];
    int IC = input.shape_[1];
    int IH = input.shape_[2];
    int IW = input.shape_[3];
    int OH = grad_output.shape_[2];
    int OW = grad_output.shape_[3];

    Tensor<float> grad_weight({OC, IC, KH, KW}, Device::GPU);

    int total = OC * IC * KH * KW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_backward_weight_kernel<<<blocks, threads>>>(
        grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr(),
        B, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return grad_weight;
}

// ===========================================================================
// CPU Reference: Forward pass (for finite difference checking)
// ===========================================================================

void conv2d_forward_cpu(
    const float* input, const float* weight, float* output,
    int B, int IC, int IH, int IW,
    int OC, int OH, int OW,
    int KH, int KW,
    int stride, int pad
) {
    for (int b = 0; b < B; b++)
        for (int oc = 0; oc < OC; oc++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < IC; ic++)
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++) {
                                int ih = oh * stride + kh - pad;
                                int iw_pos = ow * stride + kw - pad;
                                if (ih >= 0 && ih < IH && iw_pos >= 0 && iw_pos < IW) {
                                    int in_idx = ((b * IC + ic) * IH + ih) * IW + iw_pos;
                                    int w_idx  = ((oc * IC + ic) * KH + kh) * KW + kw;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                    int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
                    output[out_idx] = sum;
                }
}

// ===========================================================================
// Finite Difference Gradient Check (CPU, for correctness verification)
// ===========================================================================
//
// For a scalar-valued function L = sum(output), we compute:
//
//   dL/d(param[i]) = (L(param[i] + eps) - L(param[i] - eps)) / (2 * eps)
//
// This approximates the true gradient to O(eps^2).
// We compare against our analytical backward-pass gradient.
//
// ===========================================================================

float compute_loss_cpu(
    const float* input, const float* weight,
    int B, int IC, int IH, int IW,
    int OC, int OH, int OW,
    int KH, int KW, int stride, int pad
) {
    int out_size = B * OC * OH * OW;
    std::vector<float> output(out_size);
    conv2d_forward_cpu(input, weight, output.data(),
                       B, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad);
    // Loss = sum of all output elements (simple scalar loss for testing)
    float loss = 0.0f;
    for (int i = 0; i < out_size; i++) loss += output[i];
    return loss;
}

// ===========================================================================
// Test: Conv2D Backward
// ===========================================================================

int main() {
    printf("=== Chapter 15: Conv2D Backward Pass Test ===\n\n");

    // -----------------------------------------------------------------
    // Test configuration
    // -----------------------------------------------------------------
    int B = 2, IC = 3, IH = 5, IW = 5;
    int OC = 4, KH = 3, KW = 3;
    int pad = 1, stride = 1;
    int OH = (IH + 2 * pad - KH) / stride + 1;  // = 5
    int OW = (IW + 2 * pad - KW) / stride + 1;  // = 5

    printf("Configuration:\n");
    printf("  Input:  (%d, %d, %d, %d)\n", B, IC, IH, IW);
    printf("  Weight: (%d, %d, %d, %d)\n", OC, IC, KH, KW);
    printf("  Output: (%d, %d, %d, %d)\n", B, OC, OH, OW);
    printf("  Padding: %d, Stride: %d\n\n", pad, stride);

    // Create random input and weight on CPU
    Tensor<float> input_cpu = Tensor<float>::randn({B, IC, IH, IW}, Device::CPU);
    Tensor<float> weight_cpu = Tensor<float>::randn({OC, IC, KH, KW}, Device::CPU);

    // Scale down for numerical stability
    for (int i = 0; i < input_cpu.size_; i++)
        input_cpu.data_ptr()[i] *= 0.1f;
    for (int i = 0; i < weight_cpu.size_; i++)
        weight_cpu.data_ptr()[i] *= 0.1f;

    // -----------------------------------------------------------------
    // Forward pass on GPU
    // -----------------------------------------------------------------
    Tensor<float> input_gpu = input_cpu.to_gpu();
    Tensor<float> weight_gpu = weight_cpu.to_gpu();
    Tensor<float> output_gpu = conv2d_forward(input_gpu, weight_gpu, stride, pad);

    // For testing, use grad_output = all 1s (gradient of sum(output) w.r.t. output)
    // This means dL/d(output) = 1 for all elements, and L = sum(output).
    Tensor<float> grad_output = Tensor<float>::ones({B, OC, OH, OW}, Device::GPU);

    // -----------------------------------------------------------------
    // Test 1: Gradient w.r.t. Input
    // -----------------------------------------------------------------
    {
        printf("Test 1: Gradient w.r.t. Input (conv2d_backward_input)\n");

        // Analytical gradient
        Tensor<float> grad_input = conv2d_backward_input(
            grad_output, weight_gpu, B, IC, IH, IW, stride, pad);
        Tensor<float> gi_cpu = grad_input.to_cpu();

        // Numerical gradient (finite differences)
        float eps = 1e-3f;
        int input_size = B * IC * IH * IW;
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;
        int num_checked = 0;

        for (int i = 0; i < input_size; i++) {
            float orig = input_cpu.data_ptr()[i];

            // f(x + eps)
            input_cpu.data_ptr()[i] = orig + eps;
            float loss_plus = compute_loss_cpu(
                input_cpu.data_ptr(), weight_cpu.data_ptr(),
                B, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad);

            // f(x - eps)
            input_cpu.data_ptr()[i] = orig - eps;
            float loss_minus = compute_loss_cpu(
                input_cpu.data_ptr(), weight_cpu.data_ptr(),
                B, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad);

            // Restore
            input_cpu.data_ptr()[i] = orig;

            // Numerical gradient
            float numerical = (loss_plus - loss_minus) / (2.0f * eps);
            float analytical = gi_cpu.data_ptr()[i];

            float abs_err = fabsf(numerical - analytical);
            float denom = fmaxf(fabsf(numerical) + fabsf(analytical), 1e-8f);
            float rel_err = abs_err / denom;

            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            num_checked++;
        }

        printf("  Checked %d elements\n", num_checked);
        printf("  Max absolute error: %.6e\n", max_abs_err);
        printf("  Max relative error: %.6e\n", max_rel_err);
        printf("  Result: %s\n\n",
               max_rel_err < 1e-3 ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 2: Gradient w.r.t. Weight
    // -----------------------------------------------------------------
    {
        printf("Test 2: Gradient w.r.t. Weight (conv2d_backward_weight)\n");

        // Analytical gradient
        Tensor<float> grad_weight = conv2d_backward_weight(
            grad_output, input_gpu, OC, KH, KW, stride, pad);
        Tensor<float> gw_cpu = grad_weight.to_cpu();

        // Numerical gradient (finite differences)
        float eps = 1e-3f;
        int weight_size = OC * IC * KH * KW;
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;
        int num_checked = 0;

        for (int i = 0; i < weight_size; i++) {
            float orig = weight_cpu.data_ptr()[i];

            // f(w + eps)
            weight_cpu.data_ptr()[i] = orig + eps;
            float loss_plus = compute_loss_cpu(
                input_cpu.data_ptr(), weight_cpu.data_ptr(),
                B, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad);

            // f(w - eps)
            weight_cpu.data_ptr()[i] = orig - eps;
            float loss_minus = compute_loss_cpu(
                input_cpu.data_ptr(), weight_cpu.data_ptr(),
                B, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad);

            // Restore
            weight_cpu.data_ptr()[i] = orig;

            float numerical = (loss_plus - loss_minus) / (2.0f * eps);
            float analytical = gw_cpu.data_ptr()[i];

            float abs_err = fabsf(numerical - analytical);
            float denom = fmaxf(fabsf(numerical) + fabsf(analytical), 1e-8f);
            float rel_err = abs_err / denom;

            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            num_checked++;
        }

        printf("  Checked %d elements\n", num_checked);
        printf("  Max absolute error: %.6e\n", max_abs_err);
        printf("  Max relative error: %.6e\n", max_rel_err);
        printf("  Result: %s\n\n",
               max_rel_err < 1e-3 ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 3: Simple known-values test
    // -----------------------------------------------------------------
    {
        printf("Test 3: Known-values sanity check\n");

        // Input (1,1,3,3) = all 1s, Weight (1,1,3,3) = all 1s
        // Forward: output (1,1,1,1) = 9 (sum of 3x3 of ones)
        // grad_output = 1 (scalar)
        //
        // grad_input: each input element contributes to exactly one output
        //   element (no padding), so dL/d(input[i]) = weight[i] = 1
        //
        // grad_weight: each weight element is multiplied by exactly one input
        //   element, so dL/d(weight[i]) = input[i] = 1

        float in_data[] = {1,1,1, 1,1,1, 1,1,1};
        float w_data[]  = {1,1,1, 1,1,1, 1,1,1};

        Tensor<float> in_t({1,1,3,3}, in_data, Device::GPU);
        Tensor<float> w_t({1,1,3,3}, w_data, Device::GPU);
        Tensor<float> go_t = Tensor<float>::ones({1,1,1,1}, Device::GPU);

        Tensor<float> gi = conv2d_backward_input(go_t, w_t, 1, 1, 3, 3, 1, 0);
        Tensor<float> gw = conv2d_backward_weight(go_t, in_t, 1, 3, 3, 1, 0);

        Tensor<float> gi_c = gi.to_cpu();
        Tensor<float> gw_c = gw.to_cpu();

        // grad_input should be all 1s
        bool gi_pass = true;
        for (int i = 0; i < 9; i++) {
            if (fabsf(gi_c.data_ptr()[i] - 1.0f) > 1e-5f) gi_pass = false;
        }
        printf("  grad_input all 1s: %s\n", gi_pass ? "PASS" : "FAIL");

        // grad_weight should be all 1s
        bool gw_pass = true;
        for (int i = 0; i < 9; i++) {
            if (fabsf(gw_c.data_ptr()[i] - 1.0f) > 1e-5f) gw_pass = false;
        }
        printf("  grad_weight all 1s: %s\n", gw_pass ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 4: Padded convolution
    // -----------------------------------------------------------------
    {
        printf("\nTest 4: Padded convolution backward (pad=1)\n");

        // Smaller test for padded case
        int tB = 1, tIC = 2, tIH = 4, tIW = 4;
        int tOC = 2, tKH = 3, tKW = 3;
        int tpad = 1, tstride = 1;
        int tOH = (tIH + 2*tpad - tKH) / tstride + 1;  // = 4
        int tOW = (tIW + 2*tpad - tKW) / tstride + 1;  // = 4

        Tensor<float> t_in_cpu = Tensor<float>::randn({tB, tIC, tIH, tIW}, Device::CPU);
        Tensor<float> t_w_cpu  = Tensor<float>::randn({tOC, tIC, tKH, tKW}, Device::CPU);

        // Scale down
        for (int i = 0; i < t_in_cpu.size_; i++) t_in_cpu.data_ptr()[i] *= 0.1f;
        for (int i = 0; i < t_w_cpu.size_; i++)  t_w_cpu.data_ptr()[i] *= 0.1f;

        Tensor<float> t_in_gpu = t_in_cpu.to_gpu();
        Tensor<float> t_w_gpu  = t_w_cpu.to_gpu();
        Tensor<float> t_go = Tensor<float>::ones({tB, tOC, tOH, tOW}, Device::GPU);

        // Analytical
        Tensor<float> t_gi = conv2d_backward_input(
            t_go, t_w_gpu, tB, tIC, tIH, tIW, tstride, tpad);
        Tensor<float> t_gi_c = t_gi.to_cpu();

        // Numerical
        float eps = 1e-3f;
        float max_rel = 0.0f;
        int in_sz = tB * tIC * tIH * tIW;
        for (int i = 0; i < in_sz; i++) {
            float orig = t_in_cpu.data_ptr()[i];
            t_in_cpu.data_ptr()[i] = orig + eps;
            float lp = compute_loss_cpu(t_in_cpu.data_ptr(), t_w_cpu.data_ptr(),
                tB, tIC, tIH, tIW, tOC, tOH, tOW, tKH, tKW, tstride, tpad);
            t_in_cpu.data_ptr()[i] = orig - eps;
            float lm = compute_loss_cpu(t_in_cpu.data_ptr(), t_w_cpu.data_ptr(),
                tB, tIC, tIH, tIW, tOC, tOH, tOW, tKH, tKW, tstride, tpad);
            t_in_cpu.data_ptr()[i] = orig;

            float num = (lp - lm) / (2.0f * eps);
            float ana = t_gi_c.data_ptr()[i];
            float rel = fabsf(num - ana) / fmaxf(fabsf(num) + fabsf(ana), 1e-8f);
            if (rel > max_rel) max_rel = rel;
        }
        printf("  grad_input max relative error: %.6e -- %s\n",
               max_rel, max_rel < 1e-3 ? "PASS" : "FAIL");

        // Weight gradient
        Tensor<float> t_gw = conv2d_backward_weight(
            t_go, t_in_gpu, tOC, tKH, tKW, tstride, tpad);
        Tensor<float> t_gw_c = t_gw.to_cpu();

        max_rel = 0.0f;
        int w_sz = tOC * tIC * tKH * tKW;
        for (int i = 0; i < w_sz; i++) {
            float orig = t_w_cpu.data_ptr()[i];
            t_w_cpu.data_ptr()[i] = orig + eps;
            float lp = compute_loss_cpu(t_in_cpu.data_ptr(), t_w_cpu.data_ptr(),
                tB, tIC, tIH, tIW, tOC, tOH, tOW, tKH, tKW, tstride, tpad);
            t_w_cpu.data_ptr()[i] = orig - eps;
            float lm = compute_loss_cpu(t_in_cpu.data_ptr(), t_w_cpu.data_ptr(),
                tB, tIC, tIH, tIW, tOC, tOH, tOW, tKH, tKW, tstride, tpad);
            t_w_cpu.data_ptr()[i] = orig;

            float num = (lp - lm) / (2.0f * eps);
            float ana = t_gw_c.data_ptr()[i];
            float rel = fabsf(num - ana) / fmaxf(fabsf(num) + fabsf(ana), 1e-8f);
            if (rel > max_rel) max_rel = rel;
        }
        printf("  grad_weight max relative error: %.6e -- %s\n",
               max_rel, max_rel < 1e-3 ? "PASS" : "FAIL");
    }

    printf("\n=== Conv2D Backward Tests Complete ===\n");
    return 0;
}
