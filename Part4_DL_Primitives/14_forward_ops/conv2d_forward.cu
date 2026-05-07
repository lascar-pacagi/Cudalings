// ===========================================================================
// Chapter 14: conv2d_forward.cu -- Conv2D Forward Pass Kernel
// ===========================================================================
//
// Implements a direct 2D convolution in NCHW format (batch, channels,
// height, width). Each CUDA thread computes exactly one output element.
//
// We use "direct convolution" rather than the im2col approach for
// educational clarity. Direct convolution has each thread loop over the
// input channels and kernel spatial dimensions, accumulating the dot
// product that forms one output pixel.
//
// ===========================================================================
//
//  CONVOLUTION INDEXING DIAGRAM
//  ============================
//
//  For output element output[b][oc][oh][ow]:
//
//    weight tensor: (OC, IC, KH, KW)       input tensor: (B, IC, IH, IW)
//
//    For oc=0, one "filter" is weight[0][*][*][*] -- shape (IC, KH, KW)
//    This filter has IC "slices", each of size KH x KW.
//
//    output[b][oc][oh][ow] =
//        SUM_{ic=0}^{IC-1}
//          SUM_{kh=0}^{KH-1}
//            SUM_{kw=0}^{KW-1}
//              input[b][ic][oh*stride + kh - pad][ow*stride + kw - pad]
//              * weight[oc][ic][kh][kw]
//
//    With pad=1, stride=1, KH=KW=3:
//
//      ih = oh + kh - 1     (ranges from oh-1 to oh+1)
//      iw = ow + kw - 1     (ranges from ow-1 to ow+1)
//
//      If ih or iw is out of bounds [0, IH) or [0, IW), use 0 (zero-padding).
//
//  EXAMPLE (IC=2, KH=KW=3, pad=1, stride=1):
//
//    For output[0][0][1][1]:
//
//      Channel 0 of input, 3x3 patch centered at (1,1):
//        input[0][0][0][0]  input[0][0][0][1]  input[0][0][0][2]
//        input[0][0][1][0]  input[0][0][1][1]  input[0][0][1][2]
//        input[0][0][2][0]  input[0][0][2][1]  input[0][0][2][2]
//
//      Multiply element-wise with weight[0][0][*][*], sum up.
//
//      Channel 1 of input, same 3x3 patch:
//        input[0][1][0][0]  input[0][1][0][1]  input[0][1][0][2]
//        input[0][1][1][0]  input[0][1][1][1]  input[0][1][1][2]
//        input[0][1][2][0]  input[0][1][2][1]  input[0][1][2][2]
//
//      Multiply element-wise with weight[0][1][*][*], sum up.
//
//      Add both partial sums -> output[0][0][1][1]
//
// ===========================================================================

#include "../13_tensor_class/tensor.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>

// ===========================================================================
// CUDA Kernel: Direct 2D Convolution Forward
// ===========================================================================
// Each thread computes one element of the output tensor.
//
// Thread mapping:
//   Global thread index -> (b, oc, oh, ow)
//   where b  = batch index
//         oc = output channel
//         oh = output height position
//         ow = output width position
//
// Parameters:
//   input:  pointer to input data,  shape (B, IC, IH, IW) in NCHW
//   weight: pointer to filter data, shape (OC, IC, KH, KW)
//   output: pointer to output data, shape (B, OC, OH, OW)
//   B, IC, IH, IW:    input dimensions
//   OC, OH, OW:       output dimensions
//   KH, KW:           kernel dimensions
//   stride, pad:      convolution parameters
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
    // Total number of output elements: B * OC * OH * OW
    int total = B * OC * OH * OW;

    // Grid-stride loop: each thread may compute multiple output elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += grid_stride) {
        // ---------------------------------------------------------------
        // Decompose flat index i -> (b, oc, oh, ow)
        //
        // Memory layout is NCHW (row-major), so:
        //   i = b * (OC*OH*OW) + oc * (OH*OW) + oh * OW + ow
        // ---------------------------------------------------------------
        int ow = i % OW;
        int tmp = i / OW;
        int oh = tmp % OH;
        tmp = tmp / OH;
        int oc = tmp % OC;
        int b  = tmp / OC;

        // ---------------------------------------------------------------
        // Accumulate the convolution sum for this output element.
        //
        // output[b][oc][oh][ow] =
        //   SUM_{ic, kh, kw} input[b][ic][ih][iw] * weight[oc][ic][kh][kw]
        //
        // where ih = oh * stride + kh - pad
        //       iw = ow * stride + kw - pad
        //
        // If (ih, iw) is out of bounds, the input value is 0 (zero-padding).
        // ---------------------------------------------------------------
        float sum = 0.0f;

        for (int ic = 0; ic < IC; ic++) {
            for (int kh = 0; kh < KH; kh++) {
                for (int kw = 0; kw < KW; kw++) {
                    // Compute the input spatial position
                    int ih = oh * stride + kh - pad;
                    int iw = ow * stride + kw - pad;

                    // Zero-padding: skip if out of bounds
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        // input index: b*IC*IH*IW + ic*IH*IW + ih*IW + iw
                        int input_idx = ((b * IC + ic) * IH + ih) * IW + iw;

                        // weight index: oc*IC*KH*KW + ic*KH*KW + kh*KW + kw
                        int weight_idx = ((oc * IC + ic) * KH + kh) * KW + kw;

                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }

        // Write the result to the output tensor
        // output index: b*OC*OH*OW + oc*OH*OW + oh*OW + ow = i
        output[i] = sum;
    }
}

// ===========================================================================
// Host Wrapper: conv2d_forward
// ===========================================================================
// Takes Tensor objects for input and weight, returns the output Tensor.
//
//   input:  (B, IC, IH, IW) on GPU
//   weight: (OC, IC, KH, KW) on GPU
//   stride: convolution stride (default 1)
//   pad:    zero-padding on each side (default 0)
//
// Returns: output (B, OC, OH, OW) on GPU
//
// Output spatial dimensions:
//   OH = (IH + 2*pad - KH) / stride + 1
//   OW = (IW + 2*pad - KW) / stride + 1
// ===========================================================================

Tensor<float> conv2d_forward(
    const Tensor<float>& input,
    const Tensor<float>& weight,
    int stride = 1,
    int pad = 0
) {
    // Extract dimensions from tensor shapes
    int B  = input.shape_[0];   // batch size
    int IC = input.shape_[1];   // input channels
    int IH = input.shape_[2];   // input height
    int IW = input.shape_[3];   // input width

    int OC = weight.shape_[0];  // output channels
    // weight.shape_[1] should == IC
    int KH = weight.shape_[2];  // kernel height
    int KW = weight.shape_[3];  // kernel width

    // Compute output spatial dimensions
    int OH = (IH + 2 * pad - KH) / stride + 1;
    int OW = (IW + 2 * pad - KW) / stride + 1;

    // Allocate output tensor on GPU
    Tensor<float> output({B, OC, OH, OW}, Device::GPU);

    // Launch kernel
    int total = B * OC * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv2d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr(), weight.data_ptr(), output.data_ptr(),
        B, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, pad
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// ===========================================================================
// CPU Reference: conv2d_forward_cpu
// ===========================================================================
// Straightforward CPU implementation for verification.
// Same formula, just four nested loops (plus the three inner loops).
// ===========================================================================

void conv2d_forward_cpu(
    const float* input, const float* weight, float* output,
    int B, int IC, int IH, int IW,
    int OC, int OH, int OW,
    int KH, int KW,
    int stride, int pad
) {
    for (int b = 0; b < B; b++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            for (int kw = 0; kw < KW; kw++) {
                                int ih = oh * stride + kh - pad;
                                int iw = ow * stride + kw - pad;
                                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                                    int in_idx = ((b * IC + ic) * IH + ih) * IW + iw;
                                    int w_idx  = ((oc * IC + ic) * KH + kh) * KW + kw;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = ((b * OC + oc) * OH + oh) * OW + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// ===========================================================================
// Test: Conv2D Forward
// ===========================================================================

int main() {
    printf("=== Chapter 14: Conv2D Forward Pass Test ===\n\n");

    // -----------------------------------------------------------------
    // Test 1: Small known-values test
    // -----------------------------------------------------------------
    // Input: (1, 1, 3, 3) -- 1 batch, 1 channel, 3x3
    // Weight: (1, 1, 3, 3) -- 1 output channel, 1 input channel, 3x3
    // No padding, stride=1 -> Output: (1, 1, 1, 1) -- single value
    //
    // The single output value is the full dot product of input and filter.
    // -----------------------------------------------------------------
    {
        printf("Test 1: (1,1,3,3) input, (1,1,3,3) weight, pad=0, stride=1\n");

        // Input: 1..9
        float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        // Weight: all ones
        float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        // Expected: sum(1..9) = 45
        float expected = 45.0f;

        Tensor<float> input({1, 1, 3, 3}, input_data, Device::GPU);
        Tensor<float> weight({1, 1, 3, 3}, weight_data, Device::GPU);

        Tensor<float> output = conv2d_forward(input, weight, 1, 0);
        Tensor<float> out_cpu = output.to_cpu();

        printf("  Output: %.2f (expected: %.2f) -- %s\n",
               out_cpu(0, 0, 0, 0), expected,
               fabsf(out_cpu(0, 0, 0, 0) - expected) < 1e-4 ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 2: Padding test
    // -----------------------------------------------------------------
    // Input: (1, 1, 3, 3), Weight: (1, 1, 3, 3), pad=1, stride=1
    // Output should be (1, 1, 3, 3) -- same spatial size
    // -----------------------------------------------------------------
    {
        printf("\nTest 2: (1,1,3,3) input, (1,1,3,3) weight, pad=1, stride=1\n");

        // Input: all 1s, weight: all 1s
        // Center element output[0][0][1][1] should see full 3x3 = 9
        // Corner element output[0][0][0][0] should see 2x2 = 4 (padding zeros)
        float input_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

        Tensor<float> input({1, 1, 3, 3}, input_data, Device::GPU);
        Tensor<float> weight({1, 1, 3, 3}, weight_data, Device::GPU);

        Tensor<float> output = conv2d_forward(input, weight, 1, 1);
        Tensor<float> out_cpu = output.to_cpu();

        printf("  Shape: (%d, %d, %d, %d)\n",
               out_cpu.shape_[0], out_cpu.shape_[1],
               out_cpu.shape_[2], out_cpu.shape_[3]);

        // Corner: 2x2 overlap = 4
        printf("  Corner [0][0][0][0]: %.2f (expected: 4.00) -- %s\n",
               out_cpu(0, 0, 0, 0),
               fabsf(out_cpu(0, 0, 0, 0) - 4.0f) < 1e-4 ? "PASS" : "FAIL");

        // Edge: 2x3 overlap = 6
        printf("  Edge   [0][0][0][1]: %.2f (expected: 6.00) -- %s\n",
               out_cpu(0, 0, 0, 1),
               fabsf(out_cpu(0, 0, 0, 1) - 6.0f) < 1e-4 ? "PASS" : "FAIL");

        // Center: full 3x3 overlap = 9
        printf("  Center [0][0][1][1]: %.2f (expected: 9.00) -- %s\n",
               out_cpu(0, 0, 1, 1),
               fabsf(out_cpu(0, 0, 1, 1) - 9.0f) < 1e-4 ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 3: Multi-channel test with CPU verification
    // -----------------------------------------------------------------
    // Input: (2, 4, 8, 8) -- our ResNet input shape
    // Weight: (16, 4, 3, 3) -- Conv(4->16, 3x3)
    // pad=1, stride=1 -> Output: (2, 16, 8, 8)
    // Verify GPU output against CPU reference.
    // -----------------------------------------------------------------
    {
        printf("\nTest 3: (2,4,8,8) -> Conv(4->16, 3x3, pad=1) -> (2,16,8,8)\n");

        int B = 2, IC = 4, IH = 8, IW = 8;
        int OC = 16, KH = 3, KW = 3;
        int pad = 1, stride = 1;
        int OH = (IH + 2 * pad - KH) / stride + 1;
        int OW = (IW + 2 * pad - KW) / stride + 1;

        // Create random input and weight on CPU
        Tensor<float> input_cpu = Tensor<float>::randn({B, IC, IH, IW}, Device::CPU);
        Tensor<float> weight_cpu = Tensor<float>::randn({OC, IC, KH, KW}, Device::CPU);

        // GPU forward
        Tensor<float> input_gpu = input_cpu.to_gpu();
        Tensor<float> weight_gpu = weight_cpu.to_gpu();
        Tensor<float> output_gpu = conv2d_forward(input_gpu, weight_gpu, stride, pad);
        Tensor<float> output_from_gpu = output_gpu.to_cpu();

        // CPU reference
        int out_size = B * OC * OH * OW;
        std::vector<float> output_ref(out_size, 0.0f);
        conv2d_forward_cpu(
            input_cpu.data_ptr(), weight_cpu.data_ptr(), output_ref.data(),
            B, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad
        );

        // Compare
        float max_err = 0.0f;
        for (int i = 0; i < out_size; i++) {
            float err = fabsf(output_from_gpu.data_ptr()[i] - output_ref[i]);
            if (err > max_err) max_err = err;
        }

        printf("  Output shape: (%d, %d, %d, %d)\n", B, OC, OH, OW);
        printf("  Max error vs CPU: %.6e -- %s\n",
               max_err, max_err < 1e-4 ? "PASS" : "FAIL");
    }

    printf("\n=== Conv2D Forward Tests Complete ===\n");
    return 0;
}
