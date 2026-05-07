// ===========================================================================
// Chapter 14: linear_relu_pool.cu -- Linear, ReLU, GAP, Cross-Entropy
// ===========================================================================
//
// This file implements four operations commonly used in the "head" of a
// ResNet (after the convolutional blocks):
//
//   1. Linear (fully connected):  y = x @ W^T + b
//   2. ReLU activation:           y = max(0, x)
//   3. Global Average Pooling:    (B, C, H, W) -> (B, C) via mean over H,W
//   4. Cross-Entropy Loss:        softmax + negative log-likelihood
//
// Each operation is implemented as a CUDA kernel with a host wrapper
// that takes and returns Tensor objects.
//
// ===========================================================================

#include "../13_tensor_class/tensor.cuh"
#include <cstdio>
#include <cmath>
#include <cfloat>

// ===========================================================================
// Kernel: ReLU Forward
// ===========================================================================
// The simplest possible activation function:
//   output[i] = max(0, input[i])
//
// One thread per element, grid-stride loop for large tensors.
// No parameters, no cross-element dependencies.
//
// Math:
//   ReLU(x) = x  if x > 0
//           = 0  otherwise
//
// Derivative (for backprop, implemented in Chapter 15):
//   d ReLU/dx = 1  if x > 0
//             = 0  otherwise
// ===========================================================================

__global__ void relu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

// ===========================================================================
// Host Wrapper: relu_forward
// ===========================================================================
// Input:  any shape tensor on GPU
// Output: same shape tensor on GPU, with ReLU applied element-wise
// ===========================================================================

Tensor<float> relu_forward(const Tensor<float>& input) {
    Tensor<float> output(input.shape_, Device::GPU);

    int threads = 256;
    int blocks = (input.size_ + threads - 1) / threads;

    relu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr(), output.data_ptr(), input.size_
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// ===========================================================================
// Kernel: Global Average Pooling Forward
// ===========================================================================
// Reduces spatial dimensions by averaging over H and W:
//
//   output[b][c] = (1 / (H * W)) * SUM_{h=0}^{H-1} SUM_{w=0}^{W-1}
//                                      input[b][c][h][w]
//
// Input:  (B, C, H, W) -- 4D convolutional feature map
// Output: (B, C)        -- 2D, one value per batch element per channel
//
// This bridges conv layers (4D) and fully connected layers (2D).
//
// Thread mapping: one thread per (b, c) pair.
// Each thread loops over the H*W spatial positions to compute the average.
// ===========================================================================

__global__ void global_avg_pool_kernel(
    const float* __restrict__ input,   // (B, C, H, W) in NCHW
    float* __restrict__ output,        // (B, C)
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    int stride = blockDim.x * gridDim.x;

    int HW = H * W;

    for (int i = idx; i < total; i += stride) {
        int b = i / C;
        int c = i % C;

        // Sum all spatial elements for this (b, c)
        float sum = 0.0f;
        int base = (b * C + c) * HW;  // start of input[b][c][0][0]
        for (int s = 0; s < HW; s++) {
            sum += input[base + s];
        }

        // Average
        output[i] = sum / (float)HW;
    }
}

// ===========================================================================
// Host Wrapper: global_avg_pool_forward
// ===========================================================================

Tensor<float> global_avg_pool_forward(const Tensor<float>& input) {
    int B = input.shape_[0];
    int C = input.shape_[1];
    int H = input.shape_[2];
    int W = input.shape_[3];

    Tensor<float> output({B, C}, Device::GPU);

    int total = B * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    global_avg_pool_kernel<<<blocks, threads>>>(
        input.data_ptr(), output.data_ptr(), B, C, H, W
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// ===========================================================================
// Kernel: Linear (Fully Connected) Forward
// ===========================================================================
// Computes y = x @ W^T + b  (or y = x @ W^T if no bias)
//
//   input:  (B, in_features)
//   weight: (out_features, in_features)
//   bias:   (out_features,) or nullptr
//   output: (B, out_features)
//
//   output[b][j] = SUM_{k=0}^{in_features-1} input[b][k] * weight[j][k]
//                  + bias[j]
//
// We use a simple tiled approach. Each thread computes one element of
// the output matrix. For our small ResNet sizes (C=16..256, fc=8..64),
// this is adequate. For larger sizes, use cuBLAS.
//
// Thread mapping: one thread per output element (b, j)
// ===========================================================================

__global__ void linear_forward_kernel(
    const float* __restrict__ input,    // (B, K) where K = in_features
    const float* __restrict__ weight,   // (J, K) where J = out_features
    const float* __restrict__ bias,     // (J,) or nullptr
    float* __restrict__ output,         // (B, J)
    int B, int K, int J                 // dimensions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * J;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int b = i / J;
        int j = i % J;

        // Dot product: input[b][:] . weight[j][:]
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += input[b * K + k] * weight[j * K + k];
        }

        // Add bias if present
        if (bias != nullptr) {
            sum += bias[j];
        }

        output[i] = sum;
    }
}

// ===========================================================================
// Host Wrapper: linear_forward
// ===========================================================================
// input:  (B, in_features) on GPU
// weight: (out_features, in_features) on GPU
// bias:   (out_features,) on GPU, or empty Tensor for no bias
//
// Returns: (B, out_features) on GPU
// ===========================================================================

Tensor<float> linear_forward(
    const Tensor<float>& input,
    const Tensor<float>& weight,
    const Tensor<float>& bias
) {
    int B = input.shape_[0];
    int K = input.shape_[1];   // in_features
    int J = weight.shape_[0];  // out_features

    Tensor<float> output({B, J}, Device::GPU);

    int total = B * J;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* bias_ptr = (bias.size_ > 0) ? bias.data_ptr() : nullptr;

    linear_forward_kernel<<<blocks, threads>>>(
        input.data_ptr(), weight.data_ptr(), bias_ptr,
        output.data_ptr(), B, K, J
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

// ===========================================================================
// Kernel: Cross-Entropy Loss Forward
// ===========================================================================
// Combines softmax and negative log-likelihood in a numerically stable way.
//
// For each sample b in the batch:
//
//   1. Find max logit for numerical stability:
//      m = max_j(logits[b][j])
//
//   2. Compute log-sum-exp:
//      lse = m + log(SUM_j exp(logits[b][j] - m))
//
//   3. Log-softmax for the target class:
//      log_softmax[b][target[b]] = logits[b][target[b]] - lse
//
//   4. Loss for this sample:
//      loss[b] = -log_softmax[b][target[b]]
//              = lse - logits[b][target[b]]
//
// Final loss = mean over batch: (1/B) * SUM_b loss[b]
//
// Thread mapping: one thread per batch element (each thread handles
// the full class dimension for its sample).
// ===========================================================================

__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ logits,   // (B, num_classes)
    const int* __restrict__ targets,    // (B,) integer class labels
    float* __restrict__ losses,         // (B,) per-sample losses
    int B, int num_classes
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* row = logits + b * num_classes;

    // Step 1: Find max logit for numerical stability
    float max_val = -FLT_MAX;
    for (int j = 0; j < num_classes; j++) {
        if (row[j] > max_val) max_val = row[j];
    }

    // Step 2: Compute log-sum-exp = max + log(sum(exp(x - max)))
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++) {
        sum_exp += expf(row[j] - max_val);
    }
    float log_sum_exp = max_val + logf(sum_exp);

    // Step 3: Loss = log_sum_exp - logit[target]
    int target = targets[b];
    losses[b] = log_sum_exp - row[target];
}

// ===========================================================================
// Host Wrapper: cross_entropy_loss
// ===========================================================================
// logits:  (B, num_classes) on GPU
// targets: (B,) int tensor on GPU (class labels, 0-indexed)
//
// Returns: scalar loss (float) -- mean over batch
// ===========================================================================

float cross_entropy_loss(
    const Tensor<float>& logits,
    const int* d_targets,   // device pointer to integer targets
    int B, int num_classes
) {
    // Allocate per-sample losses
    Tensor<float> losses({B}, Device::GPU);

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    cross_entropy_loss_kernel<<<blocks, threads>>>(
        logits.data_ptr(), d_targets, losses.data_ptr(), B, num_classes
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Sum losses on CPU (B is small)
    Tensor<float> losses_cpu = losses.to_cpu();
    float total_loss = 0.0f;
    for (int b = 0; b < B; b++) {
        total_loss += losses_cpu.data_ptr()[b];
    }

    return total_loss / (float)B;
}

// ===========================================================================
// Test: All Operations
// ===========================================================================

int main() {
    printf("=== Chapter 14: Linear, ReLU, GAP, Cross-Entropy Tests ===\n\n");

    // -----------------------------------------------------------------
    // Test 1: ReLU
    // -----------------------------------------------------------------
    {
        printf("Test 1: ReLU Forward\n");

        float data[] = {-3, -2, -1, 0, 1, 2, 3, 4};
        float expected[] = {0, 0, 0, 0, 1, 2, 3, 4};

        Tensor<float> input({1, 8}, data, Device::GPU);
        Tensor<float> output = relu_forward(input);
        Tensor<float> out_cpu = output.to_cpu();

        bool pass = true;
        for (int i = 0; i < 8; i++) {
            if (fabsf(out_cpu.data_ptr()[i] - expected[i]) > 1e-6) {
                pass = false;
                break;
            }
        }
        printf("  Input:    [-3, -2, -1, 0, 1, 2, 3, 4]\n");
        printf("  Output:   [%.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f, %.0f]\n",
               out_cpu.data_ptr()[0], out_cpu.data_ptr()[1],
               out_cpu.data_ptr()[2], out_cpu.data_ptr()[3],
               out_cpu.data_ptr()[4], out_cpu.data_ptr()[5],
               out_cpu.data_ptr()[6], out_cpu.data_ptr()[7]);
        printf("  Expected: [0, 0, 0, 0, 1, 2, 3, 4] -- %s\n",
               pass ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 2: Global Average Pooling
    // -----------------------------------------------------------------
    {
        printf("\nTest 2: Global Average Pooling\n");

        // Input: (1, 2, 2, 2)
        // Channel 0: [1, 2, 3, 4] -> mean = 2.5
        // Channel 1: [5, 6, 7, 8] -> mean = 6.5
        float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
        Tensor<float> input({1, 2, 2, 2}, data, Device::GPU);

        Tensor<float> output = global_avg_pool_forward(input);
        Tensor<float> out_cpu = output.to_cpu();

        printf("  Input shape: (1, 2, 2, 2)\n");
        printf("  Output shape: (%d, %d)\n",
               output.shape_[0], output.shape_[1]);
        printf("  Channel 0 mean: %.4f (expected: 2.5) -- %s\n",
               out_cpu.data_ptr()[0],
               fabsf(out_cpu.data_ptr()[0] - 2.5f) < 1e-5 ? "PASS" : "FAIL");
        printf("  Channel 1 mean: %.4f (expected: 6.5) -- %s\n",
               out_cpu.data_ptr()[1],
               fabsf(out_cpu.data_ptr()[1] - 6.5f) < 1e-5 ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 3: Linear Forward
    // -----------------------------------------------------------------
    {
        printf("\nTest 3: Linear Forward (y = x @ W^T + b)\n");

        // Input: (2, 3) -- 2 samples, 3 features
        float x_data[] = {1, 2, 3,
                          4, 5, 6};

        // Weight: (2, 3) -- 2 output features, 3 input features
        float w_data[] = {1, 0, 0,   // extracts feature 0
                          0, 1, 0};  // extracts feature 1

        // Bias: (2,)
        float b_data[] = {10, 20};

        Tensor<float> input({2, 3}, x_data, Device::GPU);
        Tensor<float> weight({2, 3}, w_data, Device::GPU);
        Tensor<float> bias({2}, b_data, Device::GPU);

        Tensor<float> output = linear_forward(input, weight, bias);
        Tensor<float> out_cpu = output.to_cpu();

        // Expected: sample 0: [1+10, 2+20] = [11, 22]
        //           sample 1: [4+10, 5+20] = [14, 25]
        printf("  Output shape: (%d, %d)\n",
               output.shape_[0], output.shape_[1]);
        printf("  Sample 0: [%.1f, %.1f] (expected: [11.0, 22.0]) -- %s\n",
               out_cpu(0, 0), out_cpu(0, 1),
               (fabsf(out_cpu(0, 0) - 11.0f) < 1e-5 &&
                fabsf(out_cpu(0, 1) - 22.0f) < 1e-5) ? "PASS" : "FAIL");
        printf("  Sample 1: [%.1f, %.1f] (expected: [14.0, 25.0]) -- %s\n",
               out_cpu(1, 0), out_cpu(1, 1),
               (fabsf(out_cpu(1, 0) - 14.0f) < 1e-5 &&
                fabsf(out_cpu(1, 1) - 25.0f) < 1e-5) ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 4: Linear Forward without bias
    // -----------------------------------------------------------------
    {
        printf("\nTest 4: Linear Forward (no bias)\n");

        float x_data[] = {1, 2, 3};
        float w_data[] = {1, 1, 1,   // sum of features
                          2, 0, 0};  // 2x feature 0

        Tensor<float> input({1, 3}, x_data, Device::GPU);
        Tensor<float> weight({2, 3}, w_data, Device::GPU);
        Tensor<float> no_bias;  // empty tensor, size=0

        Tensor<float> output = linear_forward(input, weight, no_bias);
        Tensor<float> out_cpu = output.to_cpu();

        // Expected: [1+2+3, 2*1] = [6, 2]
        printf("  Output: [%.1f, %.1f] (expected: [6.0, 2.0]) -- %s\n",
               out_cpu(0, 0), out_cpu(0, 1),
               (fabsf(out_cpu(0, 0) - 6.0f) < 1e-5 &&
                fabsf(out_cpu(0, 1) - 2.0f) < 1e-5) ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 5: Cross-Entropy Loss
    // -----------------------------------------------------------------
    {
        printf("\nTest 5: Cross-Entropy Loss\n");

        // Logits: (2, 3) -- 2 samples, 3 classes
        // Sample 0: [2.0, 1.0, 0.1] -> target=0 (correct class has highest logit)
        // Sample 1: [0.1, 0.2, 3.0] -> target=2 (correct class has highest logit)
        float logits_data[] = {2.0f, 1.0f, 0.1f,
                               0.1f, 0.2f, 3.0f};
        int targets_data[] = {0, 2};

        Tensor<float> logits({2, 3}, logits_data, Device::GPU);

        // Copy targets to GPU
        int* d_targets;
        CUDA_CHECK(cudaMalloc(&d_targets, 2 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_targets, targets_data, 2 * sizeof(int),
                              cudaMemcpyHostToDevice));

        float loss = cross_entropy_loss(logits, d_targets, 2, 3);

        // Manual computation for sample 0:
        //   max = 2.0
        //   sum_exp = exp(0) + exp(-1) + exp(-1.9) = 1 + 0.3679 + 0.1496 = 1.5175
        //   log_sum_exp = 2.0 + log(1.5175) = 2.0 + 0.4170 = 2.4170
        //   loss_0 = 2.4170 - 2.0 = 0.4170
        //
        // For sample 1:
        //   max = 3.0
        //   sum_exp = exp(-2.9) + exp(-2.8) + exp(0) = 0.0550 + 0.0608 + 1 = 1.1158
        //   log_sum_exp = 3.0 + log(1.1158) = 3.0 + 0.1095 = 3.1095
        //   loss_1 = 3.1095 - 3.0 = 0.1095
        //
        // Mean loss = (0.4170 + 0.1095) / 2 = 0.2633

        // Compute expected on CPU
        float expected_loss = 0.0f;
        for (int b = 0; b < 2; b++) {
            float max_v = -FLT_MAX;
            for (int j = 0; j < 3; j++) {
                if (logits_data[b * 3 + j] > max_v) max_v = logits_data[b * 3 + j];
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < 3; j++) {
                sum_exp += expf(logits_data[b * 3 + j] - max_v);
            }
            float lse = max_v + logf(sum_exp);
            expected_loss += lse - logits_data[b * 3 + targets_data[b]];
        }
        expected_loss /= 2.0f;

        printf("  Loss:     %.6f\n", loss);
        printf("  Expected: %.6f\n", expected_loss);
        printf("  Error:    %.6e -- %s\n",
               fabsf(loss - expected_loss),
               fabsf(loss - expected_loss) < 1e-5 ? "PASS" : "FAIL");

        CUDA_CHECK(cudaFree(d_targets));
    }

    // -----------------------------------------------------------------
    // Test 6: GAP with larger tensor (ResNet-like shape)
    // -----------------------------------------------------------------
    {
        printf("\nTest 6: Global Average Pooling (ResNet shape)\n");

        int B = 2, C = 16, H = 8, W = 8;
        Tensor<float> input = Tensor<float>::ones({B, C, H, W}, Device::GPU);

        Tensor<float> output = global_avg_pool_forward(input);

        printf("  Input shape:  (%d, %d, %d, %d)\n", B, C, H, W);
        printf("  Output shape: (%d, %d)\n",
               output.shape_[0], output.shape_[1]);

        Tensor<float> out_cpu = output.to_cpu();
        // All ones averaged over 8x8 = 64 elements -> mean = 1.0
        bool pass = true;
        for (int i = 0; i < B * C; i++) {
            if (fabsf(out_cpu.data_ptr()[i] - 1.0f) > 1e-5) {
                pass = false;
                break;
            }
        }
        printf("  All values should be 1.0: %s\n", pass ? "PASS" : "FAIL");
    }

    printf("\n=== Linear, ReLU, GAP, Cross-Entropy Tests Complete ===\n");
    return 0;
}
