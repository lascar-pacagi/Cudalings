// ===========================================================================
// Chapter 14: forward_test.cu -- Integration Test: Full ResNet Forward Pass
// ===========================================================================
//
// This file chains together all forward-pass operations to simulate a
// single forward pass through a ResNet-style network:
//
//   Input: (2, 4, 8, 8)                     -- 2 boards, 4 channels, 8x8
//   Stem:  Conv2d(4->16, 3x3, pad=1)        -- (2, 16, 8, 8)
//          BatchNorm2d(16) + ReLU            -- (2, 16, 8, 8)
//
//   ResBlock: (pre-activation style)
//     identity = x
//     BN(16) -> ReLU -> Conv(16->16, 3x3, pad=1) ->
//     BN(16) -> ReLU -> Conv(16->16, 3x3, pad=1)
//     output = x + identity                  -- (2, 16, 8, 8)
//
//   Head:
//     BN(16) -> ReLU -> GlobalAvgPool        -- (2, 16)
//     Linear(16->8) -> ReLU                  -- (2, 8)
//     Linear(8->3)                           -- (2, 3)
//
//   Loss:
//     CrossEntropyLoss(logits, targets)      -- scalar
//
// We verify shapes at each step and print the final logits.
//
// ===========================================================================

#include "../13_tensor_class/tensor.cuh"
#include <cstdio>
#include <cmath>
#include <cfloat>

// ===========================================================================
// Forward declarations of kernels and wrappers from other files.
// We re-declare the kernels inline here to keep this file self-contained
// (avoids linker issues with separate compilation of template code).
// ===========================================================================

// ---- Conv2D Forward ----

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
        B, IC, IH, IW, OC, OH, OW, KH, KW, stride, pad
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}

// ---- BatchNorm2D Forward ----

__global__ void compute_channel_mean_var(
    const float* __restrict__ input,
    float* __restrict__ mean_out,
    float* __restrict__ var_out,
    int B, int C, int H, int W
) {
    int c = blockIdx.x;
    if (c >= C) return;
    int N = B * H * W;
    int HW = H * W;
    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int b = i / HW;
        int spatial = i % HW;
        local_sum += input[b * C * HW + c * HW + spatial];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float channel_mean = sdata[0] / (float)N;
    if (threadIdx.x == 0) mean_out[c] = channel_mean;
    __syncthreads();
    if (threadIdx.x == 0) sdata[0] = channel_mean;
    __syncthreads();
    channel_mean = sdata[0];

    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int b = i / HW;
        int spatial = i % HW;
        float diff = input[b * C * HW + c * HW + spatial] - channel_mean;
        local_var_sum += diff * diff;
    }
    sdata[threadIdx.x] = local_var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) var_out[c] = sdata[0] / (float)N;
}

__global__ void batchnorm_normalize(
    const float* __restrict__ input, float* __restrict__ output,
    const float* __restrict__ mean, const float* __restrict__ var,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    int B, int C, int H, int W, float eps
) {
    int total = B * C * H * W;
    int HW = H * W;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += blockDim.x * gridDim.x)
    {
        int c = (idx / HW) % C;
        float x = input[idx];
        float x_hat = (x - mean[c]) / sqrtf(var[c] + eps);
        output[idx] = gamma[c] * x_hat + beta[c];
    }
}

__global__ void update_running_stats(
    float* __restrict__ running_mean, float* __restrict__ running_var,
    const float* __restrict__ batch_mean, const float* __restrict__ batch_var,
    int C, float momentum
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * batch_mean[c];
        running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * batch_var[c];
    }
}

Tensor<float> batchnorm2d_forward(
    const Tensor<float>& input, const Tensor<float>& gamma,
    const Tensor<float>& beta, Tensor<float>& running_mean,
    Tensor<float>& running_var, bool training, float momentum, float eps
) {
    int B = input.shape_[0], C = input.shape_[1];
    int H = input.shape_[2], W = input.shape_[3];
    Tensor<float> output({B, C, H, W}, Device::GPU);

    if (training) {
        Tensor<float> batch_mean({C}, Device::GPU);
        Tensor<float> batch_var({C}, Device::GPU);
        int N = B * H * W;
        int tpb = 1;
        while (tpb * 2 <= N && tpb * 2 <= 1024) tpb *= 2;
        compute_channel_mean_var<<<C, tpb, tpb * sizeof(float)>>>(
            input.data_ptr(), batch_mean.data_ptr(), batch_var.data_ptr(),
            B, C, H, W);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int total = B * C * H * W;
        batchnorm_normalize<<<(total + 255) / 256, 256>>>(
            input.data_ptr(), output.data_ptr(),
            batch_mean.data_ptr(), batch_var.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(), B, C, H, W, eps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        update_running_stats<<<(C + 255) / 256, 256>>>(
            running_mean.data_ptr(), running_var.data_ptr(),
            batch_mean.data_ptr(), batch_var.data_ptr(), C, momentum);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        int total = B * C * H * W;
        batchnorm_normalize<<<(total + 255) / 256, 256>>>(
            input.data_ptr(), output.data_ptr(),
            running_mean.data_ptr(), running_var.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(), B, C, H, W, eps);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return output;
}

// ---- ReLU Forward ----

__global__ void relu_forward_kernel(
    const float* __restrict__ input, float* __restrict__ output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

Tensor<float> relu_forward(const Tensor<float>& input) {
    Tensor<float> output(input.shape_, Device::GPU);
    int threads = 256, blocks = (input.size_ + 255) / 256;
    relu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr(), output.data_ptr(), input.size_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}

// ---- Global Average Pooling ----

__global__ void global_avg_pool_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int B, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    int HW = H * W;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int b = i / C, c = i % C;
        float sum = 0.0f;
        int base = (b * C + c) * HW;
        for (int s = 0; s < HW; s++) sum += input[base + s];
        output[i] = sum / (float)HW;
    }
}

Tensor<float> global_avg_pool_forward(const Tensor<float>& input) {
    int B = input.shape_[0], C = input.shape_[1];
    int H = input.shape_[2], W = input.shape_[3];
    Tensor<float> output({B, C}, Device::GPU);
    int total = B * C;
    global_avg_pool_kernel<<<(total + 255) / 256, 256>>>(
        input.data_ptr(), output.data_ptr(), B, C, H, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}

// ---- Linear Forward ----

__global__ void linear_forward_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int B, int K, int J
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * J;
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int b = i / J, j = i % J;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += input[b * K + k] * weight[j * K + k];
        }
        if (bias != nullptr) sum += bias[j];
        output[i] = sum;
    }
}

Tensor<float> linear_forward(
    const Tensor<float>& input, const Tensor<float>& weight,
    const Tensor<float>& bias
) {
    int B = input.shape_[0], K = input.shape_[1], J = weight.shape_[0];
    Tensor<float> output({B, J}, Device::GPU);
    int total = B * J;
    const float* bias_ptr = (bias.size_ > 0) ? bias.data_ptr() : nullptr;
    linear_forward_kernel<<<(total + 255) / 256, 256>>>(
        input.data_ptr(), weight.data_ptr(), bias_ptr,
        output.data_ptr(), B, K, J);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}

// ---- Cross-Entropy Loss ----

__global__ void cross_entropy_loss_kernel(
    const float* __restrict__ logits, const int* __restrict__ targets,
    float* __restrict__ losses, int B, int num_classes
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float* row = logits + b * num_classes;
    float max_val = -FLT_MAX;
    for (int j = 0; j < num_classes; j++)
        if (row[j] > max_val) max_val = row[j];
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; j++)
        sum_exp += expf(row[j] - max_val);
    float lse = max_val + logf(sum_exp);
    losses[b] = lse - row[targets[b]];
}

float cross_entropy_loss(
    const Tensor<float>& logits, const int* d_targets,
    int B, int num_classes
) {
    Tensor<float> losses({B}, Device::GPU);
    cross_entropy_loss_kernel<<<(B + 255) / 256, 256>>>(
        logits.data_ptr(), d_targets, losses.data_ptr(), B, num_classes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    Tensor<float> losses_cpu = losses.to_cpu();
    float total = 0.0f;
    for (int b = 0; b < B; b++) total += losses_cpu.data_ptr()[b];
    return total / (float)B;
}

// ===========================================================================
// Element-wise add kernel for skip connections
// ===========================================================================

__global__ void elementwise_add_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        out[i] = a[i] + b[i];
    }
}

Tensor<float> elementwise_add(const Tensor<float>& a, const Tensor<float>& b) {
    Tensor<float> output(a.shape_, Device::GPU);
    int threads = 256, blocks = (a.size_ + 255) / 256;
    elementwise_add_kernel<<<blocks, threads>>>(
        a.data_ptr(), b.data_ptr(), output.data_ptr(), a.size_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return output;
}

// ===========================================================================
// Helper: print shape
// ===========================================================================

void print_shape(const char* label, const Tensor<float>& t) {
    printf("  %-30s shape: (", label);
    for (int i = 0; i < t.ndim(); i++) {
        if (i > 0) printf(", ");
        printf("%d", t.shape_[i]);
    }
    printf(")\n");
}

// ===========================================================================
// Kaiming/He initialization for conv weights
// ===========================================================================
// std = sqrt(2 / fan_in) where fan_in = IC * KH * KW
// We scale randn by this factor.
// ===========================================================================

Tensor<float> kaiming_init(const std::vector<int>& shape, Device device) {
    Tensor<float> w = Tensor<float>::randn(shape, device);

    // fan_in = product of shape[1:]
    int fan_in = 1;
    for (int i = 1; i < (int)shape.size(); i++) fan_in *= shape[i];

    float scale = sqrtf(2.0f / (float)fan_in);

    // Scale on CPU then transfer, or scale on GPU
    if (device == Device::CPU) {
        for (int i = 0; i < w.size_; i++) w.data_ptr()[i] *= scale;
    } else {
        // Scale via a simple kernel
        Tensor<float> w_cpu = w.to_cpu();
        for (int i = 0; i < w_cpu.size_; i++) w_cpu.data_ptr()[i] *= scale;
        w = w_cpu.to_gpu();
    }
    return w;
}

// ===========================================================================
// Main: Full ResNet Forward Pass Integration Test
// ===========================================================================

int main() {
    printf("=== Chapter 14: Full ResNet Forward Pass Integration Test ===\n\n");

    // ---- Network hyperparameters ----
    int B  = 2;    // batch size
    int IC = 4;    // input channels (board representation)
    int H  = 8;    // board height
    int W  = 8;    // board width
    int C  = 16;   // hidden channels
    int fc = 8;    // fully connected hidden size
    int num_classes = 3;  // output classes (win/draw/loss)

    float eps = 1e-5f;
    float momentum = 0.1f;

    printf("Network configuration:\n");
    printf("  Input:       (%d, %d, %d, %d)\n", B, IC, H, W);
    printf("  Channels:    %d\n", C);
    printf("  FC hidden:   %d\n", fc);
    printf("  Classes:     %d\n\n", num_classes);

    // ==================================================================
    // Initialize all weights and parameters
    // ==================================================================

    printf("--- Initializing parameters ---\n");

    // Stem conv: (C, IC, 3, 3)
    Tensor<float> stem_conv_w = kaiming_init({C, IC, 3, 3}, Device::GPU);

    // Stem BN: gamma=1, beta=0
    Tensor<float> stem_bn_gamma = Tensor<float>::ones({C}, Device::GPU);
    Tensor<float> stem_bn_beta  = Tensor<float>::zeros({C}, Device::GPU);
    Tensor<float> stem_bn_rmean = Tensor<float>::zeros({C}, Device::GPU);
    Tensor<float> stem_bn_rvar  = Tensor<float>::ones({C}, Device::GPU);

    // ResBlock: 2 conv layers + 2 BN layers
    Tensor<float> rb_conv1_w = kaiming_init({C, C, 3, 3}, Device::GPU);
    Tensor<float> rb_bn1_gamma = Tensor<float>::ones({C}, Device::GPU);
    Tensor<float> rb_bn1_beta  = Tensor<float>::zeros({C}, Device::GPU);
    Tensor<float> rb_bn1_rmean = Tensor<float>::zeros({C}, Device::GPU);
    Tensor<float> rb_bn1_rvar  = Tensor<float>::ones({C}, Device::GPU);

    Tensor<float> rb_conv2_w = kaiming_init({C, C, 3, 3}, Device::GPU);
    Tensor<float> rb_bn2_gamma = Tensor<float>::ones({C}, Device::GPU);
    Tensor<float> rb_bn2_beta  = Tensor<float>::zeros({C}, Device::GPU);
    Tensor<float> rb_bn2_rmean = Tensor<float>::zeros({C}, Device::GPU);
    Tensor<float> rb_bn2_rvar  = Tensor<float>::ones({C}, Device::GPU);

    // Final BN (before head)
    Tensor<float> final_bn_gamma = Tensor<float>::ones({C}, Device::GPU);
    Tensor<float> final_bn_beta  = Tensor<float>::zeros({C}, Device::GPU);
    Tensor<float> final_bn_rmean = Tensor<float>::zeros({C}, Device::GPU);
    Tensor<float> final_bn_rvar  = Tensor<float>::ones({C}, Device::GPU);

    // Linear layers
    Tensor<float> fc1_w = kaiming_init({fc, C}, Device::GPU);
    Tensor<float> fc1_b = Tensor<float>::zeros({fc}, Device::GPU);
    Tensor<float> fc2_w = kaiming_init({num_classes, fc}, Device::GPU);
    Tensor<float> fc2_b = Tensor<float>::zeros({num_classes}, Device::GPU);

    printf("  All parameters initialized.\n\n");

    // ==================================================================
    // Create input tensor (random board states)
    // ==================================================================

    Tensor<float> x = Tensor<float>::randn({B, IC, H, W}, Device::GPU);
    printf("--- Forward Pass ---\n");
    print_shape("Input", x);

    // ==================================================================
    // Stem: Conv(4->16, 3x3, pad=1) -> BN -> ReLU
    // ==================================================================

    x = conv2d_forward(x, stem_conv_w, 1, 1);
    print_shape("After stem conv", x);

    x = batchnorm2d_forward(x, stem_bn_gamma, stem_bn_beta,
                             stem_bn_rmean, stem_bn_rvar,
                             true, momentum, eps);
    print_shape("After stem BN", x);

    x = relu_forward(x);
    print_shape("After stem ReLU", x);

    // ==================================================================
    // ResBlock (pre-activation: BN -> ReLU -> Conv -> BN -> ReLU -> Conv + skip)
    // ==================================================================

    printf("\n  --- ResBlock ---\n");

    // Save identity for skip connection
    // (We need a copy because x will be modified through the block)
    Tensor<float> identity = x.to_gpu();  // deep copy

    // BN1 -> ReLU -> Conv1
    x = batchnorm2d_forward(x, rb_bn1_gamma, rb_bn1_beta,
                             rb_bn1_rmean, rb_bn1_rvar,
                             true, momentum, eps);
    print_shape("  After RB BN1", x);

    x = relu_forward(x);
    print_shape("  After RB ReLU1", x);

    x = conv2d_forward(x, rb_conv1_w, 1, 1);
    print_shape("  After RB Conv1", x);

    // BN2 -> ReLU -> Conv2
    x = batchnorm2d_forward(x, rb_bn2_gamma, rb_bn2_beta,
                             rb_bn2_rmean, rb_bn2_rvar,
                             true, momentum, eps);
    print_shape("  After RB BN2", x);

    x = relu_forward(x);
    print_shape("  After RB ReLU2", x);

    x = conv2d_forward(x, rb_conv2_w, 1, 1);
    print_shape("  After RB Conv2", x);

    // Skip connection: x = x + identity
    x = elementwise_add(x, identity);
    print_shape("  After skip add", x);

    // ==================================================================
    // Head: Final BN -> ReLU -> GlobalAvgPool -> FC1 -> ReLU -> FC2
    // ==================================================================

    printf("\n  --- Head ---\n");

    x = batchnorm2d_forward(x, final_bn_gamma, final_bn_beta,
                             final_bn_rmean, final_bn_rvar,
                             true, momentum, eps);
    print_shape("  After final BN", x);

    x = relu_forward(x);
    print_shape("  After final ReLU", x);

    x = global_avg_pool_forward(x);
    print_shape("  After GAP", x);

    x = linear_forward(x, fc1_w, fc1_b);
    print_shape("  After FC1", x);

    x = relu_forward(x);
    print_shape("  After FC1 ReLU", x);

    x = linear_forward(x, fc2_w, fc2_b);
    print_shape("  After FC2 (logits)", x);

    // ==================================================================
    // Print output logits
    // ==================================================================

    Tensor<float> logits_cpu = x.to_cpu();
    printf("\n--- Output Logits ---\n");
    for (int b = 0; b < B; b++) {
        printf("  Sample %d: [", b);
        for (int j = 0; j < num_classes; j++) {
            if (j > 0) printf(", ");
            printf("%.4f", logits_cpu(b, j));
        }
        printf("]\n");
    }

    // ==================================================================
    // Compute Cross-Entropy Loss
    // ==================================================================

    // Dummy targets: sample 0 -> class 0, sample 1 -> class 2
    int targets_data[] = {0, 2};
    int* d_targets;
    CUDA_CHECK(cudaMalloc(&d_targets, B * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_targets, targets_data, B * sizeof(int),
                          cudaMemcpyHostToDevice));

    float loss = cross_entropy_loss(x, d_targets, B, num_classes);
    printf("\n--- Loss ---\n");
    printf("  Targets: [%d, %d]\n", targets_data[0], targets_data[1]);
    printf("  Cross-Entropy Loss: %.6f\n", loss);

    CUDA_CHECK(cudaFree(d_targets));

    // ==================================================================
    // Shape verification summary
    // ==================================================================

    printf("\n--- Shape Verification Summary ---\n");
    printf("  Expected flow:\n");
    printf("    (2,4,8,8) -> Conv -> (2,16,8,8) -> BN -> ReLU\n");
    printf("    -> ResBlock [BN->ReLU->Conv->BN->ReLU->Conv + skip] -> (2,16,8,8)\n");
    printf("    -> BN -> ReLU -> GAP -> (2,16)\n");
    printf("    -> FC1 -> (2,8) -> ReLU -> FC2 -> (2,3)\n");
    printf("    -> CrossEntropyLoss -> scalar\n");

    bool shape_ok = (logits_cpu.shape_[0] == B &&
                     logits_cpu.shape_[1] == num_classes);
    printf("  Final logits shape (%d, %d): %s\n",
           logits_cpu.shape_[0], logits_cpu.shape_[1],
           shape_ok ? "CORRECT" : "WRONG");

    printf("\n=== Integration Test Complete ===\n");
    return 0;
}
