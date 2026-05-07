// ===========================================================================
// Chapter 14: batchnorm_forward.cu -- BatchNorm2D Forward Pass
// ===========================================================================
//
// Implements Batch Normalization for 2D convolutional layers (NCHW format).
//
// BatchNorm normalizes each channel independently across the batch and
// spatial dimensions. For input shape (B, C, H, W), each channel c has
// B*H*W elements that are normalized together.
//
// The forward pass consists of two stages:
//
//   Stage 1: Compute per-channel mean and variance (parallel reduction)
//
//     mean[c] = (1 / N) * SUM_{b,h,w} x[b][c][h][w]     where N = B*H*W
//     var[c]  = (1 / N) * SUM_{b,h,w} (x[b][c][h][w] - mean[c])^2
//
//   Stage 2: Normalize and apply affine transform
//
//     x_hat[b][c][h][w] = (x[b][c][h][w] - mean[c]) / sqrt(var[c] + eps)
//     y[b][c][h][w]     = gamma[c] * x_hat[b][c][h][w] + beta[c]
//
// During training, we also update running statistics:
//     running_mean = (1 - momentum) * running_mean + momentum * mean
//     running_var  = (1 - momentum) * running_var  + momentum * var
//
// During inference, we use running_mean/running_var instead of computing
// batch statistics.
//
// ===========================================================================

#include "../13_tensor_class/tensor.cuh"
#include <cstdio>
#include <cmath>

// ===========================================================================
// Kernel 1: compute_channel_mean_var
// ===========================================================================
// Each block handles one channel. We use a parallel reduction within the
// block to sum all B*H*W elements for that channel.
//
// Two-pass approach within one kernel:
//   Pass 1: Compute mean = sum(x) / N
//   Pass 2: Compute var = sum((x - mean)^2) / N
//
// We use shared memory for the reduction.
//
// Grid: (C, 1, 1) -- one block per channel
// Block: (min(N, 1024), 1, 1)
//
// Outputs: mean[C], var[C]
// ===========================================================================

__global__ void compute_channel_mean_var(
    const float* __restrict__ input,   // (B, C, H, W) in NCHW
    float* __restrict__ mean_out,      // (C,) per-channel mean
    float* __restrict__ var_out,       // (C,) per-channel variance
    int B, int C, int H, int W
) {
    // Which channel does this block handle?
    int c = blockIdx.x;
    if (c >= C) return;

    // Number of elements per channel: B * H * W
    int N = B * H * W;
    int HW = H * W;

    // Shared memory for block-level reduction
    extern __shared__ float sdata[];

    // -----------------------------------------------------------------
    // Pass 1: Compute sum of all elements in this channel
    // -----------------------------------------------------------------
    // Each thread accumulates multiple elements (stride = blockDim.x)
    // Then we do a shared-memory reduction to get the total sum.
    // -----------------------------------------------------------------
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        // Convert flat index i within channel c to (b, h, w)
        //   i = b * HW + h * W + w
        // Input index in NCHW: b * C * HW + c * HW + h * W + w
        //                    = (i / HW) * C * HW + c * HW + (i % HW)
        int b = i / HW;
        int spatial = i % HW;
        int input_idx = b * C * HW + c * HW + spatial;
        local_sum += input[input_idx];
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 computes the mean
    float channel_mean = sdata[0] / (float)N;
    if (threadIdx.x == 0) {
        mean_out[c] = channel_mean;
    }
    __syncthreads();

    // Broadcast the mean to all threads via shared memory
    if (threadIdx.x == 0) {
        sdata[0] = channel_mean;
    }
    __syncthreads();
    channel_mean = sdata[0];

    // -----------------------------------------------------------------
    // Pass 2: Compute variance = sum((x - mean)^2) / N
    // -----------------------------------------------------------------
    float local_var_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int b = i / HW;
        int spatial = i % HW;
        int input_idx = b * C * HW + c * HW + spatial;
        float diff = input[input_idx] - channel_mean;
        local_var_sum += diff * diff;
    }

    sdata[threadIdx.x] = local_var_sum;
    __syncthreads();

    // Tree reduction for variance
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        var_out[c] = sdata[0] / (float)N;
    }
}

// ===========================================================================
// Kernel 2: batchnorm_normalize
// ===========================================================================
// Applies the normalization and affine transform element-wise:
//
//   y[b][c][h][w] = gamma[c] * (x[b][c][h][w] - mean[c])
//                   / sqrt(var[c] + eps) + beta[c]
//
// Each thread handles one element. Grid-stride loop for large tensors.
// ===========================================================================

__global__ void batchnorm_normalize(
    const float* __restrict__ input,    // (B, C, H, W)
    float* __restrict__ output,         // (B, C, H, W)
    const float* __restrict__ mean,     // (C,) -- per-channel mean
    const float* __restrict__ var,      // (C,) -- per-channel variance
    const float* __restrict__ gamma,    // (C,) -- scale parameter
    const float* __restrict__ beta,     // (C,) -- shift parameter
    int B, int C, int H, int W,
    float eps
) {
    int total = B * C * H * W;
    int HW = H * W;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        // Decompose idx -> (b, c, h, w) in NCHW layout
        //   idx = b * C * HW + c * HW + h * W + w
        int c = (idx / HW) % C;

        // Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
        float x = input[idx];
        float x_hat = (x - mean[c]) / sqrtf(var[c] + eps);
        output[idx] = gamma[c] * x_hat + beta[c];
    }
}

// ===========================================================================
// Kernel 3: update_running_stats
// ===========================================================================
// Updates the exponential moving average of mean and variance.
//   running_mean = (1 - momentum) * running_mean + momentum * batch_mean
//   running_var  = (1 - momentum) * running_var  + momentum * batch_var
//
// One thread per channel.
// ===========================================================================

__global__ void update_running_stats(
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    const float* __restrict__ batch_mean,
    const float* __restrict__ batch_var,
    int C, float momentum
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        running_mean[c] = (1.0f - momentum) * running_mean[c]
                        + momentum * batch_mean[c];
        running_var[c]  = (1.0f - momentum) * running_var[c]
                        + momentum * batch_var[c];
    }
}

// ===========================================================================
// Host Wrapper: batchnorm2d_forward
// ===========================================================================
// Parameters:
//   input:        (B, C, H, W) on GPU
//   gamma:        (C,) on GPU -- scale (initialized to 1)
//   beta:         (C,) on GPU -- shift (initialized to 0)
//   running_mean: (C,) on GPU -- running mean (updated in-place during training)
//   running_var:  (C,) on GPU -- running var  (updated in-place during training)
//   training:     true = compute batch stats, false = use running stats
//   momentum:     EMA momentum (default 0.1)
//   eps:          small constant for numerical stability (default 1e-5)
//
// Returns: output (B, C, H, W) on GPU
// ===========================================================================

Tensor<float> batchnorm2d_forward(
    const Tensor<float>& input,
    const Tensor<float>& gamma,
    const Tensor<float>& beta,
    Tensor<float>& running_mean,
    Tensor<float>& running_var,
    bool training = true,
    float momentum = 0.1f,
    float eps = 1e-5f
) {
    int B = input.shape_[0];
    int C = input.shape_[1];
    int H = input.shape_[2];
    int W = input.shape_[3];

    Tensor<float> output({B, C, H, W}, Device::GPU);

    if (training) {
        // -----------------------------------------------------------
        // Training mode: compute batch statistics
        // -----------------------------------------------------------

        // Allocate temporary buffers for batch mean and variance
        Tensor<float> batch_mean({C}, Device::GPU);
        Tensor<float> batch_var({C}, Device::GPU);

        // Kernel 1: Compute mean and variance per channel
        // One block per channel, up to 1024 threads per block
        int N = B * H * W;
        int threads_per_block = (N < 1024) ? N : 1024;
        // Round threads_per_block down to nearest power of 2 for reduction
        int t = 1;
        while (t * 2 <= threads_per_block) t *= 2;
        threads_per_block = t;

        int shared_mem = threads_per_block * sizeof(float);

        compute_channel_mean_var<<<C, threads_per_block, shared_mem>>>(
            input.data_ptr(),
            batch_mean.data_ptr(),
            batch_var.data_ptr(),
            B, C, H, W
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Kernel 2: Normalize using batch statistics
        int total = B * C * H * W;
        int norm_threads = 256;
        int norm_blocks = (total + norm_threads - 1) / norm_threads;

        batchnorm_normalize<<<norm_blocks, norm_threads>>>(
            input.data_ptr(), output.data_ptr(),
            batch_mean.data_ptr(), batch_var.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            B, C, H, W, eps
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Kernel 3: Update running statistics
        int stat_threads = 256;
        int stat_blocks = (C + stat_threads - 1) / stat_threads;

        update_running_stats<<<stat_blocks, stat_threads>>>(
            running_mean.data_ptr(), running_var.data_ptr(),
            batch_mean.data_ptr(), batch_var.data_ptr(),
            C, momentum
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

    } else {
        // -----------------------------------------------------------
        // Inference mode: use running statistics (no batch stats)
        // -----------------------------------------------------------
        int total = B * C * H * W;
        int norm_threads = 256;
        int norm_blocks = (total + norm_threads - 1) / norm_threads;

        batchnorm_normalize<<<norm_blocks, norm_threads>>>(
            input.data_ptr(), output.data_ptr(),
            running_mean.data_ptr(), running_var.data_ptr(),
            gamma.data_ptr(), beta.data_ptr(),
            B, C, H, W, eps
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    return output;
}

// ===========================================================================
// CPU Reference: batchnorm2d_forward_cpu
// ===========================================================================

void batchnorm2d_forward_cpu(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int B, int C, int H, int W, float eps
) {
    int HW = H * W;
    int N = B * HW;   // elements per channel

    for (int c = 0; c < C; c++) {
        // Compute mean for channel c
        float mean = 0.0f;
        for (int b = 0; b < B; b++) {
            for (int hw = 0; hw < HW; hw++) {
                mean += input[b * C * HW + c * HW + hw];
            }
        }
        mean /= (float)N;

        // Compute variance for channel c
        float var = 0.0f;
        for (int b = 0; b < B; b++) {
            for (int hw = 0; hw < HW; hw++) {
                float diff = input[b * C * HW + c * HW + hw] - mean;
                var += diff * diff;
            }
        }
        var /= (float)N;

        // Normalize and apply affine transform
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int b = 0; b < B; b++) {
            for (int hw = 0; hw < HW; hw++) {
                int idx = b * C * HW + c * HW + hw;
                float x_hat = (input[idx] - mean) * inv_std;
                output[idx] = gamma[c] * x_hat + beta[c];
            }
        }
    }
}

// ===========================================================================
// Test: BatchNorm2D Forward
// ===========================================================================

int main() {
    printf("=== Chapter 14: BatchNorm2D Forward Pass Test ===\n\n");

    // -----------------------------------------------------------------
    // Test 1: Known values -- identity transform
    // -----------------------------------------------------------------
    // With gamma=1, beta=0, the output should be the standardized input.
    // For a constant input, mean=constant, var=0, so output should be
    // all beta (0) since (x - mean) / sqrt(0 + eps) * 1 + 0 ~ 0.
    // -----------------------------------------------------------------
    {
        printf("Test 1: Constant input -> output should be all beta (0)\n");

        int B = 2, C = 3, H = 4, W = 4;
        // Constant input: all 5.0 for each channel
        Tensor<float> input = Tensor<float>::full({B, C, H, W}, 5.0f, Device::GPU);
        Tensor<float> gamma = Tensor<float>::ones({C}, Device::GPU);
        Tensor<float> beta  = Tensor<float>::zeros({C}, Device::GPU);
        Tensor<float> running_mean = Tensor<float>::zeros({C}, Device::GPU);
        Tensor<float> running_var  = Tensor<float>::ones({C}, Device::GPU);

        Tensor<float> output = batchnorm2d_forward(
            input, gamma, beta, running_mean, running_var,
            true, 0.1f, 1e-5f
        );

        Tensor<float> out_cpu = output.to_cpu();
        float max_val = 0.0f;
        for (int i = 0; i < out_cpu.size_; i++) {
            float v = fabsf(out_cpu.data_ptr()[i]);
            if (v > max_val) max_val = v;
        }
        printf("  Max abs value: %.6e (expected ~0) -- %s\n",
               max_val, max_val < 1e-3 ? "PASS" : "FAIL");

        // Check running_mean was updated
        Tensor<float> rm = running_mean.to_cpu();
        printf("  Running mean[0]: %.4f (expected: 0.9*0 + 0.1*5 = 0.5) -- %s\n",
               rm.data_ptr()[0],
               fabsf(rm.data_ptr()[0] - 0.5f) < 1e-3 ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 2: Compare GPU vs CPU reference
    // -----------------------------------------------------------------
    {
        printf("\nTest 2: GPU vs CPU reference comparison\n");

        int B = 2, C = 16, H = 8, W = 8;

        Tensor<float> input_cpu = Tensor<float>::randn({B, C, H, W}, Device::CPU);
        Tensor<float> gamma_cpu = Tensor<float>::ones({C}, Device::CPU);
        Tensor<float> beta_cpu  = Tensor<float>::zeros({C}, Device::CPU);

        // Modify gamma and beta to non-trivial values
        for (int c = 0; c < C; c++) {
            gamma_cpu.data_ptr()[c] = 1.0f + 0.1f * c;
            beta_cpu.data_ptr()[c]  = 0.05f * c;
        }

        // CPU reference
        std::vector<float> output_ref(B * C * H * W);
        batchnorm2d_forward_cpu(
            input_cpu.data_ptr(), output_ref.data(),
            gamma_cpu.data_ptr(), beta_cpu.data_ptr(),
            B, C, H, W, 1e-5f
        );

        // GPU forward
        Tensor<float> input_gpu = input_cpu.to_gpu();
        Tensor<float> gamma_gpu = gamma_cpu.to_gpu();
        Tensor<float> beta_gpu  = beta_cpu.to_gpu();
        Tensor<float> running_mean = Tensor<float>::zeros({C}, Device::GPU);
        Tensor<float> running_var  = Tensor<float>::ones({C}, Device::GPU);

        Tensor<float> output_gpu = batchnorm2d_forward(
            input_gpu, gamma_gpu, beta_gpu, running_mean, running_var,
            true, 0.1f, 1e-5f
        );
        Tensor<float> out_from_gpu = output_gpu.to_cpu();

        float max_err = 0.0f;
        for (int i = 0; i < B * C * H * W; i++) {
            float err = fabsf(out_from_gpu.data_ptr()[i] - output_ref[i]);
            if (err > max_err) max_err = err;
        }

        printf("  Max error vs CPU: %.6e -- %s\n",
               max_err, max_err < 1e-4 ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 3: Inference mode uses running stats
    // -----------------------------------------------------------------
    {
        printf("\nTest 3: Inference mode uses running statistics\n");

        int B = 1, C = 2, H = 2, W = 2;

        // Input: different per channel
        float input_data[] = {1, 2, 3, 4,   // channel 0
                              5, 6, 7, 8};  // channel 1
        Tensor<float> input({B, C, H, W}, input_data, Device::GPU);

        Tensor<float> gamma = Tensor<float>::ones({C}, Device::GPU);
        Tensor<float> beta  = Tensor<float>::zeros({C}, Device::GPU);

        // Set known running stats
        float rm_data[] = {2.5f, 6.5f};  // means that match batch means
        float rv_data[] = {1.25f, 1.25f}; // variances that match batch vars
        Tensor<float> running_mean({C}, rm_data, Device::GPU);
        Tensor<float> running_var({C}, rv_data, Device::GPU);

        // Inference mode: use running stats
        Tensor<float> output = batchnorm2d_forward(
            input, gamma, beta, running_mean, running_var,
            false,  // inference mode
            0.1f, 1e-5f
        );

        Tensor<float> out_cpu = output.to_cpu();

        // Expected: (x - running_mean) / sqrt(running_var + eps)
        // For channel 0, element 0: (1 - 2.5) / sqrt(1.25 + 1e-5) = -1.5 / 1.1180
        float expected_00 = (1.0f - 2.5f) / sqrtf(1.25f + 1e-5f);
        printf("  output[0][0][0][0]: %.4f (expected: %.4f) -- %s\n",
               out_cpu(0, 0, 0, 0), expected_00,
               fabsf(out_cpu(0, 0, 0, 0) - expected_00) < 1e-3 ? "PASS" : "FAIL");
    }

    printf("\n=== BatchNorm2D Forward Tests Complete ===\n");
    return 0;
}
