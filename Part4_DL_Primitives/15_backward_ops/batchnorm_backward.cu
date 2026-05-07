// ===========================================================================
// Chapter 15: batchnorm_backward.cu -- BatchNorm2D Backward Pass
// ===========================================================================
//
// Implements the backward pass for Batch Normalization (2D, NCHW format).
//
// This is the most mathematically complex backward pass in standard neural
// networks. The derivation involves multiple intermediate variables, each
// depending on channel-wide statistics (mean, variance).
//
// ===========================================================================
//
//  COMPLETE DERIVATION OF BATCHNORM BACKWARD
//  ==========================================
//
//  Forward pass (per channel c, N = B*H*W elements):
//
//    (1)  mu    = (1/N) * SUM_i x_i
//    (2)  var   = (1/N) * SUM_i (x_i - mu)^2
//    (3)  x_hat = (x_i - mu) / sqrt(var + eps)       -- normalized
//    (4)  y_i   = gamma * x_hat_i + beta              -- scaled & shifted
//
//  Given: dL/dy  (upstream gradient, same shape as y)
//
//  We need: dL/dx, dL/dgamma, dL/dbeta
//
//  ---- Step 1: dL/dbeta ----
//
//    From (4):  dy_i / dbeta = 1
//
//    dL/dbeta = SUM_i dL/dy_i                       ... (simply sum upstream grad)
//
//  ---- Step 2: dL/dgamma ----
//
//    From (4):  dy_i / dgamma = x_hat_i
//
//    dL/dgamma = SUM_i dL/dy_i * x_hat_i
//
//  ---- Step 3: dL/dx_hat ----
//
//    From (4):  dy_i / dx_hat_i = gamma
//
//    dL/dx_hat_i = dL/dy_i * gamma
//
//  ---- Step 4: dL/dvar ----
//
//    From (3):  dx_hat_i / dvar = (x_i - mu) * (-1/2) * (var + eps)^(-3/2)
//
//    dL/dvar = SUM_i dL/dx_hat_i * (x_i - mu) * (-1/2) * (var + eps)^(-3/2)
//
//  ---- Step 5: dL/dmu ----
//
//    x_hat_i depends on mu through two paths:
//      Path A: directly in (x_i - mu) in equation (3)
//      Path B: indirectly through var in equation (2)
//
//    Path A:  dx_hat_i / dmu = -1 / sqrt(var + eps)
//    Path B:  dvar / dmu = (2/N) * SUM_i (x_i - mu) * (-1) = 0
//             (because SUM(x_i - mu) = 0 by definition of mu)
//
//    dL/dmu = SUM_i dL/dx_hat_i * (-1 / sqrt(var + eps))
//
//  ---- Step 6: dL/dx_i ----
//
//    x_i contributes to the loss through three paths:
//      Path 1: directly through x_hat (equation 3)
//      Path 2: through mu (equation 1)
//      Path 3: through var (equation 2)
//
//    dL/dx_i = dL/dx_hat_i * (1 / sqrt(var + eps))     -- Path 1
//            + dL/dvar * (2/N) * (x_i - mu)             -- Path 3
//            + dL/dmu * (1/N)                            -- Path 2
//
//  ---- Simplified Form ----
//
//  Let inv_std = 1 / sqrt(var + eps)
//  Let S1 = SUM_i dL/dy_i                    (sum of upstream gradients)
//  Let S2 = SUM_i dL/dy_i * x_hat_i          (sum of grad * normalized)
//
//  Then (substituting and simplifying):
//
//    dL/dx_i = gamma * inv_std * (
//                  dL/dy_i
//                - (1/N) * S1
//                - (1/N) * x_hat_i * S2
//              )
//
//  This requires:
//    - Two reductions per channel to compute S1 and S2
//    - One element-wise pass to compute the final gradient
//
//  This is the form we implement in the kernel below.
//
// ===========================================================================

#include "../13_tensor_class/tensor.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>

// ===========================================================================
// Forward pass kernels (needed for gradient checking)
// ===========================================================================

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

    // Pass 1: compute mean
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

    // Pass 2: compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int b = i / HW;
        int spatial = i % HW;
        float diff = input[b * C * HW + c * HW + spatial] - channel_mean;
        local_var += diff * diff;
    }
    sdata[threadIdx.x] = local_var;
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

// Host wrapper for BN forward (training mode only, computes batch stats)
void batchnorm2d_forward_with_stats(
    const Tensor<float>& input,
    const Tensor<float>& gamma,
    const Tensor<float>& beta,
    Tensor<float>& output,
    Tensor<float>& batch_mean,
    Tensor<float>& batch_var,
    float eps
) {
    int B = input.shape_[0], C = input.shape_[1];
    int H = input.shape_[2], W = input.shape_[3];
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
}

// ===========================================================================
// Kernel: BatchNorm2D Backward -- Compute S1 and S2 per channel
// ===========================================================================
//
// For each channel c, we compute two sums needed for the backward pass:
//
//   S1[c] = SUM_{b,h,w} dL/dy[b][c][h][w]
//   S2[c] = SUM_{b,h,w} dL/dy[b][c][h][w] * x_hat[b][c][h][w]
//
// where x_hat = (x - mean) / sqrt(var + eps) is the normalized input.
//
// We use parallel reduction (one block per channel) like the forward pass.
//
// Grid: (C, 1, 1) -- one block per channel
// Block: (min(N, 1024), 1, 1) -- power of 2 for reduction
//
// Shared memory layout: [0..tpb-1] for S1 reduction, [tpb..2*tpb-1] for S2
// ===========================================================================

__global__ void batchnorm_backward_reduce(
    const float* __restrict__ grad_output,  // dL/dy, shape (B, C, H, W)
    const float* __restrict__ input,        // x, shape (B, C, H, W)
    const float* __restrict__ mean,         // per-channel mean (C,)
    const float* __restrict__ var,          // per-channel variance (C,)
    float* __restrict__ sum_dy,             // S1: sum of grad_output per channel (C,)
    float* __restrict__ sum_dy_xhat,        // S2: sum of grad_output * x_hat per channel (C,)
    int B, int C, int H, int W,
    float eps
) {
    int c = blockIdx.x;
    if (c >= C) return;

    int N = B * H * W;
    int HW = H * W;

    // Shared memory: first half for S1, second half for S2
    extern __shared__ float sdata[];
    float* s_dy = sdata;                        // [0 .. blockDim.x-1]
    float* s_dy_xhat = sdata + blockDim.x;      // [blockDim.x .. 2*blockDim.x-1]

    float channel_mean = mean[c];
    float inv_std = 1.0f / sqrtf(var[c] + eps);

    // Each thread accumulates partial sums
    float local_s1 = 0.0f;
    float local_s2 = 0.0f;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        int b = i / HW;
        int spatial = i % HW;
        int idx = b * C * HW + c * HW + spatial;

        float dy = grad_output[idx];
        float x_hat = (input[idx] - channel_mean) * inv_std;

        local_s1 += dy;
        local_s2 += dy * x_hat;
    }

    s_dy[threadIdx.x] = local_s1;
    s_dy_xhat[threadIdx.x] = local_s2;
    __syncthreads();

    // Tree reduction for both sums simultaneously
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_dy[threadIdx.x] += s_dy[threadIdx.x + s];
            s_dy_xhat[threadIdx.x] += s_dy_xhat[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        sum_dy[c] = s_dy[0];
        sum_dy_xhat[c] = s_dy_xhat[0];
    }
}

// ===========================================================================
// Kernel: BatchNorm2D Backward -- Compute grad_input, grad_gamma, grad_beta
// ===========================================================================
//
// Using the simplified formula:
//
//   dL/dx_i = gamma * inv_std * (dL/dy_i - (1/N)*S1 - (1/N)*x_hat_i*S2)
//
// Also atomically accumulates:
//   dL/dgamma[c] = S2[c]    (already computed in reduce kernel)
//   dL/dbeta[c]  = S1[c]    (already computed in reduce kernel)
//
// Each thread handles one element of the (B, C, H, W) tensor.
// ===========================================================================

__global__ void batchnorm_backward_elementwise(
    const float* __restrict__ grad_output,  // dL/dy, shape (B, C, H, W)
    const float* __restrict__ input,        // x, shape (B, C, H, W)
    const float* __restrict__ mean,         // per-channel mean (C,)
    const float* __restrict__ var,          // per-channel variance (C,)
    const float* __restrict__ gamma,        // scale parameter (C,)
    const float* __restrict__ sum_dy,       // S1 per channel (C,)
    const float* __restrict__ sum_dy_xhat,  // S2 per channel (C,)
    float* __restrict__ grad_input,         // dL/dx, shape (B, C, H, W)
    int B, int C, int H, int W,
    float eps
) {
    int total = B * C * H * W;
    int HW = H * W;
    int N = B * HW;  // number of elements per channel

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        // Determine which channel this element belongs to
        int c = (idx / HW) % C;

        float inv_std = 1.0f / sqrtf(var[c] + eps);
        float x_hat = (input[idx] - mean[c]) * inv_std;

        // ---------------------------------------------------------------
        // Apply the simplified backward formula:
        //
        //   dL/dx_i = gamma[c] * inv_std * (
        //               dL/dy_i
        //             - (1/N) * S1[c]
        //             - (1/N) * x_hat_i * S2[c]
        //           )
        //
        // where S1 = sum(dL/dy) over the channel
        //       S2 = sum(dL/dy * x_hat) over the channel
        // ---------------------------------------------------------------
        float dy = grad_output[idx];
        float inv_N = 1.0f / (float)N;

        grad_input[idx] = gamma[c] * inv_std * (
            dy
            - inv_N * sum_dy[c]
            - inv_N * x_hat * sum_dy_xhat[c]
        );
    }
}

// ===========================================================================
// Host Wrapper: batchnorm2d_backward
// ===========================================================================
//
// Computes dL/dx, dL/dgamma, dL/dbeta.
//
//   grad_output: (B, C, H, W) -- upstream gradient
//   input:       (B, C, H, W) -- saved from forward pass
//   gamma:       (C,)         -- scale parameter
//   mean:        (C,)         -- batch mean (saved from forward pass)
//   var:         (C,)         -- batch variance (saved from forward pass)
//   eps:         float        -- numerical stability constant
//
// Returns: tuple of (grad_input, grad_gamma, grad_beta)
// ===========================================================================

struct BNBackwardResult {
    Tensor<float> grad_input;   // (B, C, H, W)
    Tensor<float> grad_gamma;   // (C,)
    Tensor<float> grad_beta;    // (C,)
};

BNBackwardResult batchnorm2d_backward(
    const Tensor<float>& grad_output,
    const Tensor<float>& input,
    const Tensor<float>& gamma,
    const Tensor<float>& mean,
    const Tensor<float>& var,
    float eps
) {
    int B = input.shape_[0], C = input.shape_[1];
    int H = input.shape_[2], W = input.shape_[3];
    int N = B * H * W;

    // Allocate intermediate buffers for channel-wise sums
    Tensor<float> sum_dy({C}, Device::GPU);
    Tensor<float> sum_dy_xhat({C}, Device::GPU);

    // Step 1: Compute S1 and S2 per channel (parallel reduction)
    int tpb = 1;
    while (tpb * 2 <= N && tpb * 2 <= 1024) tpb *= 2;
    int shared_mem = 2 * tpb * sizeof(float);  // two arrays in shared memory

    batchnorm_backward_reduce<<<C, tpb, shared_mem>>>(
        grad_output.data_ptr(), input.data_ptr(),
        mean.data_ptr(), var.data_ptr(),
        sum_dy.data_ptr(), sum_dy_xhat.data_ptr(),
        B, C, H, W, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // grad_gamma = S2, grad_beta = S1  (already computed per channel)
    // Note: S2 = sum(dL/dy * x_hat) which IS dL/dgamma
    //       S1 = sum(dL/dy)          which IS dL/dbeta
    Tensor<float> grad_gamma = sum_dy_xhat;  // share the buffer (it's already correct)
    Tensor<float> grad_beta  = sum_dy;

    // Step 2: Compute grad_input element-wise
    Tensor<float> grad_input({B, C, H, W}, Device::GPU);
    int total = B * C * H * W;

    batchnorm_backward_elementwise<<<(total + 255) / 256, 256>>>(
        grad_output.data_ptr(), input.data_ptr(),
        mean.data_ptr(), var.data_ptr(), gamma.data_ptr(),
        sum_dy.data_ptr(), sum_dy_xhat.data_ptr(),
        grad_input.data_ptr(),
        B, C, H, W, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return {grad_input, grad_gamma, grad_beta};
}

// ===========================================================================
// CPU Reference: BN forward (for finite-difference checking)
// ===========================================================================

void batchnorm_forward_cpu(
    const float* input, const float* gamma, const float* beta,
    float* output, float* out_mean, float* out_var,
    int B, int C, int H, int W, float eps
) {
    int HW = H * W;
    int N = B * HW;

    for (int c = 0; c < C; c++) {
        // Compute mean
        float mu = 0.0f;
        for (int b = 0; b < B; b++)
            for (int s = 0; s < HW; s++)
                mu += input[b * C * HW + c * HW + s];
        mu /= (float)N;
        out_mean[c] = mu;

        // Compute variance
        float v = 0.0f;
        for (int b = 0; b < B; b++)
            for (int s = 0; s < HW; s++) {
                float d = input[b * C * HW + c * HW + s] - mu;
                v += d * d;
            }
        v /= (float)N;
        out_var[c] = v;

        // Normalize and apply affine
        float inv_std = 1.0f / sqrtf(v + eps);
        for (int b = 0; b < B; b++)
            for (int s = 0; s < HW; s++) {
                int idx = b * C * HW + c * HW + s;
                float x_hat = (input[idx] - mu) * inv_std;
                output[idx] = gamma[c] * x_hat + beta[c];
            }
    }
}

// Compute scalar loss = sum(BN(input))
float bn_loss_cpu(
    const float* input, const float* gamma, const float* beta,
    int B, int C, int H, int W, float eps
) {
    int total = B * C * H * W;
    std::vector<float> output(total);
    std::vector<float> mu(C), v(C);
    batchnorm_forward_cpu(input, gamma, beta, output.data(),
                          mu.data(), v.data(), B, C, H, W, eps);
    float loss = 0.0f;
    for (int i = 0; i < total; i++) loss += output[i];
    return loss;
}

// ===========================================================================
// Test: BatchNorm2D Backward
// ===========================================================================

int main() {
    printf("=== Chapter 15: BatchNorm2D Backward Pass Test ===\n\n");

    float eps = 1e-5f;

    // -----------------------------------------------------------------
    // Test configuration
    // -----------------------------------------------------------------
    int B = 2, C = 4, H = 3, W = 3;
    int total = B * C * H * W;
    int N = B * H * W;

    printf("Configuration:\n");
    printf("  Input shape: (%d, %d, %d, %d), N=%d per channel\n", B, C, H, W, N);
    printf("  Epsilon: %.1e\n\n", eps);

    // Create random input and parameters
    Tensor<float> input_cpu = Tensor<float>::randn({B, C, H, W}, Device::CPU);
    Tensor<float> gamma_cpu = Tensor<float>::ones({C}, Device::CPU);
    Tensor<float> beta_cpu  = Tensor<float>::zeros({C}, Device::CPU);

    // Make gamma non-trivial for a better test
    for (int c = 0; c < C; c++) {
        gamma_cpu.data_ptr()[c] = 1.0f + 0.5f * c;
        beta_cpu.data_ptr()[c]  = 0.1f * c;
    }

    // Scale input for numerical stability
    for (int i = 0; i < total; i++)
        input_cpu.data_ptr()[i] *= 0.5f;

    // -----------------------------------------------------------------
    // GPU forward pass (save mean, var for backward)
    // -----------------------------------------------------------------
    Tensor<float> input_gpu = input_cpu.to_gpu();
    Tensor<float> gamma_gpu = gamma_cpu.to_gpu();
    Tensor<float> beta_gpu  = beta_cpu.to_gpu();
    Tensor<float> output_gpu({B, C, H, W}, Device::GPU);
    Tensor<float> batch_mean({C}, Device::GPU);
    Tensor<float> batch_var({C}, Device::GPU);

    batchnorm2d_forward_with_stats(
        input_gpu, gamma_gpu, beta_gpu,
        output_gpu, batch_mean, batch_var, eps);

    // Use grad_output = all 1s (loss = sum(output))
    Tensor<float> grad_output = Tensor<float>::ones({B, C, H, W}, Device::GPU);

    // -----------------------------------------------------------------
    // Analytical backward pass
    // -----------------------------------------------------------------
    BNBackwardResult bwd = batchnorm2d_backward(
        grad_output, input_gpu, gamma_gpu, batch_mean, batch_var, eps);

    Tensor<float> gi_cpu  = bwd.grad_input.to_cpu();
    Tensor<float> gg_cpu  = bwd.grad_gamma.to_cpu();
    Tensor<float> gb_cpu  = bwd.grad_beta.to_cpu();

    // -----------------------------------------------------------------
    // Test 1: Gradient w.r.t. Input
    // -----------------------------------------------------------------
    {
        printf("Test 1: Gradient w.r.t. Input (dL/dx)\n");

        float fd_eps = 1e-3f;
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;

        for (int i = 0; i < total; i++) {
            float orig = input_cpu.data_ptr()[i];

            input_cpu.data_ptr()[i] = orig + fd_eps;
            float lp = bn_loss_cpu(input_cpu.data_ptr(), gamma_cpu.data_ptr(),
                                   beta_cpu.data_ptr(), B, C, H, W, eps);

            input_cpu.data_ptr()[i] = orig - fd_eps;
            float lm = bn_loss_cpu(input_cpu.data_ptr(), gamma_cpu.data_ptr(),
                                   beta_cpu.data_ptr(), B, C, H, W, eps);

            input_cpu.data_ptr()[i] = orig;

            float numerical  = (lp - lm) / (2.0f * fd_eps);
            float analytical = gi_cpu.data_ptr()[i];

            float abs_err = fabsf(numerical - analytical);
            float denom = fmaxf(fabsf(numerical) + fabsf(analytical), 1e-8f);
            float rel_err = abs_err / denom;

            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }

        printf("  Checked %d elements\n", total);
        printf("  Max absolute error: %.6e\n", max_abs_err);
        printf("  Max relative error: %.6e\n", max_rel_err);
        printf("  Result: %s\n\n",
               max_rel_err < 1e-2 ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 2: Gradient w.r.t. Gamma
    // -----------------------------------------------------------------
    {
        printf("Test 2: Gradient w.r.t. Gamma (dL/dgamma)\n");

        float fd_eps = 1e-3f;
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;

        for (int c = 0; c < C; c++) {
            float orig = gamma_cpu.data_ptr()[c];

            gamma_cpu.data_ptr()[c] = orig + fd_eps;
            float lp = bn_loss_cpu(input_cpu.data_ptr(), gamma_cpu.data_ptr(),
                                   beta_cpu.data_ptr(), B, C, H, W, eps);

            gamma_cpu.data_ptr()[c] = orig - fd_eps;
            float lm = bn_loss_cpu(input_cpu.data_ptr(), gamma_cpu.data_ptr(),
                                   beta_cpu.data_ptr(), B, C, H, W, eps);

            gamma_cpu.data_ptr()[c] = orig;

            float numerical  = (lp - lm) / (2.0f * fd_eps);
            float analytical = gg_cpu.data_ptr()[c];

            float abs_err = fabsf(numerical - analytical);
            float denom = fmaxf(fabsf(numerical) + fabsf(analytical), 1e-8f);
            float rel_err = abs_err / denom;

            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }

        printf("  Checked %d elements\n", C);
        printf("  Max absolute error: %.6e\n", max_abs_err);
        printf("  Max relative error: %.6e\n", max_rel_err);
        printf("  Result: %s\n\n",
               max_rel_err < 1e-2 ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 3: Gradient w.r.t. Beta
    // -----------------------------------------------------------------
    {
        printf("Test 3: Gradient w.r.t. Beta (dL/dbeta)\n");

        float fd_eps = 1e-3f;
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;

        for (int c = 0; c < C; c++) {
            float orig = beta_cpu.data_ptr()[c];

            beta_cpu.data_ptr()[c] = orig + fd_eps;
            float lp = bn_loss_cpu(input_cpu.data_ptr(), gamma_cpu.data_ptr(),
                                   beta_cpu.data_ptr(), B, C, H, W, eps);

            beta_cpu.data_ptr()[c] = orig - fd_eps;
            float lm = bn_loss_cpu(input_cpu.data_ptr(), gamma_cpu.data_ptr(),
                                   beta_cpu.data_ptr(), B, C, H, W, eps);

            beta_cpu.data_ptr()[c] = orig;

            float numerical  = (lp - lm) / (2.0f * fd_eps);
            float analytical = gb_cpu.data_ptr()[c];

            float abs_err = fabsf(numerical - analytical);
            float denom = fmaxf(fabsf(numerical) + fabsf(analytical), 1e-8f);
            float rel_err = abs_err / denom;

            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
        }

        printf("  Checked %d elements\n", C);
        printf("  Max absolute error: %.6e\n", max_abs_err);
        printf("  Max relative error: %.6e\n", max_rel_err);
        printf("  Result: %s\n\n",
               max_rel_err < 1e-2 ? "PASS" : "FAIL");
    }

    // -----------------------------------------------------------------
    // Test 4: Known property -- grad_beta should equal N (with all-ones grad)
    // -----------------------------------------------------------------
    {
        printf("Test 4: Sanity check -- grad_beta with all-ones upstream grad\n");

        // If dL/dy = 1 for all elements, then:
        //   dL/dbeta[c] = sum(dL/dy) over channel = N
        bool pass = true;
        for (int c = 0; c < C; c++) {
            float expected = (float)N;
            if (fabsf(gb_cpu.data_ptr()[c] - expected) > 1e-3f) {
                pass = false;
                printf("  FAIL: grad_beta[%d] = %.4f, expected %.4f\n",
                       c, gb_cpu.data_ptr()[c], expected);
            }
        }
        if (pass) {
            printf("  grad_beta = [%.1f", gb_cpu.data_ptr()[0]);
            for (int c = 1; c < C; c++) printf(", %.1f", gb_cpu.data_ptr()[c]);
            printf("] (expected all %.1f) -- PASS\n", (float)N);
        }
    }

    printf("\n=== BatchNorm2D Backward Tests Complete ===\n");
    return 0;
}
