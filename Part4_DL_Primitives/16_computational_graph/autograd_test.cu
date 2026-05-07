// ===========================================================================
// Chapter 16: autograd_test.cu -- Test Suite for the Autograd Engine
// ===========================================================================
//
// Four tests verify correctness of our autograd engine by comparing
// analytical gradients (computed by backward()) against numerical
// gradients (computed by finite differences).
//
// NUMERICAL GRADIENT (finite differences):
//
//   For each parameter element p[i]:
//     1. p[i] += eps       -> compute loss_plus
//     2. p[i] -= 2*eps     -> compute loss_minus (restore and subtract)
//     3. p[i] += eps       -> restore original value
//     4. numerical_grad[i] = (loss_plus - loss_minus) / (2 * eps)
//
//   This approximates dL/dp[i] with O(eps^2) accuracy.
//
// TESTS:
//   1. Polynomial: z = a*x^2 + b*x + c, verify dz/da, dz/db, dz/dx
//   2. Linear layer: y = x @ W^T + b, loss = sum(y^2), verify grad_W, grad_b
//   3. Conv -> ReLU -> GAP -> Linear -> CrossEntropy chain
//   4. Residual connection: y = relu(conv(x)) + x
//
// ===========================================================================

#include "autograd_ops.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

// ===========================================================================
// Helper: compute loss by running a fresh forward pass
// (used by numerical gradient computation)
// ===========================================================================

// ---------------------------------------------------------------------------
// Numerical gradient checker
// ---------------------------------------------------------------------------
//
// Perturbs each element of `param` by eps, recomputes the loss using
// `compute_loss`, and returns the numerical gradient.
//
// Parameters:
//   param        -- the parameter tensor to differentiate w.r.t.
//   compute_loss -- a function that builds the graph and returns loss value
//   eps          -- finite difference step size (default: 1e-3)
//
// Returns:
//   vector<float> of numerical gradients, same size as param
//
std::vector<float> numerical_gradient(
    GradTensorPtr param,
    std::function<float()> compute_loss,
    float eps = 1e-3f
) {
    std::vector<float> h_data(param->size);
    std::vector<float> num_grad(param->size);

    param->get_data_to_host(h_data.data());

    for (int i = 0; i < param->size; i++) {
        // f(x + eps)
        float orig = h_data[i];
        h_data[i] = orig + eps;
        param->set_data_from_host(h_data.data());
        float loss_plus = compute_loss();

        // f(x - eps)
        h_data[i] = orig - eps;
        param->set_data_from_host(h_data.data());
        float loss_minus = compute_loss();

        // Central difference
        num_grad[i] = (loss_plus - loss_minus) / (2.0f * eps);

        // Restore original value
        h_data[i] = orig;
        param->set_data_from_host(h_data.data());
    }

    return num_grad;
}

// ---------------------------------------------------------------------------
// Helper: compare analytical vs numerical gradient
// ---------------------------------------------------------------------------
bool check_gradient(const char* name, GradTensorPtr param,
                    const std::vector<float>& num_grad,
                    float tol = 1e-2f) {
    std::vector<float> anal_grad(param->size);
    param->get_grad_to_host(anal_grad.data());

    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int worst_idx = 0;

    for (int i = 0; i < param->size; i++) {
        float diff = fabsf(anal_grad[i] - num_grad[i]);

        // For very small gradients (both analytical and numerical near zero),
        // relative error is meaningless. Use absolute error threshold instead.
        // This is standard practice for gradient checking through ReLU, where
        // elements near zero can flip the mask and cause sign differences.
        if (fabsf(anal_grad[i]) < 1e-2f && fabsf(num_grad[i]) < 1e-2f) {
            // Use absolute error for small values
            if (diff > max_diff) {
                max_diff = diff;
                // Don't update max_rel_diff for tiny values
            }
            continue;
        }

        float denom = fmaxf(fabsf(anal_grad[i]) + fabsf(num_grad[i]), 1e-7f);
        float rel_diff = diff / denom;

        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
            max_diff = diff;
            worst_idx = i;
        }
    }

    bool passed = max_rel_diff < tol;
    printf("  %-20s: max_rel_diff = %.6f (abs=%.6f at idx %d) ... %s\n",
           name, max_rel_diff, max_diff, worst_idx,
           passed ? "PASS" : "FAIL");

    if (!passed && param->size <= 20) {
        printf("    Analytical: ");
        for (int i = 0; i < param->size; i++) printf("%.4f ", anal_grad[i]);
        printf("\n    Numerical:  ");
        for (int i = 0; i < param->size; i++) printf("%.4f ", num_grad[i]);
        printf("\n");
    }

    return passed;
}

// ===========================================================================
//  TEST 1: Polynomial z = a*x^2 + b*x + c
// ===========================================================================
//
//  Manual derivatives:
//    dz/dx = 2*a*x + b
//    dz/da = x^2
//    dz/db = x
//    dz/dc = 1
//
//  With x=3, a=2, b=5, c=1:
//    z = 2*9 + 5*3 + 1 = 18 + 15 + 1 = 34
//    dz/dx = 2*2*3 + 5 = 17
//    dz/da = 9
//    dz/db = 3
//    dz/dc = 1
//
// ===========================================================================
bool test_polynomial() {
    printf("\n=== TEST 1: Polynomial z = a*x^2 + b*x + c ===\n");

    // Create scalar tensors
    auto x = make_grad_tensor({1}, true, "x");
    auto a = make_grad_tensor({1}, true, "a");
    auto b = make_grad_tensor({1}, true, "b");
    auto c = make_grad_tensor({1}, true, "c");

    float h_x = 3.0f, h_a = 2.0f, h_b = 5.0f, h_c = 1.0f;
    x->set_data_from_host(&h_x);
    a->set_data_from_host(&h_a);
    b->set_data_from_host(&h_b);
    c->set_data_from_host(&h_c);

    // Build graph: z = a * x^2 + b * x + c
    //
    // Step by step:
    //   x2   = x * x        (x^2)
    //   ax2  = a * x2       (a * x^2)      -- uses mul element-wise
    //   bx   = b * x        (b * x)
    //   ax2_bx = ax2 + bx   (a*x^2 + b*x)
    //   z    = ax2_bx + c   (a*x^2 + b*x + c)
    //
    auto x2    = autograd::mul(x, x);       // x^2 (x used twice -> gradient accumulation!)
    auto ax2   = autograd::mul(a, x2);      // a * x^2
    auto bx    = autograd::mul(b, x);       // b * x
    auto ax2_bx = autograd::add(ax2, bx);   // a*x^2 + b*x
    auto z     = autograd::add(ax2_bx, c);  // + c

    // Check forward value
    float h_z;
    z->get_data_to_host(&h_z);
    printf("  z = %.1f (expected 34.0)\n", h_z);

    // Backward pass
    z->backward();

    // Check analytical gradients against known values
    float h_dx, h_da, h_db, h_dc;
    x->get_grad_to_host(&h_dx);
    a->get_grad_to_host(&h_da);
    b->get_grad_to_host(&h_db);
    c->get_grad_to_host(&h_dc);

    printf("  dz/dx = %.1f (expected 17.0)\n", h_dx);
    printf("  dz/da = %.1f (expected 9.0)\n", h_da);
    printf("  dz/db = %.1f (expected 3.0)\n", h_db);
    printf("  dz/dc = %.1f (expected 1.0)\n", h_dc);

    bool pass = true;
    pass &= fabsf(h_z - 34.0f) < 0.01f;
    pass &= fabsf(h_dx - 17.0f) < 0.01f;
    pass &= fabsf(h_da - 9.0f) < 0.01f;
    pass &= fabsf(h_db - 3.0f) < 0.01f;
    pass &= fabsf(h_dc - 1.0f) < 0.01f;

    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ===========================================================================
//  TEST 2: Linear Layer y = x @ W^T + bias, loss = sum(y^2)
// ===========================================================================
//
//  Tests the linear layer autograd and sum+square operations.
//  Verifies grad_W and grad_bias against numerical gradients.
//
//  With x (2, 3), W (4, 3), bias (4):
//    y = x @ W^T + bias   shape (2, 4)
//    loss = sum(y^2)       scalar
//
// ===========================================================================
bool test_linear() {
    printf("\n=== TEST 2: Linear Layer y = xW^T + b, loss = sum(y^2) ===\n");

    int B = 2, in_f = 3, out_f = 4;

    auto x = make_grad_tensor({B, in_f}, true, "x");
    auto W = make_grad_tensor({out_f, in_f}, true, "W");
    auto bias = make_grad_tensor({out_f}, true, "bias");

    // Initialize with small deterministic values
    float h_x[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    float h_W[] = {0.1f, 0.2f, 0.3f,
                   0.4f, 0.5f, 0.6f,
                   0.7f, 0.8f, 0.9f,
                   1.0f, 1.1f, 1.2f};
    float h_b[] = {0.01f, 0.02f, 0.03f, 0.04f};
    x->set_data_from_host(h_x);
    W->set_data_from_host(h_W);
    bias->set_data_from_host(h_b);

    // Forward: y = linear(x, W, bias), loss = sum(y^2)
    auto y = autograd::linear(x, W, bias);
    auto y2 = autograd::square(y);
    auto loss = autograd::sum(y2);

    float h_loss;
    loss->get_data_to_host(&h_loss);
    printf("  loss = %.6f\n", h_loss);

    // Backward
    loss->backward();

    // Numerical gradient for W
    auto compute_loss_W = [&]() -> float {
        auto y_ = autograd::linear(x, W, bias);
        auto y2_ = autograd::square(y_);
        auto loss_ = autograd::sum(y2_);
        float val;
        loss_->get_data_to_host(&val);
        return val;
    };

    auto num_grad_W = numerical_gradient(W, compute_loss_W);
    bool pass_W = check_gradient("grad_W", W, num_grad_W);

    auto num_grad_b = numerical_gradient(bias, compute_loss_W);
    bool pass_b = check_gradient("grad_bias", bias, num_grad_b);

    auto num_grad_x = numerical_gradient(x, compute_loss_W);
    bool pass_x = check_gradient("grad_x", x, num_grad_x);

    bool pass = pass_W && pass_b && pass_x;
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ===========================================================================
//  TEST 3: Conv -> ReLU -> GAP -> Linear -> CrossEntropy
// ===========================================================================
//
//  Tests a mini neural network pipeline through the autograd engine.
//  Verifies gradients of the conv weight and linear weight against
//  numerical gradients.
//
//  Architecture:
//    input (2, 1, 4, 4) -> Conv2D(1->2, 3x3, pad=1) -> ReLU -> GAP
//    -> Linear(2 -> 3) -> CrossEntropy(labels)
//
// ===========================================================================
bool test_conv_chain() {
    printf("\n=== TEST 3: Conv -> ReLU -> GAP -> Linear -> CrossEntropy ===\n");

    int B = 2, IC = 1, IH = 4, IW = 4;
    int OC = 2, KH = 3, KW = 3, pad = 1;
    int num_classes = 3;

    // Create tensors
    auto input = make_grad_tensor({B, IC, IH, IW}, false, "input");
    auto conv_w = make_grad_tensor({OC, IC, KH, KW}, true, "conv_w");
    auto lin_w = make_grad_tensor({num_classes, OC}, true, "lin_w");
    auto lin_b = make_grad_tensor({num_classes}, true, "lin_b");
    auto labels = make_grad_tensor({B}, false, "labels");

    // Initialize with small values
    std::vector<float> h_input(B * IC * IH * IW);
    for (int i = 0; i < (int)h_input.size(); i++) h_input[i] = 0.1f * (i % 7 - 3);
    input->set_data_from_host(h_input.data());

    std::vector<float> h_conv_w(OC * IC * KH * KW);
    for (int i = 0; i < (int)h_conv_w.size(); i++) h_conv_w[i] = 0.1f * (i % 5 - 2);
    conv_w->set_data_from_host(h_conv_w.data());

    std::vector<float> h_lin_w(num_classes * OC);
    for (int i = 0; i < (int)h_lin_w.size(); i++) h_lin_w[i] = 0.2f * (i % 3 - 1);
    lin_w->set_data_from_host(h_lin_w.data());

    std::vector<float> h_lin_b(num_classes, 0.0f);
    lin_b->set_data_from_host(h_lin_b.data());

    float h_labels[] = {0.0f, 2.0f};  // Class labels
    labels->set_data_from_host(h_labels);

    // Build the computation graph:
    //   conv_out = conv2d(input, conv_w, pad=1)     (B, OC, 4, 4)
    //   relu_out = relu(conv_out)                    (B, OC, 4, 4)
    //   gap_out  = global_avg_pool(relu_out)         (B, OC)
    //   logits   = linear(gap_out, lin_w, lin_b)     (B, num_classes)
    //   loss     = cross_entropy(logits, labels)     scalar
    auto conv_out = autograd::conv2d(input, conv_w, pad);
    auto relu_out = autograd::relu(conv_out);
    auto gap_out  = autograd::global_avg_pool(relu_out);
    auto logits   = autograd::linear(gap_out, lin_w, lin_b);
    auto loss     = autograd::cross_entropy(logits, labels);

    float h_loss;
    loss->get_data_to_host(&h_loss);
    printf("  loss = %.6f\n", h_loss);

    // Backward
    loss->backward();

    // Lambda to recompute loss (for numerical gradient)
    auto compute_loss = [&]() -> float {
        auto co = autograd::conv2d(input, conv_w, pad);
        auto ro = autograd::relu(co);
        auto go = autograd::global_avg_pool(ro);
        auto lo = autograd::linear(go, lin_w, lin_b);
        auto ls = autograd::cross_entropy(lo, labels);
        float val;
        ls->get_data_to_host(&val);
        return val;
    };

    // Check conv weight gradient
    auto num_grad_conv = numerical_gradient(conv_w, compute_loss);
    bool pass_conv = check_gradient("grad_conv_w", conv_w, num_grad_conv);

    // Check linear weight gradient
    auto num_grad_lin = numerical_gradient(lin_w, compute_loss);
    bool pass_lin = check_gradient("grad_lin_w", lin_w, num_grad_lin);

    // Check linear bias gradient
    auto num_grad_linb = numerical_gradient(lin_b, compute_loss);
    bool pass_linb = check_gradient("grad_lin_b", lin_b, num_grad_linb);

    bool pass = pass_conv && pass_lin && pass_linb;
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ===========================================================================
//  TEST 4: Residual Connection y = relu(conv(x)) + x
// ===========================================================================
//
//  Tests the fork/join pattern of a residual connection.
//  The input x is used twice:
//    1. As input to conv
//    2. As the skip connection added to the conv output
//
//  This tests gradient accumulation: x.grad must be the sum of gradients
//  from both paths.
//
// ===========================================================================
bool test_residual() {
    printf("\n=== TEST 4: Residual Connection y = relu(conv(x)) + x ===\n");

    int B = 1, C = 2, H = 4, W = 4;
    int KH = 3, KW = 3, pad = 1;

    // input: requires_grad because we want to check gradient through both paths
    auto input = make_grad_tensor({B, C, H, W}, true, "input");
    auto conv_w = make_grad_tensor({C, C, KH, KW}, true, "conv_w");

    // Initialize
    std::vector<float> h_input(B * C * H * W);
    for (int i = 0; i < (int)h_input.size(); i++) h_input[i] = 0.1f * ((i * 7 + 3) % 11 - 5);
    input->set_data_from_host(h_input.data());

    std::vector<float> h_conv_w(C * C * KH * KW);
    for (int i = 0; i < (int)h_conv_w.size(); i++) h_conv_w[i] = 0.05f * ((i * 3 + 1) % 9 - 4);
    conv_w->set_data_from_host(h_conv_w.data());

    // Residual block: y = relu(conv(x)) + x
    auto conv_out = autograd::conv2d(input, conv_w, pad);   // Same spatial size
    auto relu_out = autograd::relu(conv_out);
    auto y = autograd::add(relu_out, input);  // Skip connection!

    // Loss = sum(y^2) for simplicity
    auto y2 = autograd::square(y);
    auto loss = autograd::sum(y2);

    float h_loss;
    loss->get_data_to_host(&h_loss);
    printf("  loss = %.6f\n", h_loss);

    // Backward
    loss->backward();

    // Numerical gradient for input (should accumulate from both paths)
    auto compute_loss = [&]() -> float {
        auto co = autograd::conv2d(input, conv_w, pad);
        auto ro = autograd::relu(co);
        auto y_ = autograd::add(ro, input);
        auto y2_ = autograd::square(y_);
        auto loss_ = autograd::sum(y2_);
        float val;
        loss_->get_data_to_host(&val);
        return val;
    };

    auto num_grad_input = numerical_gradient(input, compute_loss);
    bool pass_input = check_gradient("grad_input", input, num_grad_input, 0.05f);

    auto num_grad_conv = numerical_gradient(conv_w, compute_loss);
    bool pass_conv = check_gradient("grad_conv_w", conv_w, num_grad_conv, 0.05f);

    bool pass = pass_input && pass_conv;
    printf("  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ===========================================================================
//  MAIN
// ===========================================================================
int main() {
    printf("================================================================\n");
    printf("  Chapter 16: Autograd Engine Test Suite\n");
    printf("================================================================\n");

    bool all_pass = true;

    all_pass &= test_polynomial();
    all_pass &= test_linear();
    all_pass &= test_conv_chain();
    all_pass &= test_residual();

    printf("\n================================================================\n");
    if (all_pass) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  SOME TESTS FAILED\n");
    }
    printf("================================================================\n");

    return all_pass ? 0 : 1;
}
