// ===========================================================================
// Chapter 13: tensor_test.cu -- Tensor Class Test Program
// ===========================================================================
// This program exercises the entire Tensor API we've built:
//
//   1. Tensor creation (zeros, ones, randn, arange, from data)
//   2. Device transfer (CPU <-> GPU)
//   3. Shape operations (reshape, view, transpose, contiguous)
//   4. Element access (operator(), at())
//   5. Element-wise arithmetic (+, -, *, /, scalar ops)
//   6. Unary operations (neg, abs, exp, log, sqrt)
//   7. Shared storage (reshape shares data via reference counting)
//
// The goal is for this to feel like a mini PyTorch demo:
//   auto x = Tensor<float>::randn({3, 4}, Device::GPU);
//   auto y = Tensor<float>::ones({3, 4}, Device::GPU);
//   auto z = x + y;
//   z.print();
//
// Each test prints PASS or FAIL. At the end, a summary is printed.
// ===========================================================================

#include "tensor_ops.cuh"
#include <cstdio>
#include <cmath>

// ===========================================================================
// Test helpers
// ===========================================================================

int tests_passed = 0;
int tests_failed = 0;

// Print a test result and update counters
void check(bool condition, const char* test_name) {
    if (condition) {
        printf("  [PASS] %s\n", test_name);
        tests_passed++;
    } else {
        printf("  [FAIL] %s\n", test_name);
        tests_failed++;
    }
}

// Check if two floats are approximately equal (within tolerance)
bool approx_equal(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

// ===========================================================================
// Test 1: Tensor Creation
// ===========================================================================
// Test all the factory methods and constructors.
// Verify shapes, sizes, and initial values.
// ===========================================================================

void test_creation() {
    printf("\n=== Test 1: Tensor Creation ===\n");

    // --- zeros ---
    // Like torch.zeros(3, 4): all elements should be 0.0
    auto z = Tensor<float>::zeros({3, 4});
    check(z.size_ == 12, "zeros: size = 12");
    check(z.ndim() == 2, "zeros: ndim = 2");
    check(z.shape_[0] == 3 && z.shape_[1] == 4, "zeros: shape = {3, 4}");
    check(z.strides_[0] == 4 && z.strides_[1] == 1,
          "zeros: strides = {4, 1}");
    check(z.device_ == Device::CPU, "zeros: device = CPU");
    check(z(0, 0) == 0.0f && z(2, 3) == 0.0f, "zeros: all elements are 0");

    // --- ones ---
    // Like torch.ones(2, 3): all elements should be 1.0
    auto o = Tensor<float>::ones({2, 3});
    check(o.size_ == 6, "ones: size = 6");
    bool all_ones = true;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            if (o(i, j) != 1.0f) all_ones = false;
    check(all_ones, "ones: all elements are 1");

    // --- arange ---
    // Like torch.arange(5): [0, 1, 2, 3, 4]
    auto a = Tensor<float>::arange(5);
    check(a.size_ == 5, "arange: size = 5");
    check(a.ndim() == 1, "arange: ndim = 1");
    check(a(0) == 0.0f && a(4) == 4.0f, "arange: values [0..4]");

    // --- from vector ---
    // Convenience factory for quick testing
    auto v = Tensor<float>::from_vec({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    check(v.size_ == 5, "from_vector: size = 5");
    check(v(2) == 3.0f, "from_vector: v[2] = 3.0");

    // --- from data pointer + shape ---
    // Create a 2x3 tensor from raw data
    float raw_data[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    Tensor<float> d({2, 3}, raw_data);
    check(d(0, 0) == 10.0f && d(1, 2) == 60.0f,
          "from_data: correct values");

    // --- full ---
    // Like torch.full({2, 2}, 3.14)
    auto f = Tensor<float>::full({2, 2}, 3.14f);
    check(approx_equal(f(0, 0), 3.14f) && approx_equal(f(1, 1), 3.14f),
          "full: all elements are 3.14");

    // --- randn ---
    // Statistical test: mean should be close to 0, std close to 1
    // With 10000 samples, we expect mean within ~0.03 of 0
    auto r = Tensor<float>::randn({10000});
    float sum = 0.0f;
    for (int i = 0; i < r.size_; i++) sum += r(i);
    float mean = sum / r.size_;
    check(std::fabs(mean) < 0.1f,
          "randn: mean approximately 0 (statistical)");

    z.print("zeros_3x4");
    o.print("ones_2x3");
    a.print("arange_5");
}

// ===========================================================================
// Test 2: Device Transfer
// ===========================================================================
// Test moving tensors between CPU and GPU.
// The key invariant: values must survive the round trip.
// ===========================================================================

void test_device_transfer() {
    printf("\n=== Test 2: Device Transfer (CPU <-> GPU) ===\n");

    // Create a tensor on CPU with known values
    auto cpu_tensor = Tensor<float>::arange(12).reshape({3, 4});
    check(cpu_tensor.device_ == Device::CPU, "initial: device = CPU");

    // Transfer to GPU
    auto gpu_tensor = cpu_tensor.to_gpu();
    check(gpu_tensor.device_ == Device::GPU, "to_gpu: device = GPU");
    check(gpu_tensor.size_ == 12, "to_gpu: size preserved");
    check(gpu_tensor.shape_[0] == 3 && gpu_tensor.shape_[1] == 4,
          "to_gpu: shape preserved");

    // Transfer back to CPU and verify values survived the round trip
    auto back_to_cpu = gpu_tensor.to_cpu();
    check(back_to_cpu.device_ == Device::CPU, "to_cpu: device = CPU");

    bool values_match = true;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            if (back_to_cpu(i, j) != cpu_tensor(i, j))
                values_match = false;
    check(values_match, "round trip: values preserved (CPU->GPU->CPU)");

    // Test .to(Device) generic method
    auto gpu2 = cpu_tensor.to(Device::GPU);
    check(gpu2.device_ == Device::GPU, "to(GPU): works");
    auto cpu2 = gpu2.to(Device::CPU);
    check(cpu2.device_ == Device::CPU, "to(CPU): works");

    // Test GPU factory methods (create directly on GPU)
    auto gpu_ones = Tensor<float>::ones({2, 3}, Device::GPU);
    check(gpu_ones.device_ == Device::GPU, "ones on GPU: device = GPU");
    auto gpu_ones_cpu = gpu_ones.to_cpu();
    bool gpu_ones_correct = true;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            if (gpu_ones_cpu(i, j) != 1.0f) gpu_ones_correct = false;
    check(gpu_ones_correct, "ones on GPU: all values are 1");

    gpu_tensor.print("gpu_tensor");
    back_to_cpu.print("back_to_cpu");
}

// ===========================================================================
// Test 3: Reshape and View
// ===========================================================================
// Test that reshape changes shape/strides without copying data.
// The key invariant: reshaped tensor shares the same data buffer.
// ===========================================================================

void test_reshape() {
    printf("\n=== Test 3: Reshape and View ===\n");

    // Create a 3x4 tensor with sequential values [0..11]
    auto a = Tensor<float>::arange(12).reshape({3, 4});
    check(a.shape_[0] == 3 && a.shape_[1] == 4, "original: shape {3,4}");

    // Reshape to 4x3
    auto b = a.reshape({4, 3});
    check(b.shape_[0] == 4 && b.shape_[1] == 3, "reshape: shape {4,3}");
    check(b.strides_[0] == 3 && b.strides_[1] == 1,
          "reshape: strides {3,1}");

    // Verify they share the same data buffer
    check(a.data_.get() == b.data_.get(),
          "reshape: shares data (same pointer)");

    // Reshape to 2x6
    auto c = a.reshape({2, 6});
    check(c.shape_[0] == 2 && c.shape_[1] == 6, "reshape: shape {2,6}");
    check(c.data_.get() == a.data_.get(),
          "reshape: still shares data");

    // Reshape to 1D (flatten)
    auto d = a.reshape({12});
    check(d.ndim() == 1 && d.shape_[0] == 12, "reshape: flatten to {12}");

    // Reshape with -1 (infer dimension)
    auto e = a.reshape({-1, 2});
    check(e.shape_[0] == 6 && e.shape_[1] == 2,
          "reshape with -1: inferred {6,2}");

    auto f = a.reshape({4, -1});
    check(f.shape_[0] == 4 && f.shape_[1] == 3,
          "reshape with -1: inferred {4,3}");

    // View is an alias for reshape
    auto g = a.view({6, 2});
    check(g.shape_[0] == 6 && g.shape_[1] == 2,
          "view: same as reshape {6,2}");

    // Verify values are correct through the reshaped view
    // a(1, 2) = element at row 1, col 2 of 3x4 = index 6 = value 6.0
    check(a(1, 2) == 6.0f, "original a(1,2) = 6.0");
    // b(2, 0) = element at row 2, col 0 of 4x3 = index 6 = value 6.0
    check(b(2, 0) == 6.0f, "reshaped b(2,0) = 6.0 (same element)");

    a.print("original_3x4");
    b.print("reshaped_4x3");
    c.print("reshaped_2x6");
}

// ===========================================================================
// Test 4: Transpose
// ===========================================================================
// Test that transpose swaps strides without copying data.
// The transposed tensor is non-contiguous.
// ===========================================================================

void test_transpose() {
    printf("\n=== Test 4: Transpose ===\n");

    // Create a 3x4 matrix
    auto a = Tensor<float>::arange(12).reshape({3, 4});
    check(a.is_contiguous(), "original: is contiguous");

    // Transpose to 4x3
    auto t = a.transpose();
    check(t.shape_[0] == 4 && t.shape_[1] == 3,
          "transpose: shape {4,3}");
    check(t.strides_[0] == 1 && t.strides_[1] == 4,
          "transpose: strides {1,4} (swapped)");
    check(!t.is_contiguous(),
          "transpose: is NOT contiguous");
    check(t.data_.get() == a.data_.get(),
          "transpose: shares data (no copy)");

    // Verify transposed values using element access
    // Original a:
    //   [ 0  1  2  3 ]
    //   [ 4  5  6  7 ]
    //   [ 8  9  10 11]
    //
    // Transposed t:
    //   [ 0  4  8  ]
    //   [ 1  5  9  ]
    //   [ 2  6  10 ]
    //   [ 3  7  11 ]
    check(t(0, 0) == 0.0f, "transpose: t(0,0) = 0");
    check(t(0, 1) == 4.0f, "transpose: t(0,1) = 4");
    check(t(1, 0) == 1.0f, "transpose: t(1,0) = 1");
    check(t(3, 2) == 11.0f, "transpose: t(3,2) = 11");

    // Make contiguous (should copy data into row-major order for {4,3})
    auto tc = t.contiguous();
    check(tc.is_contiguous(), "contiguous(): now contiguous");
    check(tc.shape_[0] == 4 && tc.shape_[1] == 3,
          "contiguous(): shape preserved {4,3}");
    check(tc.strides_[0] == 3 && tc.strides_[1] == 1,
          "contiguous(): strides {3,1}");
    // Data should be different pointer (new allocation)
    check(tc.data_.get() != t.data_.get(),
          "contiguous(): different data buffer (copied)");
    // Values should match the transposed view
    check(tc(0, 1) == 4.0f && tc(3, 2) == 11.0f,
          "contiguous(): values preserved");

    a.print("original_3x4");
    t.print("transposed_4x3");
    tc.print("contiguous_4x3");
}

// ===========================================================================
// Test 5: GPU Transpose
// ===========================================================================
// Test that transpose and contiguous work on GPU tensors.
// This exercises the GPU contiguous kernel.
// ===========================================================================

void test_gpu_transpose() {
    printf("\n=== Test 5: GPU Transpose + Contiguous ===\n");

    // Create a 3x4 matrix on GPU
    auto a = Tensor<float>::arange(12).reshape({3, 4}).to_gpu();
    check(a.device_ == Device::GPU, "GPU tensor created");

    // Transpose on GPU
    auto t = a.transpose();
    check(t.shape_[0] == 4 && t.shape_[1] == 3,
          "GPU transpose: shape {4,3}");
    check(!t.is_contiguous(), "GPU transpose: not contiguous");

    // Make contiguous on GPU (uses CUDA kernel)
    auto tc = t.contiguous();
    check(tc.is_contiguous(), "GPU contiguous: now contiguous");

    // Transfer back to CPU and verify values
    auto tc_cpu = tc.to_cpu();
    check(tc_cpu(0, 0) == 0.0f, "GPU transpose->contiguous: [0,0] = 0");
    check(tc_cpu(0, 1) == 4.0f, "GPU transpose->contiguous: [0,1] = 4");
    check(tc_cpu(3, 2) == 11.0f, "GPU transpose->contiguous: [3,2] = 11");

    tc.print("gpu_transposed_contiguous");
}

// ===========================================================================
// Test 6: Element-wise Arithmetic on CPU
// ===========================================================================
// Test +, -, *, / between tensors and with scalars.
// All operations verified against expected values.
// ===========================================================================

void test_arithmetic_cpu() {
    printf("\n=== Test 6: Element-wise Arithmetic (CPU) ===\n");

    // Two simple tensors
    auto a = Tensor<float>::from_vec({1.0f, 2.0f, 3.0f, 4.0f});
    auto b = Tensor<float>::from_vec({10.0f, 20.0f, 30.0f, 40.0f});

    // Addition: [11, 22, 33, 44]
    auto c = a + b;
    check(approx_equal(c(0), 11.0f) && approx_equal(c(3), 44.0f),
          "CPU add: a + b correct");

    // Subtraction: [-9, -18, -27, -36]
    auto d = a - b;
    check(approx_equal(d(0), -9.0f) && approx_equal(d(3), -36.0f),
          "CPU sub: a - b correct");

    // Multiplication: [10, 40, 90, 160]
    auto e = a * b;
    check(approx_equal(e(0), 10.0f) && approx_equal(e(3), 160.0f),
          "CPU mul: a * b correct");

    // Division: [0.1, 0.1, 0.1, 0.1]
    auto f = a / b;
    check(approx_equal(f(0), 0.1f) && approx_equal(f(3), 0.1f),
          "CPU div: a / b correct");

    // Scalar operations
    auto g = a + 100.0f;   // [101, 102, 103, 104]
    check(approx_equal(g(0), 101.0f), "CPU add scalar: a + 100");

    auto h = a * 3.0f;     // [3, 6, 9, 12]
    check(approx_equal(h(0), 3.0f) && approx_equal(h(3), 12.0f),
          "CPU mul scalar: a * 3");

    auto i = 10.0f + a;    // [11, 12, 13, 14] (scalar + tensor)
    check(approx_equal(i(0), 11.0f), "CPU scalar + tensor: 10 + a");

    auto j = a - 1.0f;     // [0, 1, 2, 3]
    check(approx_equal(j(0), 0.0f), "CPU tensor - scalar: a - 1");

    auto k = a / 2.0f;     // [0.5, 1.0, 1.5, 2.0]
    check(approx_equal(k(0), 0.5f), "CPU tensor / scalar: a / 2");

    c.print("a + b");
    g.print("a + 100");
}

// ===========================================================================
// Test 7: Element-wise Arithmetic on GPU
// ===========================================================================
// Same tests as CPU, but on GPU. The key test is that CUDA kernels
// produce the same results as CPU operations.
// ===========================================================================

void test_arithmetic_gpu() {
    printf("\n=== Test 7: Element-wise Arithmetic (GPU) ===\n");

    // Create tensors on GPU
    auto a_cpu = Tensor<float>::from_vec({1.0f, 2.0f, 3.0f, 4.0f});
    auto b_cpu = Tensor<float>::from_vec({10.0f, 20.0f, 30.0f, 40.0f});
    auto a = a_cpu.to_gpu();
    auto b = b_cpu.to_gpu();

    // Addition on GPU
    auto c = (a + b).to_cpu();
    check(approx_equal(c(0), 11.0f) && approx_equal(c(3), 44.0f),
          "GPU add: a + b correct");

    // Subtraction on GPU
    auto d = (a - b).to_cpu();
    check(approx_equal(d(0), -9.0f) && approx_equal(d(3), -36.0f),
          "GPU sub: a - b correct");

    // Multiplication on GPU
    auto e = (a * b).to_cpu();
    check(approx_equal(e(0), 10.0f) && approx_equal(e(3), 160.0f),
          "GPU mul: a * b correct");

    // Division on GPU
    auto f = (a / b).to_cpu();
    check(approx_equal(f(0), 0.1f) && approx_equal(f(3), 0.1f),
          "GPU div: a / b correct");

    // Scalar operations on GPU
    auto g = (a + 100.0f).to_cpu();
    check(approx_equal(g(0), 101.0f), "GPU add scalar: a + 100");

    auto h = (a * 3.0f).to_cpu();
    check(approx_equal(h(0), 3.0f) && approx_equal(h(3), 12.0f),
          "GPU mul scalar: a * 3");

    (a + b).print("GPU a + b");
}

// ===========================================================================
// Test 8: Unary Operations
// ===========================================================================
// Test neg, abs, exp, log, sqrt on both CPU and GPU.
// ===========================================================================

void test_unary_ops() {
    printf("\n=== Test 8: Unary Operations ===\n");

    auto a = Tensor<float>::from_vec({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});

    // --- Negate ---
    auto neg = tensor_neg(a);
    check(approx_equal(neg(0), 2.0f) && approx_equal(neg(4), -2.0f),
          "CPU neg: -[-2,-1,0,1,2] = [2,1,0,-1,-2]");

    // Unary minus operator
    auto neg2 = -a;
    check(approx_equal(neg2(0), 2.0f), "CPU unary minus: -a");

    // --- Abs ---
    auto ab = tensor_abs(a);
    check(approx_equal(ab(0), 2.0f) && approx_equal(ab(2), 0.0f),
          "CPU abs: |[-2,-1,0,1,2]| = [2,1,0,1,2]");

    // --- Exp ---
    auto b = Tensor<float>::from_vec({0.0f, 1.0f, 2.0f});
    auto ex = tensor_exp(b);
    check(approx_equal(ex(0), 1.0f) &&
          approx_equal(ex(1), std::exp(1.0f)),
          "CPU exp: e^[0,1,2]");

    // --- Log ---
    auto c = Tensor<float>::from_vec({1.0f, std::exp(1.0f), std::exp(2.0f)});
    auto lg = tensor_log(c);
    check(approx_equal(lg(0), 0.0f) && approx_equal(lg(1), 1.0f),
          "CPU log: ln[1, e, e^2] = [0, 1, 2]");

    // --- Sqrt ---
    auto d = Tensor<float>::from_vec({0.0f, 1.0f, 4.0f, 9.0f});
    auto sq = tensor_sqrt(d);
    check(approx_equal(sq(2), 2.0f) && approx_equal(sq(3), 3.0f),
          "CPU sqrt: sqrt[0,1,4,9] = [0,1,2,3]");

    // --- GPU versions ---
    auto a_gpu = a.to_gpu();

    auto neg_gpu = tensor_neg(a_gpu).to_cpu();
    check(approx_equal(neg_gpu(0), 2.0f),
          "GPU neg: correct");

    auto abs_gpu = tensor_abs(a_gpu).to_cpu();
    check(approx_equal(abs_gpu(0), 2.0f),
          "GPU abs: correct");

    auto b_gpu = b.to_gpu();
    auto exp_gpu = tensor_exp(b_gpu).to_cpu();
    check(approx_equal(exp_gpu(1), std::exp(1.0f)),
          "GPU exp: correct");

    auto d_gpu = d.to_gpu();
    auto sqrt_gpu = tensor_sqrt(d_gpu).to_cpu();
    check(approx_equal(sqrt_gpu(2), 2.0f),
          "GPU sqrt: correct");

    neg.print("neg(a)");
    ab.print("abs(a)");
    ex.print("exp(b)");
    sq.print("sqrt(d)");
}

// ===========================================================================
// Test 9: Reference Counting (Shared Storage)
// ===========================================================================
// Verify that reshape/view share the underlying data buffer.
// Modifying one should be visible through the other.
// ===========================================================================

void test_shared_storage() {
    printf("\n=== Test 9: Shared Storage (Reference Counting) ===\n");

    // Create original tensor
    auto a = Tensor<float>::arange(12).reshape({3, 4});

    // Create two views
    auto b = a.reshape({4, 3});
    auto c = a.reshape({2, 6});

    // All three should share the same data pointer
    check(a.data_.get() == b.data_.get(),
          "a and b share data pointer");
    check(a.data_.get() == c.data_.get(),
          "a and c share data pointer");

    // shared_ptr reference count should be 3
    // (use_count() returns the number of shared_ptr instances sharing ownership)
    check(a.data_.use_count() == 3,
          "reference count = 3 (a, b, c share storage)");

    // Modify through one view, see it through another
    // a(0, 0) and b(0, 0) both map to data[0]
    a(0, 0) = 99.0f;
    check(b(0, 0) == 99.0f,
          "write through a visible through b (shared storage)");
    check(c(0, 0) == 99.0f,
          "write through a visible through c (shared storage)");

    // When we destroy b, reference count drops but data lives on
    {
        auto d = a.reshape({12});
        check(a.data_.use_count() == 4, "ref count = 4 (added d)");
    }
    // d is out of scope now
    check(a.data_.use_count() == 3,
          "ref count back to 3 (d destroyed, data survives)");

    printf("  (Reference counting works -- memory freed only when "
           "last tensor is destroyed)\n");
}

// ===========================================================================
// Test 10: GPU Random Number Generation
// ===========================================================================
// Test that randn works on GPU using cuRAND.
// Verify statistical properties of the generated numbers.
// ===========================================================================

void test_gpu_randn() {
    printf("\n=== Test 10: GPU Random Number Generation (cuRAND) ===\n");

    // Generate random numbers on GPU
    auto r = Tensor<float>::randn({10000}, Device::GPU);
    check(r.device_ == Device::GPU, "randn on GPU: device = GPU");
    check(r.size_ == 10000, "randn on GPU: size = 10000");

    // Transfer to CPU and compute statistics
    auto r_cpu = r.to_cpu();
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < r_cpu.size_; i++) {
        sum += r_cpu(i);
        sum_sq += r_cpu(i) * r_cpu(i);
    }
    float mean = sum / r_cpu.size_;
    float variance = sum_sq / r_cpu.size_ - mean * mean;
    float std_dev = std::sqrt(variance);

    printf("  Generated %d random numbers on GPU:\n", r_cpu.size_);
    printf("    mean = %.4f (expected ~0.0)\n", mean);
    printf("    std  = %.4f (expected ~1.0)\n", std_dev);

    check(std::fabs(mean) < 0.1f, "GPU randn: mean ~ 0");
    check(std::fabs(std_dev - 1.0f) < 0.1f, "GPU randn: std ~ 1");

    // Test odd-sized generation (cuRAND requires even count internally)
    auto r_odd = Tensor<float>::randn({101}, Device::GPU);
    check(r_odd.size_ == 101, "randn odd size (101): handled correctly");
}

// ===========================================================================
// Test 11: Chained Operations (Mini PyTorch Demo)
// ===========================================================================
// Show that our API supports fluent, readable code.
// This is what using the library should feel like.
// ===========================================================================

void test_chained_operations() {
    printf("\n=== Test 11: Chained Operations (Mini PyTorch Demo) ===\n");

    // -----------------------------------------------------------------------
    // Simulate a simple neural network computation:
    //   output = (input * weights + bias) * 0.5
    // -----------------------------------------------------------------------

    printf("  Simulating: output = (input * weights + bias) * 0.5\n\n");

    // "Input": a 2x3 matrix (batch_size=2, features=3)
    float input_data[] = {1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f};
    auto input = Tensor<float>({2, 3}, input_data, Device::GPU);

    // "Weights": element-wise scaling (not a real weight matrix, but shows ops)
    auto weights = Tensor<float>::ones({2, 3}, Device::GPU) * 2.0f;

    // "Bias": added to every element
    auto bias = Tensor<float>::full({2, 3}, 0.5f, Device::GPU);

    // Forward pass
    auto pre_activation = input * weights + bias;  // element-wise
    auto output = pre_activation * 0.5f;            // scale down

    // Display results
    input.print("input");
    weights.print("weights");
    bias.print("bias");
    pre_activation.print("pre_activation (input * weights + bias)");
    output.print("output (pre_activation * 0.5)");

    // Verify a specific value:
    // input[0,0]=1, weights[0,0]=2, bias[0,0]=0.5
    // pre_act[0,0] = 1*2 + 0.5 = 2.5
    // output[0,0] = 2.5 * 0.5 = 1.25
    auto out_cpu = output.to_cpu();
    check(approx_equal(out_cpu(0, 0), 1.25f),
          "output[0,0] = 1.25 (1*2+0.5)*0.5");
    check(approx_equal(out_cpu(1, 2), 6.25f),
          "output[1,2] = 6.25 (6*2+0.5)*0.5");

    // -----------------------------------------------------------------------
    // Demonstrate reshape + arithmetic chain
    // -----------------------------------------------------------------------
    printf("\n  Reshape + arithmetic chain:\n");

    auto x = Tensor<float>::arange(24, Device::GPU);
    auto y = x.reshape({2, 3, 4});
    y.print("arange(24).reshape({2,3,4})");

    auto z = y.reshape({6, 4});
    z.print("reshaped to {6,4}");

    auto t = z.reshape({4, 6}).transpose();
    t.print("reshape({4,6}).transpose()");
    check(t.shape_[0] == 6 && t.shape_[1] == 4,
          "chain: final shape {6,4} after transpose");
}

// ===========================================================================
// Test 12: Large Tensor Performance Sanity Check
// ===========================================================================
// Create a large tensor on GPU and perform operations.
// Not a real benchmark, just verify no crashes with realistic sizes.
// ===========================================================================

void test_large_tensor() {
    printf("\n=== Test 12: Large Tensor Smoke Test ===\n");

    int N = 1000000;  // 1 million elements
    printf("  Creating tensors with %d elements...\n", N);

    auto a = Tensor<float>::randn({N}, Device::GPU);
    auto b = Tensor<float>::randn({N}, Device::GPU);

    // Chain of operations
    auto c = a + b;
    auto d = c * 2.0f;
    auto e = tensor_abs(d);
    auto f = tensor_sqrt(e);

    check(f.size_ == N, "large tensor: size preserved through chain");
    check(f.device_ == Device::GPU, "large tensor: stays on GPU");

    // Verify result is finite (no NaN/inf from the operations)
    auto f_cpu = f.to_cpu();
    bool all_finite = true;
    for (int i = 0; i < std::min(1000, N); i++) {
        if (std::isnan(f_cpu(i)) || std::isinf(f_cpu(i))) {
            all_finite = false;
            break;
        }
    }
    check(all_finite, "large tensor: results are finite");

    printf("  Operations on %d elements completed successfully.\n", N);
}

// ===========================================================================
// Main
// ===========================================================================

int main() {
    printf("===========================================================\n");
    printf("  Chapter 13: Tensor Class Test Suite\n");
    printf("  Building the cudalearn Deep Learning Library\n");
    printf("===========================================================\n");

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  GPU: %s (Compute Capability %d.%d)\n",
           prop.name, prop.major, prop.minor);
    printf("  Global Memory: %.0f MB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("===========================================================\n");

    // Run all tests
    test_creation();
    test_device_transfer();
    test_reshape();
    test_transpose();
    test_gpu_transpose();
    test_arithmetic_cpu();
    test_arithmetic_gpu();
    test_unary_ops();
    test_shared_storage();
    test_gpu_randn();
    test_chained_operations();
    test_large_tensor();

    // Print summary
    printf("\n===========================================================\n");
    printf("  RESULTS: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("===========================================================\n");

    if (tests_failed == 0) {
        printf("  All tests passed! The Tensor class is ready.\n");
        printf("  Next: Chapter 14 -- Forward Operations (matmul, "
               "activations, reductions)\n");
    } else {
        printf("  Some tests failed. Review the output above.\n");
    }

    return tests_failed > 0 ? 1 : 0;
}
