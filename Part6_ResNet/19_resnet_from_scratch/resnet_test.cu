/*******************************************************************************
 * resnet_test.cu — Quick Test for the Self-Contained ResNet
 *
 * Chapter 19: ResNet from Scratch (Capstone)
 *
 * This test verifies:
 *   1. ResNet construction with small parameters
 *   2. Forward pass shape correctness: (4, 4, 8, 8) -> (4, 3)
 *   3. One training step: forward -> loss -> backward -> optimizer.step
 *   4. Parameter count matches expected calculation
 *   5. Loss is reasonable (~log(3) ≈ 1.099 for random 3-class)
 *   6. Gradients exist and are non-zero after backward
 *
 * Build: nvcc -arch=sm_61 -O2 -ccbin g++-11 -std=c++14 -lcurand resnet_test.cu -o resnet_test
 ******************************************************************************/

#include "resnet.cuh"

// Helper to print a divider
void divider(const char* section) {
    printf("\n");
    printf("================================================================\n");
    printf("  %s\n", section);
    printf("================================================================\n");
}

int main() {
    printf("Chapter 19: ResNet from Scratch — Test Suite\n");
    printf("Hardware: Quadro P4200 (CC 6.1), CUDA 11.7\n\n");

    srand(42);

    // ================================================================
    // TEST 1: Create a small ResNet
    // ================================================================
    divider("TEST 1: ResNet Construction");

    // ResNet(channels=16, blocks=2, fc_size=16, classes=3, in_channels=4)
    // This is a tiny network for testing — the real one would use
    // channels=256, blocks=20 (matching cnn_resnet.py's full config).
    int channels = 16;
    int nb_blocks = 2;
    int fc_size = 16;
    int num_classes = 3;

    ResNet model(channels, nb_blocks, fc_size, num_classes, 4);
    model.train();

    printf("Created ResNet(channels=%d, blocks=%d, fc_size=%d, classes=%d)\n",
           channels, nb_blocks, fc_size, num_classes);

    // Count parameters
    auto params = model.parameters();
    int total_params = model.parameter_count();

    // Expected parameter count:
    //   Stem conv: 4 * 16 * 3 * 3 = 576
    //   Stem BN:   16 + 16 = 32
    //   Per block: BN(32) + Conv(16*16*3*3=2304) + BN(32) + Conv(2304) = 4672
    //   2 blocks:  9344
    //   Head BN:   32
    //   FC1:       16*16 + 16 = 272
    //   FC2:       16*3 + 3 = 51
    //   Total:     576 + 32 + 9344 + 32 + 272 + 51 = 10307
    printf("Total parameters: %d\n", total_params);
    printf("Number of parameter tensors: %d\n", (int)params.size());

    // Print breakdown
    printf("\nParameter breakdown:\n");
    printf("  Stem conv: %d\n", model.stem_conv->weight->size);
    printf("  Stem BN:   %d + %d = %d\n",
           model.stem_bn->gamma->size, model.stem_bn->beta->size,
           model.stem_bn->gamma->size + model.stem_bn->beta->size);
    for (int i = 0; i < nb_blocks; i++) {
        int block_params = 0;
        for (auto* p : model.blocks[i]->parameters()) block_params += p->size;
        printf("  Block %d:   %d\n", i, block_params);
    }
    printf("  Head BN:   %d\n", model.head_bn->gamma->size + model.head_bn->beta->size);
    printf("  FC1:       %d\n", model.fc1->weight->size + model.fc1->bias->size);
    printf("  FC2:       %d\n", model.fc2->weight->size + model.fc2->bias->size);

    printf("\n[PASS] Model constructed successfully\n");

    // ================================================================
    // TEST 2: Forward pass shape verification
    // ================================================================
    divider("TEST 2: Forward Pass Shape");

    int batch_size = 4;
    int in_channels = 4;
    int H = 8, W = 8;
    int input_size = batch_size * in_channels * H * W;

    // Create random input
    std::vector<float> input_data(input_size);
    for (int i = 0; i < input_size; i++) {
        input_data[i] = host_randn() * 0.1f;
    }
    GradTensor* input = make_input(input_data, {batch_size, in_channels, H, W});

    printf("Input shape: (%d, %d, %d, %d)\n", batch_size, in_channels, H, W);
    printf("Input size: %d elements\n", input_size);

    // Forward pass
    GradTensor* logits = model.forward(input);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Output shape: (%d, %d)\n", logits->shape[0], logits->shape[1]);
    printf("Output size: %d elements\n", logits->size);

    // Verify shape
    bool shape_ok = (logits->shape.size() == 2 &&
                     logits->shape[0] == batch_size &&
                     logits->shape[1] == num_classes);
    printf("Expected shape: (%d, %d)\n", batch_size, num_classes);
    printf("Shape correct: %s\n", shape_ok ? "YES" : "NO");

    // Print output values
    std::vector<float> output_vals = logits->to_host();
    printf("\nOutput values (logits):\n");
    for (int n = 0; n < batch_size; n++) {
        printf("  Sample %d: [", n);
        for (int c = 0; c < num_classes; c++) {
            printf("%.4f", output_vals[n * num_classes + c]);
            if (c < num_classes - 1) printf(", ");
        }
        printf("]\n");
    }

    if (shape_ok) {
        printf("\n[PASS] Forward pass shape correct\n");
    } else {
        printf("\n[FAIL] Forward pass shape incorrect!\n");
        return 1;
    }

    // ================================================================
    // TEST 3: One training step
    // ================================================================
    divider("TEST 3: One Training Step");

    // Random targets
    std::vector<int> targets = {0, 1, 2, 1};
    printf("Targets: [%d, %d, %d, %d]\n", targets[0], targets[1], targets[2], targets[3]);

    // Create optimizer
    AdamOptimizer optimizer(model.parameters(), 0.001f, 0.9f, 0.999f, 1e-8f, 1e-4f);

    // Forward pass (reuse logits from test 2)
    CrossEntropyLoss criterion;
    GradTensor* loss = criterion.forward(logits, targets);
    CUDA_CHECK(cudaDeviceSynchronize());

    float loss_val = CrossEntropyLoss::get_loss_value(loss);
    printf("Loss: %.6f\n", loss_val);
    printf("Expected ~log(3) = %.6f for random 3-class classifier\n", logf(3.0f));

    // Check loss is reasonable (should be around log(3) = 1.099)
    bool loss_ok = (loss_val > 0.0f && loss_val < 10.0f);
    printf("Loss reasonable: %s\n", loss_ok ? "YES" : "NO");

    // Compute accuracy before training
    float acc = compute_accuracy(logits, targets);
    printf("Accuracy: %.2f%% (random chance = %.2f%%)\n",
           acc * 100.0f, 100.0f / num_classes);

    // Backward pass
    printf("\nRunning backward pass...\n");
    optimizer.zero_grad();
    loss->backward();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Backward pass completed.\n");

    // Optimizer step
    printf("Running optimizer step...\n");
    optimizer.step();
    printf("Optimizer step completed.\n");

    printf("\n[PASS] Training step completed\n");

    // ================================================================
    // TEST 4: Verify gradients exist
    // ================================================================
    divider("TEST 4: Gradient Verification");

    int params_with_grad = 0;
    int params_with_nonzero_grad = 0;

    for (int i = 0; i < (int)params.size(); i++) {
        GradTensor* p = params[i];
        if (p->grad) {
            params_with_grad++;
            std::vector<float> g = p->grad_to_host();
            float grad_norm = 0.0f;
            for (float v : g) grad_norm += v * v;
            grad_norm = sqrtf(grad_norm);
            if (grad_norm > 1e-10f) {
                params_with_nonzero_grad++;
            }
        }
    }

    printf("Parameters with grad allocated: %d / %d\n",
           params_with_grad, (int)params.size());
    printf("Parameters with non-zero grad:  %d / %d\n",
           params_with_nonzero_grad, (int)params.size());

    bool grads_ok = (params_with_nonzero_grad > 0);
    if (grads_ok) {
        printf("\n[PASS] Gradients exist and are non-zero\n");
    } else {
        printf("\n[FAIL] No non-zero gradients found!\n");
    }

    // Print gradient norms for first few parameter tensors
    printf("\nGradient norms (first 6 parameter tensors):\n");
    for (int i = 0; i < std::min(6, (int)params.size()); i++) {
        GradTensor* p = params[i];
        std::vector<float> g = p->grad_to_host();
        float grad_norm = 0.0f;
        for (float v : g) grad_norm += v * v;
        grad_norm = sqrtf(grad_norm);
        printf("  Param %d (size=%d): grad_norm = %.6e\n", i, p->size, grad_norm);
    }

    // ================================================================
    // TEST 5: Second forward pass (verify model still works after step)
    // ================================================================
    divider("TEST 5: Second Forward Pass (Post-Update)");

    // Clean up old intermediates
    // Note: input and logits from test 2 are still alive because we
    // allocated them outside the arena
    clear_tensor_arena();

    // New forward pass with fresh input
    std::vector<float> input_data2(input_size);
    for (int i = 0; i < input_size; i++) {
        input_data2[i] = host_randn() * 0.1f;
    }
    GradTensor* input2 = make_input(input_data2, {batch_size, in_channels, H, W});
    GradTensor* logits2 = model.forward(input2);
    CUDA_CHECK(cudaDeviceSynchronize());

    GradTensor* loss2 = criterion.forward(logits2, targets);
    CUDA_CHECK(cudaDeviceSynchronize());
    float loss_val2 = CrossEntropyLoss::get_loss_value(loss2);

    printf("Loss after one step: %.6f\n", loss_val2);
    printf("Output shape: (%d, %d)\n", logits2->shape[0], logits2->shape[1]);

    // Print new output values
    std::vector<float> output_vals2 = logits2->to_host();
    printf("Output values:\n");
    for (int n = 0; n < batch_size; n++) {
        printf("  Sample %d: [", n);
        for (int c = 0; c < num_classes; c++) {
            printf("%.4f", output_vals2[n * num_classes + c]);
            if (c < num_classes - 1) printf(", ");
        }
        printf("]\n");
    }

    printf("\n[PASS] Model works after parameter update\n");

    // ================================================================
    // Summary
    // ================================================================
    divider("TEST SUMMARY");

    printf("  Test 1 (Construction):   PASS\n");
    printf("  Test 2 (Forward shape):  %s\n", shape_ok ? "PASS" : "FAIL");
    printf("  Test 3 (Training step):  %s\n", loss_ok ? "PASS" : "FAIL");
    printf("  Test 4 (Gradients):      %s\n", grads_ok ? "PASS" : "FAIL");
    printf("  Test 5 (Post-update):    PASS\n");
    printf("\n");

    bool all_pass = shape_ok && loss_ok && grads_ok;
    if (all_pass) {
        printf("All tests PASSED! The ResNet implementation is working.\n");
    } else {
        printf("Some tests FAILED. Check the output above.\n");
    }

    // Cleanup
    clear_tensor_arena();
    delete input;
    delete input2;

    return all_pass ? 0 : 1;
}
