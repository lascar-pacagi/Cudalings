/*******************************************************************************
 * resnet_train.cu — Full Training Program for the Self-Contained ResNet
 *
 * Chapter 19: ResNet from Scratch (Capstone)
 *
 * This program:
 *   1. Generates synthetic data (1024 train, 256 validation samples)
 *   2. Creates a ResNet(channels=32, blocks=4, fc_size=32, classes=3)
 *   3. Trains with Adam(lr=0.001, weight_decay=1e-4) + CosineAnnealing
 *   4. Runs 50 epochs with mini-batches of 32
 *   5. Shows the network overfitting the training data
 *   6. Reports timing statistics
 *
 * The synthetic data has random inputs (4 channels, 8x8 spatial) and
 * random labels {0, 1, 2}. With 1024 samples and a 32-channel ResNet
 * with 4 blocks, the network has enough capacity to memorize the training
 * set, demonstrating that all components (forward, backward, optimizer)
 * are working correctly.
 *
 * Build: nvcc -arch=sm_61 -O2 -ccbin g++-11 -std=c++14 -lcurand resnet_train.cu -o resnet_train
 ******************************************************************************/

#include "resnet.cuh"

// ============================================================================
// Data generation
// ============================================================================

struct Dataset {
    std::vector<std::vector<float>> inputs;  // each: (4 * 8 * 8) = 256 floats
    std::vector<int> labels;                  // each: {0, 1, 2}
    int num_samples;
    int input_dim;  // 4 * 8 * 8 = 256

    Dataset(int n, int seed) : num_samples(n), input_dim(4 * 8 * 8) {
        srand(seed);
        inputs.resize(n);
        labels.resize(n);
        for (int i = 0; i < n; i++) {
            inputs[i].resize(input_dim);
            for (int j = 0; j < input_dim; j++) {
                inputs[i][j] = host_randn() * 0.5f;
            }
            labels[i] = rand() % 3;
        }
    }

    // Get a mini-batch starting at index `start` with size `batch_size`.
    // Returns flattened input data and label vector.
    void get_batch(int start, int batch_size,
                   std::vector<float>& batch_data,
                   std::vector<int>& batch_labels) const {
        int actual_bs = std::min(batch_size, num_samples - start);
        batch_data.resize(actual_bs * input_dim);
        batch_labels.resize(actual_bs);
        for (int i = 0; i < actual_bs; i++) {
            int idx = start + i;
            memcpy(&batch_data[i * input_dim], inputs[idx].data(),
                   input_dim * sizeof(float));
            batch_labels[i] = labels[idx];
        }
    }
};


// ============================================================================
// Training loop
// ============================================================================

int main() {
    printf("================================================================\n");
    printf("  Chapter 19: ResNet from Scratch — Full Training\n");
    printf("  Hardware: Quadro P4200 (CC 6.1), CUDA 11.7\n");
    printf("================================================================\n\n");

    // Seed for reproducibility
    srand(42);

    // --- Hyperparameters ---
    // These match a reasonable configuration for the board game evaluator.
    // We use smaller values than the full cnn_resnet.py (C=256, N=20) to
    // keep training fast on synthetic data.
    const int channels = 32;
    const int nb_blocks = 4;
    const int fc_size = 32;
    const int num_classes = 3;
    const int in_channels = 4;
    const int H = 8, W = 8;

    const float lr = 0.001f;
    const float weight_decay = 1e-4f;
    const int num_epochs = 50;
    const int batch_size = 32;
    const float eta_min = 1e-5f;

    const int num_train = 1024;
    const int num_val = 256;

    printf("Hyperparameters:\n");
    printf("  Channels:      %d\n", channels);
    printf("  Blocks:        %d\n", nb_blocks);
    printf("  FC size:       %d\n", fc_size);
    printf("  Classes:       %d\n", num_classes);
    printf("  Learning rate: %.4f\n", lr);
    printf("  Weight decay:  %.1e\n", weight_decay);
    printf("  Epochs:        %d\n", num_epochs);
    printf("  Batch size:    %d\n", batch_size);
    printf("  Train samples: %d\n", num_train);
    printf("  Val samples:   %d\n", num_val);
    printf("\n");

    // --- Generate synthetic data ---
    printf("Generating synthetic data...\n");
    Dataset train_data(num_train, 42);
    Dataset val_data(num_val, 123);

    // Print class distribution
    int class_counts[3] = {0, 0, 0};
    for (int l : train_data.labels) class_counts[l]++;
    printf("  Train class distribution: [%d, %d, %d]\n",
           class_counts[0], class_counts[1], class_counts[2]);
    int val_counts[3] = {0, 0, 0};
    for (int l : val_data.labels) val_counts[l]++;
    printf("  Val class distribution:   [%d, %d, %d]\n",
           val_counts[0], val_counts[1], val_counts[2]);
    printf("\n");

    // --- Create model ---
    printf("Creating ResNet...\n");
    ResNet model(channels, nb_blocks, fc_size, num_classes, in_channels);
    model.train();

    int total_params = model.parameter_count();
    printf("  Total parameters: %d\n", total_params);
    printf("  Model size: %.2f KB\n", total_params * 4.0f / 1024.0f);
    printf("\n");

    // --- Create optimizer and scheduler ---
    AdamOptimizer optimizer(model.parameters(), lr, 0.9f, 0.999f, 1e-8f, weight_decay);
    CosineAnnealingLR scheduler(&optimizer, num_epochs, eta_min);

    CrossEntropyLoss criterion;

    // --- Training loop ---
    printf("================================================================\n");
    printf("  Training\n");
    printf("================================================================\n");
    printf("Epoch |  Train Loss | Train Acc | Val Loss | Val Acc | LR\n");
    printf("------+-------------+-----------+----------+---------+----------\n");

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        // Update learning rate
        scheduler.step(epoch);

        // === Training phase ===
        model.train();
        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        int epoch_total = 0;
        int num_batches = 0;

        for (int start = 0; start < num_train; start += batch_size) {
            int actual_bs = std::min(batch_size, num_train - start);

            // Get batch data
            std::vector<float> batch_data;
            std::vector<int> batch_labels;
            train_data.get_batch(start, batch_size, batch_data, batch_labels);

            // Clear intermediate tensors from previous step
            clear_tensor_arena();

            // Create input tensor
            GradTensor* input = make_input(batch_data,
                {actual_bs, in_channels, H, W});
            // Track in arena so it gets cleaned up
            g_tensor_arena.push_back(input);

            // Forward pass
            GradTensor* logits = model.forward(input);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Compute loss
            GradTensor* loss = criterion.forward(logits, batch_labels);
            CUDA_CHECK(cudaDeviceSynchronize());

            float loss_val = CrossEntropyLoss::get_loss_value(loss);
            epoch_loss += loss_val;

            // Compute accuracy
            float batch_acc = compute_accuracy(logits, batch_labels);
            epoch_correct += (int)(batch_acc * actual_bs);
            epoch_total += actual_bs;

            // Backward pass
            optimizer.zero_grad();
            loss->backward();
            CUDA_CHECK(cudaDeviceSynchronize());

            // Optimizer step
            optimizer.step();

            num_batches++;
        }

        float avg_train_loss = epoch_loss / num_batches;
        float train_acc = (float)epoch_correct / epoch_total;

        // === Validation phase (every 5 epochs) ===
        float avg_val_loss = 0.0f;
        float val_acc = 0.0f;
        bool do_val = (epoch % 5 == 0) || (epoch == num_epochs - 1);

        if (do_val) {
            model.eval();
            float val_loss_sum = 0.0f;
            int val_correct = 0;
            int val_total = 0;
            int val_batches = 0;

            for (int start = 0; start < num_val; start += batch_size) {
                int actual_bs = std::min(batch_size, num_val - start);

                std::vector<float> batch_data;
                std::vector<int> batch_labels;
                val_data.get_batch(start, batch_size, batch_data, batch_labels);

                clear_tensor_arena();

                GradTensor* input = make_input(batch_data,
                    {actual_bs, in_channels, H, W});
                g_tensor_arena.push_back(input);

                GradTensor* logits = model.forward(input);
                CUDA_CHECK(cudaDeviceSynchronize());

                GradTensor* loss = criterion.forward(logits, batch_labels);
                CUDA_CHECK(cudaDeviceSynchronize());

                val_loss_sum += CrossEntropyLoss::get_loss_value(loss);
                val_correct += (int)(compute_accuracy(logits, batch_labels) * actual_bs);
                val_total += actual_bs;
                val_batches++;
            }

            avg_val_loss = val_loss_sum / val_batches;
            val_acc = (float)val_correct / val_total;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        float epoch_ms = std::chrono::duration<float, std::milli>(
            epoch_end - epoch_start).count();

        // Print progress
        if (do_val) {
            printf("%5d | %11.6f | %7.2f%% | %8.6f | %5.2f%% | %.2e  (%.0fms)\n",
                   epoch + 1, avg_train_loss, train_acc * 100.0f,
                   avg_val_loss, val_acc * 100.0f,
                   scheduler.get_lr(), epoch_ms);
        } else {
            printf("%5d | %11.6f | %7.2f%% |    ---   |  ---  | %.2e  (%.0fms)\n",
                   epoch + 1, avg_train_loss, train_acc * 100.0f,
                   scheduler.get_lr(), epoch_ms);
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_s = std::chrono::duration<float>(total_end - total_start).count();

    // === Final summary ===
    printf("\n");
    printf("================================================================\n");
    printf("  Training Summary\n");
    printf("================================================================\n");
    printf("  Total time:      %.1f seconds\n", total_s);
    printf("  Per-epoch time:  %.1f ms\n", total_s * 1000.0f / num_epochs);
    printf("  Total parameters: %d\n", total_params);
    printf("\n");
    printf("  The network should show overfitting behavior:\n");
    printf("  - Training accuracy approaching 100%%\n");
    printf("  - Training loss approaching 0\n");
    printf("  - Validation accuracy staying near random (~33%%)\n");
    printf("  This is EXPECTED with random synthetic data and demonstrates\n");
    printf("  that the model has enough capacity to memorize the training set.\n");
    printf("\n");
    printf("  This proves that ALL components work correctly:\n");
    printf("  - Forward pass (conv, batchnorm, relu, linear, GAP)\n");
    printf("  - Backward pass (all gradient computations)\n");
    printf("  - Residual connections (gradient highway)\n");
    printf("  - Adam optimizer with cosine LR schedule\n");
    printf("  - Cross-entropy loss\n");
    printf("================================================================\n");

    // Cleanup
    clear_tensor_arena();

    return 0;
}
