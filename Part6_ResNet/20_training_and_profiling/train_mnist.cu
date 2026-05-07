/*******************************************************************************
 * train_mnist.cu -- Train Our ResNet on MNIST
 *
 * Chapter 20: Training, Profiling, and the Full Journey
 *
 * This program:
 *   1. Loads MNIST data from IDX binary files
 *   2. Adapts 28x28 grayscale images to (4, 8, 8) input format
 *   3. Creates a ResNet(channels=32, blocks=4, fc_size=32, classes=10)
 *   4. Trains with Adam + CosineAnnealingLR for 20 epochs
 *   5. Reports per-epoch loss/accuracy and per-phase timing
 *
 * MNIST DATA DOWNLOAD INSTRUCTIONS:
 *   The MNIST dataset consists of 4 files in IDX binary format.
 *   Download them from: https://yann.lecun.com/exdb/mnist/
 *
 *     wget https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
 *     wget https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
 *     wget https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
 *     wget https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
 *
 *   Then decompress:
 *     gunzip *.gz
 *
 *   Place the 4 uncompressed files in the same directory as this program,
 *   or set the DATA_DIR path below.
 *
 * INPUT ADAPTATION:
 *   Our ResNet expects (B, 4, 8, 8) -- originally designed for board games.
 *   MNIST is (B, 1, 28, 28) grayscale.
 *
 *   We adapt by:
 *     1. Downsampling 28x28 -> 8x8 using block averaging (each 8x8 output
 *        pixel averages a ~3.5x3.5 block of input pixels)
 *     2. Replicating the single channel to 4 channels (simple but effective;
 *        the network can learn to ignore redundant channels)
 *
 * Compile: nvcc -arch=sm_61 -O2 -ccbin g++-11 -std=c++14 -lcurand train_mnist.cu -o train_mnist
 ******************************************************************************/

// Include our complete ResNet implementation from Chapter 19
#include "../19_resnet_from_scratch/resnet.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

// ============================================================================
// Configuration
// ============================================================================

// Path to MNIST data files (change if you put them elsewhere)
#ifndef DATA_DIR
#define DATA_DIR "."
#endif

// Training hyperparameters
static const int BATCH_SIZE   = 64;    // Mini-batch size
static const int NUM_EPOCHS   = 20;    // Number of training epochs
static const float LR_INIT    = 0.001f; // Initial learning rate
static const float LR_MIN     = 1e-5f;  // Minimum learning rate (cosine annealing)
static const float WEIGHT_DECAY = 1e-4f; // AdamW weight decay

// ResNet architecture -- adapted for MNIST
static const int CHANNELS     = 32;    // Feature map channels
static const int NUM_BLOCKS   = 4;     // Number of residual blocks
static const int FC_SIZE      = 32;    // Hidden layer in head
static const int NUM_CLASSES  = 10;    // MNIST has 10 digit classes
static const int IN_CHANNELS  = 4;     // Input channels (replicated grayscale)
static const int SPATIAL_SIZE = 8;     // Spatial dimension after downsampling


// ============================================================================
// MNIST IDX File Parser
// ============================================================================
//
// The IDX file format is dead simple:
//   Byte 0-3:   Magic number (big-endian)
//               0x00000801 = unsigned byte (labels)
//               0x00000803 = unsigned byte (images, 3 dims)
//   Byte 4-7:   Number of items (big-endian)
//   Byte 8-11:  Number of rows (images only, big-endian)
//   Byte 12-15: Number of columns (images only, big-endian)
//   Byte 16+:   Data (unsigned bytes, row-major)
//
// Big-endian means the most significant byte comes first.
// x86 is little-endian, so we need to swap bytes.
// ============================================================================

// Swap bytes for big-endian -> little-endian conversion.
// The IDX format stores integers in big-endian (MSB first).
// x86/x64 processors are little-endian (LSB first), so we must reverse.
static int swap_endian(int val) {
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >> 8)  |
           ((val & 0x0000FF00) << 8)  |
           ((val & 0x000000FF) << 24);
}

// Read MNIST image file (returns vector of float in [0, 1])
//
// File layout:
//   [magic=2051][count][rows=28][cols=28][pixel0][pixel1]...[pixelN]
//   Each pixel is an unsigned byte (0-255).
//   We convert to float in [0, 1] by dividing by 255.
static std::vector<float> read_mnist_images(const char* filename, int& count,
                                             int& rows, int& cols) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filename);
        fprintf(stderr, "Download MNIST from https://yann.lecun.com/exdb/mnist/\n");
        fprintf(stderr, "  wget https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\n");
        fprintf(stderr, "  gunzip train-images-idx3-ubyte.gz\n");
        exit(EXIT_FAILURE);
    }

    // Read and validate magic number
    int magic;
    fread(&magic, 4, 1, fp);
    magic = swap_endian(magic);
    if (magic != 2051) {
        fprintf(stderr, "ERROR: Invalid magic number %d (expected 2051) in %s\n",
                magic, filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Read dimensions
    fread(&count, 4, 1, fp);  count = swap_endian(count);
    fread(&rows, 4, 1, fp);   rows  = swap_endian(rows);
    fread(&cols, 4, 1, fp);   cols  = swap_endian(cols);

    printf("  Loaded %s: %d images, %dx%d\n", filename, count, rows, cols);

    // Read pixel data and convert to float [0, 1]
    int total_pixels = count * rows * cols;
    std::vector<unsigned char> raw(total_pixels);
    fread(raw.data(), 1, total_pixels, fp);
    fclose(fp);

    std::vector<float> images(total_pixels);
    for (int i = 0; i < total_pixels; i++) {
        images[i] = raw[i] / 255.0f;
    }
    return images;
}

// Read MNIST label file (returns vector of int labels 0-9)
//
// File layout:
//   [magic=2049][count][label0][label1]...[labelN]
//   Each label is a single unsigned byte (0-9).
static std::vector<int> read_mnist_labels(const char* filename, int& count) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open %s\n", filename);
        fprintf(stderr, "Download MNIST from https://yann.lecun.com/exdb/mnist/\n");
        fprintf(stderr, "  wget https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\n");
        fprintf(stderr, "  gunzip train-labels-idx1-ubyte.gz\n");
        exit(EXIT_FAILURE);
    }

    // Read and validate magic number
    int magic;
    fread(&magic, 4, 1, fp);
    magic = swap_endian(magic);
    if (magic != 2049) {
        fprintf(stderr, "ERROR: Invalid magic number %d (expected 2049) in %s\n",
                magic, filename);
        fclose(fp);
        exit(EXIT_FAILURE);
    }

    // Read count
    fread(&count, 4, 1, fp);
    count = swap_endian(count);

    printf("  Loaded %s: %d labels\n", filename, count);

    // Read labels
    std::vector<unsigned char> raw(count);
    fread(raw.data(), 1, count, fp);
    fclose(fp);

    std::vector<int> labels(count);
    for (int i = 0; i < count; i++) {
        labels[i] = (int)raw[i];
    }
    return labels;
}


// ============================================================================
// Image Preprocessing: 28x28 -> (4, 8, 8)
// ============================================================================
//
// Our ResNet expects (B, 4, 8, 8) input. MNIST provides (B, 1, 28, 28).
//
// Strategy:
//   1. Downsample 28x28 -> 8x8 using block averaging.
//      Each output pixel covers a (28/8) = 3.5 pixel region of the input.
//      We use a simple integer block approach: each output pixel averages
//      a ceil(28/8)=4 x 4 block, clamped to image boundaries.
//      This is crude but works fine for MNIST digits.
//
//   2. Replicate the grayscale channel to all 4 input channels.
//      The network will learn to use (or ignore) the redundancy.
//      This is simpler than zero-padding channels and works in practice.
//
// The result for one image is a float array of size 4 * 8 * 8 = 256.
// ============================================================================

// Downsample a single 28x28 image to 8x8 by block-averaging.
// Input: 28*28 floats (row-major).  Output: 8*8 floats.
static void downsample_28_to_8(const float* src_28x28, float* dst_8x8) {
    // For each output pixel (oy, ox) in the 8x8 grid, we average the
    // input pixels that fall in the corresponding region of the 28x28 image.
    //
    // The mapping: input region for output pixel (oy, ox) is
    //   iy_start = oy * 28 / 8,  iy_end = (oy + 1) * 28 / 8
    //   ix_start = ox * 28 / 8,  ix_end = (ox + 1) * 28 / 8
    //
    // Using integer division gives us approximate boundaries.
    for (int oy = 0; oy < 8; oy++) {
        for (int ox = 0; ox < 8; ox++) {
            // Compute input region boundaries
            int iy_start = (oy * 28) / 8;           // floor
            int iy_end   = ((oy + 1) * 28) / 8;     // floor of next
            int ix_start = (ox * 28) / 8;
            int ix_end   = ((ox + 1) * 28) / 8;

            // Average all input pixels in this block
            float sum = 0.0f;
            int count = 0;
            for (int iy = iy_start; iy < iy_end; iy++) {
                for (int ix = ix_start; ix < ix_end; ix++) {
                    sum += src_28x28[iy * 28 + ix];
                    count++;
                }
            }
            dst_8x8[oy * 8 + ox] = (count > 0) ? sum / count : 0.0f;
        }
    }
}

// Preprocess an entire dataset: (N, 1, 28, 28) -> (N, 4, 8, 8)
// Returns a flat float vector of size N * 4 * 8 * 8.
static std::vector<float> preprocess_mnist(const std::vector<float>& images_28x28,
                                            int N) {
    // Output: N images, each with 4 channels of 8x8 = 256 floats per image
    int out_size = N * IN_CHANNELS * SPATIAL_SIZE * SPATIAL_SIZE;
    std::vector<float> out(out_size);

    // Temporary buffer for one downsampled 8x8 image
    float ds[64]; // 8 * 8

    for (int n = 0; n < N; n++) {
        // Downsample this image from 28x28 to 8x8
        downsample_28_to_8(&images_28x28[n * 28 * 28], ds);

        // Replicate to all 4 channels.
        // Layout: out[n][c][h][w] = out[((n * 4 + c) * 8 + h) * 8 + w]
        // We put the same 8x8 data into each of the 4 channels.
        for (int c = 0; c < IN_CHANNELS; c++) {
            for (int hw = 0; hw < 64; hw++) {
                out[((n * IN_CHANNELS + c) * 64) + hw] = ds[hw];
            }
        }
    }
    return out;
}


// ============================================================================
// Timing Accumulators
// ============================================================================
//
// We track cumulative time (in milliseconds) for each phase of training.
// CUDA events give us accurate GPU-side timing (Chapter 7).
// ============================================================================

struct TrainTimers {
    float data_transfer_ms;  // Host-to-device data copy
    float forward_ms;        // Forward pass through the network
    float loss_ms;           // Cross-entropy loss computation
    float backward_ms;       // Backward pass (gradient computation)
    float optimizer_ms;      // Adam parameter update

    TrainTimers() : data_transfer_ms(0), forward_ms(0), loss_ms(0),
                    backward_ms(0), optimizer_ms(0) {}

    // Total time across all phases
    float total() const {
        return data_transfer_ms + forward_ms + loss_ms + backward_ms + optimizer_ms;
    }

    // Print a nice breakdown
    void print() const {
        float t = total();
        printf("\n===== TRAINING TIME BREAKDOWN =====\n");
        printf("  Data transfer: %8.1f ms  (%5.1f%%)\n", data_transfer_ms, 100.0f * data_transfer_ms / t);
        printf("  Forward pass:  %8.1f ms  (%5.1f%%)\n", forward_ms, 100.0f * forward_ms / t);
        printf("  Loss compute:  %8.1f ms  (%5.1f%%)\n", loss_ms, 100.0f * loss_ms / t);
        printf("  Backward pass: %8.1f ms  (%5.1f%%)\n", backward_ms, 100.0f * backward_ms / t);
        printf("  Optimizer:     %8.1f ms  (%5.1f%%)\n", optimizer_ms, 100.0f * optimizer_ms / t);
        printf("  --------------------------------\n");
        printf("  TOTAL:         %8.1f ms\n", t);
        printf("===================================\n");
    }
};


// ============================================================================
// Main: Training Loop
// ============================================================================

int main(int argc, char** argv) {
    printf("==========================================================\n");
    printf("  Chapter 20: Training ResNet on MNIST\n");
    printf("  The FINAL chapter of the CUDA Deep Learning Course\n");
    printf("==========================================================\n\n");

    // ---- GPU info ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  SMs: %d, Global mem: %.0f MB, Shared mem/block: %zu bytes\n",
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0),
           prop.sharedMemPerBlock);
    printf("\n");

    // ---- Load MNIST ----
    printf("Loading MNIST data...\n");

    // Construct file paths -- look in current directory by default
    char train_img_path[512], train_lbl_path[512];
    char test_img_path[512], test_lbl_path[512];
    snprintf(train_img_path, 512, "%s/train-images-idx3-ubyte", DATA_DIR);
    snprintf(train_lbl_path, 512, "%s/train-labels-idx1-ubyte", DATA_DIR);
    snprintf(test_img_path, 512, "%s/t10k-images-idx3-ubyte", DATA_DIR);
    snprintf(test_lbl_path, 512, "%s/t10k-labels-idx1-ubyte", DATA_DIR);

    int train_count, test_count, rows, cols;
    int label_count_train, label_count_test;

    std::vector<float> train_images_raw = read_mnist_images(train_img_path, train_count, rows, cols);
    std::vector<int> train_labels = read_mnist_labels(train_lbl_path, label_count_train);
    std::vector<float> test_images_raw = read_mnist_images(test_img_path, test_count, rows, cols);
    std::vector<int> test_labels = read_mnist_labels(test_lbl_path, label_count_test);

    // Sanity checks
    if (train_count != label_count_train || test_count != label_count_test) {
        fprintf(stderr, "ERROR: Image/label count mismatch\n");
        return 1;
    }

    printf("\n");

    // ---- Preprocess: 28x28 -> (4, 8, 8) ----
    printf("Preprocessing: downsampling 28x28 -> 8x8, replicating to 4 channels...\n");
    std::vector<float> train_data = preprocess_mnist(train_images_raw, train_count);
    std::vector<float> test_data  = preprocess_mnist(test_images_raw, test_count);
    printf("  Train: %d images -> (%d, %d, %d, %d)\n",
           train_count, train_count, IN_CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE);
    printf("  Test:  %d images -> (%d, %d, %d, %d)\n",
           test_count, test_count, IN_CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE);
    printf("\n");

    // ---- Create model ----
    // ResNet(channels=32, blocks=4, fc_size=32, classes=10, in_channels=4)
    //
    // This is smaller than a typical ResNet because:
    //   1. MNIST is easy (high accuracy even with tiny models)
    //   2. Our 8x8 spatial size doesn't need deep feature extraction
    //   3. Our naive kernels are slow -- a big model would take forever
    printf("Creating ResNet(channels=%d, blocks=%d, fc_size=%d, classes=%d)...\n",
           CHANNELS, NUM_BLOCKS, FC_SIZE, NUM_CLASSES);
    ResNet model(CHANNELS, NUM_BLOCKS, FC_SIZE, NUM_CLASSES, IN_CHANNELS);
    int param_count = model.parameter_count();
    printf("  Total parameters: %d (%.1f KB)\n", param_count,
           param_count * sizeof(float) / 1024.0f);
    printf("\n");

    // ---- Create optimizer and scheduler ----
    std::vector<GradTensor*> params = model.parameters();
    AdamOptimizer optimizer(params, LR_INIT, 0.9f, 0.999f, 1e-8f, WEIGHT_DECAY);
    CosineAnnealingLR scheduler(&optimizer, NUM_EPOCHS, LR_MIN);

    CrossEntropyLoss criterion;

    // ---- Create CUDA events for timing ----
    cudaEvent_t ev_start, ev_data, ev_fwd, ev_loss, ev_bwd, ev_opt;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_data);
    cudaEventCreate(&ev_fwd);
    cudaEventCreate(&ev_loss);
    cudaEventCreate(&ev_bwd);
    cudaEventCreate(&ev_opt);

    TrainTimers timers;

    // Number of batches per epoch
    int num_train_batches = train_count / BATCH_SIZE;
    int num_test_batches  = test_count / BATCH_SIZE;

    // Per-image size in the preprocessed dataset
    int img_size = IN_CHANNELS * SPATIAL_SIZE * SPATIAL_SIZE; // 4 * 8 * 8 = 256

    printf("Training: %d epochs, batch_size=%d, %d batches/epoch\n",
           NUM_EPOCHS, BATCH_SIZE, num_train_batches);
    printf("Optimizer: Adam (lr=%.4f, weight_decay=%.4f) + CosineAnnealingLR\n",
           LR_INIT, WEIGHT_DECAY);
    printf("\n");
    printf("%-6s  %-10s  %-10s  %-10s  %-10s  %-8s\n",
           "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "LR");
    printf("------  ----------  ----------  ----------  ----------  --------\n");

    // ====================================================================
    // TRAINING LOOP
    // ====================================================================
    //
    // Each epoch:
    //   1. Shuffle training data (via index permutation)
    //   2. For each mini-batch:
    //      a. Copy batch data to GPU (timed: data_transfer)
    //      b. Forward pass (timed: forward)
    //      c. Compute loss (timed: loss)
    //      d. Backward pass (timed: backward)
    //      e. Optimizer step (timed: optimizer)
    //   3. Evaluate on validation set
    //   4. Update learning rate
    // ====================================================================

    // Shuffle indices (we shuffle indices, not data, to avoid large copies)
    std::vector<int> shuffle_idx(train_count);
    std::iota(shuffle_idx.begin(), shuffle_idx.end(), 0);

    auto start_total = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        // Update learning rate via cosine schedule
        scheduler.step(epoch);

        // Shuffle training data indices
        // (Using a simple Fisher-Yates shuffle)
        for (int i = train_count - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            std::swap(shuffle_idx[i], shuffle_idx[j]);
        }

        // ---- Training phase ----
        model.train();  // Set BatchNorm to training mode

        float epoch_loss = 0.0f;
        int epoch_correct = 0;
        int epoch_total = 0;

        for (int batch = 0; batch < num_train_batches; batch++) {
            // -- Phase 1: Prepare batch data on host --
            // Extract the batch from the shuffled dataset.
            // Each image is img_size floats. Labels are ints.
            std::vector<float> batch_data(BATCH_SIZE * img_size);
            std::vector<int> batch_labels(BATCH_SIZE);

            for (int b = 0; b < BATCH_SIZE; b++) {
                int idx = shuffle_idx[batch * BATCH_SIZE + b];
                memcpy(&batch_data[b * img_size],
                       &train_data[idx * img_size],
                       img_size * sizeof(float));
                batch_labels[b] = train_labels[idx];
            }

            // -- Phase 2: Clear intermediate tensors from previous step --
            // The tensor arena holds all intermediate GradTensors created
            // during the previous forward/backward pass. We must free them
            // to avoid GPU memory exhaustion.
            clear_tensor_arena();
            optimizer.zero_grad();

            // -- Phase 3: Transfer data to GPU (timed) --
            cudaEventRecord(ev_start);

            // Create input tensor and copy batch data to GPU
            GradTensor* input = make_input(batch_data,
                                           {BATCH_SIZE, IN_CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE});

            cudaEventRecord(ev_data);

            // -- Phase 4: Forward pass (timed) --
            GradTensor* logits = model.forward(input);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEventRecord(ev_fwd);

            // -- Phase 5: Compute loss (timed) --
            GradTensor* loss = criterion.forward(logits, batch_labels);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEventRecord(ev_loss);

            // -- Phase 6: Backward pass (timed) --
            loss->backward();
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEventRecord(ev_bwd);

            // -- Phase 7: Optimizer step (timed) --
            optimizer.step();

            cudaEventRecord(ev_opt);
            cudaEventSynchronize(ev_opt);

            // -- Accumulate timing --
            float t_data, t_fwd, t_loss, t_bwd, t_opt;
            cudaEventElapsedTime(&t_data, ev_start, ev_data);
            cudaEventElapsedTime(&t_fwd,  ev_data,  ev_fwd);
            cudaEventElapsedTime(&t_loss, ev_fwd,   ev_loss);
            cudaEventElapsedTime(&t_bwd,  ev_loss,  ev_bwd);
            cudaEventElapsedTime(&t_opt,  ev_bwd,   ev_opt);

            timers.data_transfer_ms += t_data;
            timers.forward_ms      += t_fwd;
            timers.loss_ms         += t_loss;
            timers.backward_ms     += t_bwd;
            timers.optimizer_ms    += t_opt;

            // -- Track loss and accuracy --
            float loss_val = CrossEntropyLoss::get_loss_value(loss);
            epoch_loss += loss_val;

            float acc = compute_accuracy(logits, batch_labels);
            epoch_correct += (int)(acc * BATCH_SIZE);
            epoch_total += BATCH_SIZE;

            // Clean up the input tensor (it's NOT in the arena because
            // make_input doesn't add to arena -- we own it)
            delete input;
        }

        float train_loss = epoch_loss / num_train_batches;
        float train_acc  = (float)epoch_correct / epoch_total;

        // ---- Validation phase ----
        model.eval();  // Set BatchNorm to use running statistics

        float val_loss_sum = 0.0f;
        int val_correct = 0;
        int val_total = 0;

        for (int batch = 0; batch < num_test_batches; batch++) {
            // Prepare batch
            std::vector<float> batch_data(BATCH_SIZE * img_size);
            std::vector<int> batch_labels(BATCH_SIZE);

            for (int b = 0; b < BATCH_SIZE; b++) {
                int idx = batch * BATCH_SIZE + b;
                memcpy(&batch_data[b * img_size],
                       &test_data[idx * img_size],
                       img_size * sizeof(float));
                batch_labels[b] = test_labels[idx];
            }

            clear_tensor_arena();

            GradTensor* input = make_input(batch_data,
                                           {BATCH_SIZE, IN_CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE});
            GradTensor* logits = model.forward(input);
            CUDA_CHECK(cudaDeviceSynchronize());

            GradTensor* loss = criterion.forward(logits, batch_labels);
            CUDA_CHECK(cudaDeviceSynchronize());

            float loss_val = CrossEntropyLoss::get_loss_value(loss);
            val_loss_sum += loss_val;

            float acc = compute_accuracy(logits, batch_labels);
            val_correct += (int)(acc * BATCH_SIZE);
            val_total += BATCH_SIZE;

            delete input;
        }

        float val_loss = val_loss_sum / num_test_batches;
        float val_acc  = (float)val_correct / val_total;

        // Print epoch results
        printf("  %2d    %8.4f    %8.4f    %8.4f    %8.4f    %.6f\n",
               epoch + 1, train_loss, train_acc, val_loss, val_acc,
               scheduler.get_lr());
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    float total_seconds = std::chrono::duration<float>(end_total - start_total).count();

    // ---- Final results ----
    printf("\n");
    printf("==========================================================\n");
    printf("  TRAINING COMPLETE\n");
    printf("==========================================================\n");
    printf("  Total wall-clock time: %.1f seconds\n", total_seconds);
    printf("  Samples processed: %d (train) + %d (val) per epoch\n",
           num_train_batches * BATCH_SIZE, num_test_batches * BATCH_SIZE);
    printf("  Throughput: %.0f samples/second\n",
           (float)(NUM_EPOCHS * num_train_batches * BATCH_SIZE) / total_seconds);

    // Print detailed timing breakdown
    timers.print();

    printf("\n");
    printf("NOTE: This is our hand-written CUDA ResNet -- no cuDNN, no cuBLAS.\n");
    printf("PyTorch with cuDNN would be 10-50x faster, but we UNDERSTAND\n");
    printf("every single operation, from the atomicAdd in conv2d_backward\n");
    printf("to the bias correction in Adam.\n");
    printf("\n");
    printf("For profiling details, run:\n");
    printf("  nvprof ./train_mnist\n");
    printf("  nsys profile --stats=true ./train_mnist\n");

    // ---- Cleanup ----
    // Model, optimizer destructor handles GPU memory cleanup.
    // Clear any remaining arena tensors.
    clear_tensor_arena();

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_data);
    cudaEventDestroy(ev_fwd);
    cudaEventDestroy(ev_loss);
    cudaEventDestroy(ev_bwd);
    cudaEventDestroy(ev_opt);

    printf("\nDone.\n");
    return 0;
}
