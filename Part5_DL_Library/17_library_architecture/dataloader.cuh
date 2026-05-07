/*******************************************************************************
 * dataloader.cuh — DataLoader for cudalearn
 *
 * The DataLoader handles the boring but critical job of feeding data to the
 * model in mini-batches. It manages:
 *
 *   1. Batch slicing:    splits a dataset into mini-batches of fixed size
 *   2. Shuffling:        randomizes sample order each epoch (prevents memorization)
 *   3. Memory transfer:  copies batches from CPU to GPU efficiently using
 *                         pinned (page-locked) memory for faster DMA transfers
 *
 * Design:
 *   - The dataset lives on the host (CPU) as flat float* and int* arrays
 *   - Each call to next_batch() copies one batch to the GPU
 *   - Pinned memory (cudaMallocHost) is used for the staging buffers to
 *     enable async DMA transfers between CPU and GPU
 *
 * Usage (mirrors PyTorch's DataLoader):
 *   DataLoader loader(train_data, train_labels, num_samples, sample_size,
 *                     batch_size, true);
 *
 *   for (int epoch = 0; epoch < num_epochs; epoch++) {
 *       loader.reset();  // Shuffle and start from beginning
 *       while (loader.has_next()) {
 *           auto batch = loader.next_batch();
 *           GradTensor* x = batch.first;    // [batch_size, sample_size] on GPU
 *           int* y = batch.second;           // [batch_size] on GPU
 *           // ... forward, loss, backward, step ...
 *           delete x;
 *           cudaFree(y);
 *       }
 *   }
 ******************************************************************************/

#ifndef CUDALEARN_DATALOADER_CUH
#define CUDALEARN_DATALOADER_CUH

#include "module.cuh"
#include <utility>
#include <cstdlib>
#include <ctime>

// =============================================================================
// DataLoader
// =============================================================================
// Iterates over a dataset in mini-batches with optional shuffling.
//
// Template-free design: data is float*, labels are int*. This covers the
// most common case (image classification) without template complexity.
//
// The DataLoader does NOT own the dataset arrays. The caller must keep them
// alive for the lifetime of the DataLoader.
//
// Memory layout:
//   data:   [num_samples, sample_size] — flattened, row-major
//   labels: [num_samples]              — integer class labels
//
// For image data, sample_size = C * H * W (e.g., 3 * 32 * 32 = 3072 for CIFAR).
// For tabular data, sample_size = number of features.
// =============================================================================

class DataLoader {
public:
    // ---- Dataset ----
    float* data_;           // Host pointer to the dataset (NOT owned)
    int* labels_;           // Host pointer to labels (NOT owned)
    int num_samples_;       // Total number of samples in the dataset
    int sample_size_;       // Number of floats per sample (C*H*W or num_features)

    // ---- Batching ----
    int batch_size_;        // Number of samples per mini-batch
    bool shuffle_;          // Whether to shuffle indices each epoch
    int current_idx_;       // Current position in the shuffled index array

    // ---- Shuffled indices ----
    // Instead of physically rearranging the data (expensive for large datasets),
    // we maintain an array of indices and shuffle those. When fetching a batch,
    // we use these indices to gather the right samples.
    int* indices_;          // Array of sample indices [0, 1, 2, ..., num_samples-1]

    // ---- Pinned memory staging buffers ----
    // Pinned (page-locked) memory enables faster CPU-to-GPU transfers because:
    //   1. The OS guarantees the memory won't be paged out to disk
    //   2. The GPU can use DMA (Direct Memory Access) to transfer data without
    //      involving the CPU, which is faster than pageable memory transfers
    //   3. This typically gives 2-3x speedup on PCIe transfers
    //
    // We pre-allocate these buffers once and reuse them for every batch.
    float* pinned_data_;    // Pinned buffer for one batch of data [batch_size * sample_size]
    int* pinned_labels_;    // Pinned buffer for one batch of labels [batch_size]

    // ---- Constructor ----
    // Initialize the DataLoader with dataset pointers and batching parameters.
    //
    // Args:
    //   data:        host pointer to float array [num_samples, sample_size]
    //   labels:      host pointer to int array [num_samples]
    //   num_samples: total number of samples
    //   sample_size: number of floats per sample
    //   batch_size:  mini-batch size (e.g., 32, 64, 128)
    //   shuffle:     whether to shuffle each epoch (true for training, false for eval)
    DataLoader(float* data, int* labels, int num_samples, int sample_size,
               int batch_size, bool shuffle = true)
        : data_(data), labels_(labels), num_samples_(num_samples),
          sample_size_(sample_size), batch_size_(batch_size),
          shuffle_(shuffle), current_idx_(0)
    {
        // Allocate the index array on the host.
        // This array maps logical position -> actual sample index.
        // When shuffled, it randomizes which samples appear in each batch.
        indices_ = new int[num_samples];
        for (int i = 0; i < num_samples; i++) {
            indices_[i] = i;
        }

        // Allocate pinned (page-locked) memory for staging buffers.
        // cudaMallocHost allocates host memory that is page-locked, meaning
        // the OS won't swap it to disk. This enables:
        //   - Faster cudaMemcpy (DMA-capable)
        //   - Potential for asynchronous copies with CUDA streams
        //
        // The downside: pinned memory is a limited system resource. Allocating
        // too much can degrade system performance. We only allocate enough for
        // one batch, which is typically a few MB.
        cudaMallocHost(&pinned_data_, batch_size * sample_size * sizeof(float));
        cudaMallocHost(&pinned_labels_, batch_size * sizeof(int));

        // Seed the random number generator for shuffling.
        // Using time(NULL) gives different shuffles across runs.
        // For reproducibility, you could use a fixed seed.
        srand((unsigned int)time(NULL));

        // Shuffle indices if requested (for the first epoch)
        if (shuffle_) {
            shuffle_indices();
        }
    }

    // ---- Destructor ----
    // Free the index array and pinned memory buffers.
    // We do NOT free data_ or labels_ because we don't own them.
    ~DataLoader() {
        delete[] indices_;
        cudaFreeHost(pinned_data_);
        cudaFreeHost(pinned_labels_);
    }

    // ---- shuffle_indices() ----
    // Perform a Fisher-Yates (Knuth) shuffle on the index array.
    //
    // The Fisher-Yates algorithm produces a uniformly random permutation
    // in O(n) time by swapping each element with a randomly chosen element
    // from the unshuffled portion.
    //
    // Algorithm:
    //   for i from n-1 down to 1:
    //       j = random integer in [0, i]
    //       swap indices[i] and indices[j]
    //
    // Why shuffle?
    //   - Prevents the model from memorizing the order of samples
    //   - Ensures each mini-batch is a random subset of the data
    //   - Critical for SGD convergence theory (i.i.d. assumption)
    //   - Different shuffle each epoch exposes the model to different batch compositions
    void shuffle_indices() {
        for (int i = num_samples_ - 1; i > 0; i--) {
            // Pick a random index j in [0, i]
            int j = rand() % (i + 1);

            // Swap indices[i] and indices[j]
            int tmp = indices_[i];
            indices_[i] = indices_[j];
            indices_[j] = tmp;
        }
    }

    // ---- has_next() ----
    // Check if there are more batches remaining in this epoch.
    //
    // Returns true if current_idx_ + batch_size_ <= num_samples_.
    // This means we only return full batches. The last partial batch
    // (if num_samples is not divisible by batch_size) is dropped.
    //
    // Dropping the last batch is common practice because:
    //   1. BatchNorm requires consistent batch sizes
    //   2. It simplifies tensor shape handling
    //   3. The dropped samples will appear in future epochs (with shuffling)
    bool has_next() {
        return (current_idx_ + batch_size_) <= num_samples_;
    }

    // ---- next_batch() ----
    // Fetch the next mini-batch and transfer it to the GPU.
    //
    // Returns:
    //   pair<GradTensor*, int*> where:
    //     - first:  GradTensor of shape [batch_size, sample_size] on GPU
    //     - second: GPU int array of shape [batch_size] with labels
    //
    // The caller is responsible for freeing both returned pointers:
    //   delete batch.first;   // Frees the GradTensor and its GPU data
    //   cudaFree(batch.second); // Frees the GPU labels
    //
    // Memory transfer pipeline:
    //   1. Gather samples from the dataset into pinned staging buffers
    //   2. Copy from pinned host memory to GPU memory (fast DMA transfer)
    //   3. Wrap GPU data in a GradTensor for the model to consume
    std::pair<GradTensor*, int*> next_batch() {
        // Determine the actual batch size (handles the last batch being smaller,
        // though has_next() already prevents this in the typical usage pattern)
        int actual_batch = batch_size_;
        if (current_idx_ + batch_size_ > num_samples_) {
            actual_batch = num_samples_ - current_idx_;
        }

        // --- Step 1: Gather samples into pinned staging buffers ---
        // Using the shuffled indices, copy the selected samples into contiguous
        // pinned memory. This is necessary because the shuffled samples are
        // scattered throughout the original data array.
        for (int i = 0; i < actual_batch; i++) {
            int sample_idx = indices_[current_idx_ + i];

            // Copy one sample's data: data_[sample_idx * sample_size ... + sample_size]
            // into the contiguous pinned buffer at position i
            for (int j = 0; j < sample_size_; j++) {
                pinned_data_[i * sample_size_ + j] =
                    data_[sample_idx * sample_size_ + j];
            }

            // Copy the label for this sample
            pinned_labels_[i] = labels_[sample_idx];
        }

        // --- Step 2: Allocate GPU memory and transfer ---
        // Create a GradTensor on the GPU to hold this batch's data.
        // We use [batch_size, sample_size, 1, 1] shape for linear/MLP models.
        // For CNN models, the caller should reshape to [N, C, H, W].
        GradTensor* batch_x = new GradTensor(actual_batch, sample_size_, 1, 1, false);

        // Allocate gradient storage (needed if this tensor will participate in backward)
        cudaMalloc(&batch_x->grad, batch_x->size * sizeof(float));
        cudaMemset(batch_x->grad, 0, batch_x->size * sizeof(float));

        // Copy data from pinned host memory to GPU.
        // Because we use pinned memory, this copy uses DMA and is ~2-3x faster
        // than copying from regular (pageable) host memory.
        cudaMemcpy(batch_x->data, pinned_data_,
                   actual_batch * sample_size_ * sizeof(float),
                   cudaMemcpyHostToDevice);

        // Allocate GPU memory for labels and copy
        int* batch_y = nullptr;
        cudaMalloc(&batch_y, actual_batch * sizeof(int));
        cudaMemcpy(batch_y, pinned_labels_,
                   actual_batch * sizeof(int),
                   cudaMemcpyHostToDevice);

        // Advance the current index for the next batch
        current_idx_ += actual_batch;

        return {batch_x, batch_y};
    }

    // ---- reset() ----
    // Reset the DataLoader for a new epoch.
    //
    // This resets the current position to the beginning and optionally
    // reshuffles the indices. Call this at the start of each training epoch.
    //
    // After reset(), has_next() will return true again and next_batch()
    // will start from the beginning of the (possibly reshuffled) dataset.
    void reset() {
        current_idx_ = 0;
        if (shuffle_) {
            shuffle_indices();
        }
    }

    // ---- num_batches() ----
    // Return the total number of complete batches in the dataset.
    // Useful for progress bars and logging.
    int num_batches() {
        return num_samples_ / batch_size_;
    }
};


#endif // CUDALEARN_DATALOADER_CUH
