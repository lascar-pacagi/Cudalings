/*******************************************************************************
 * cudalearn.cuh — Master Header for the cudalearn Library
 *
 * This is the single-include header for the entire cudalearn deep learning
 * library. Users only need to write:
 *
 *   #include "cudalearn.cuh"
 *
 * to get access to all components:
 *
 *   1. module.cuh     — GradTensor (autograd) + Module base class
 *   2. layers.cuh     — Conv2d, BatchNorm2d, ReLU, Linear, GlobalAvgPool2d, Sequential
 *   3. optimizer.cuh  — SGD (with momentum), Adam, CosineAnnealingLR scheduler
 *   4. loss.cuh       — CrossEntropyLoss, MSELoss
 *   5. dataloader.cuh — DataLoader with shuffling and pinned memory
 *
 * Include order matters:
 *   - module.cuh defines GradTensor and Module (the foundation everything builds on)
 *   - layers.cuh defines layer classes that inherit from Module
 *   - optimizer.cuh uses GradTensor* from module.cuh for parameter updates
 *   - loss.cuh creates GradTensors with backward_fn for gradient computation
 *   - dataloader.cuh creates GradTensors to hold batch data
 *
 * Example usage:
 *
 *   #include "cudalearn.cuh"
 *
 *   int main() {
 *       // Build model
 *       Sequential* model = new Sequential();
 *       model->add("fc1", new Linear(784, 128));
 *       model->add("relu", new ReLU());
 *       model->add("fc2", new Linear(128, 10));
 *
 *       // Optimizer and loss
 *       auto params = model->parameters();
 *       Adam optimizer(params, 0.001f);
 *       CrossEntropyLoss criterion;
 *
 *       // Training loop
 *       for (int epoch = 0; epoch < 10; epoch++) {
 *           GradTensor* logits = model->forward(input);
 *           GradTensor* loss = criterion.forward(logits, labels);
 *           loss->backward();
 *           optimizer.step();
 *           optimizer.zero_grad();
 *       }
 *
 *       delete model;
 *       return 0;
 *   }
 ******************************************************************************/

#ifndef CUDALEARN_CUH
#define CUDALEARN_CUH

// --- Foundation: GradTensor struct and Module base class ---
// GradTensor is the fundamental data type — a GPU tensor with optional
// gradient tracking and autograd backward functions.
// Module is the abstract base class for all neural network layers.
#include "module.cuh"

// --- Layers: Neural network building blocks ---
// Conv2d, BatchNorm2d, ReLU, Linear, GlobalAvgPool2d, Sequential.
// All inherit from Module and implement forward() with autograd support.
#include "layers.cuh"

// --- Optimizers: Parameter update algorithms ---
// SGD (with momentum + weight decay), Adam, CosineAnnealingLR scheduler.
// These take model.parameters() and apply gradient-based updates.
#include "optimizer.cuh"

// --- Loss functions: Training objectives ---
// CrossEntropyLoss (softmax + NLL for classification), MSELoss (regression).
// Return scalar GradTensors whose backward() triggers gradient computation.
#include "loss.cuh"

// --- DataLoader: Mini-batch data feeding ---
// Handles batch slicing, shuffling, and fast CPU-to-GPU transfer using
// pinned memory.
#include "dataloader.cuh"

#endif // CUDALEARN_CUH
