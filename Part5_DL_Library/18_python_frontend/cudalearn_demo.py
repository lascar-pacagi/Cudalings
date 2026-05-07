#!/usr/bin/env python3
"""
cudalearn_demo.py -- Full demonstration of the cudalearn Python API.

This script shows how to use our custom CUDA deep learning library from Python.
The API mirrors PyTorch as closely as possible, so every line has a PyTorch
equivalent shown in the comments.

What this demo does:
    1. Creates GradTensors (our GPU tensors) and inspects their shapes
    2. Converts between numpy arrays and GPU tensors
    3. Builds a small MLP (multi-layer perceptron) for classification
    4. Generates synthetic training data (4-quadrant classification)
    5. Runs a full training loop: forward -> loss -> backward -> step
    6. Prints the loss at each epoch to show convergence
    7. Converts results back to numpy for inspection

Build the extension first:
    make                              (using the Makefile)
    python setup.py build_ext --inplace  (using setuptools)

Then run:
    python cudalearn_demo.py
"""

import sys
import os
import numpy as np

# =============================================================================
# Step 0: Import cudalearn (with helpful error if not built yet)
# =============================================================================
# The compiled extension is a .so file in the same directory as this script.
# We add the script's directory to sys.path so Python can find it regardless
# of where we run from.

# Add the directory containing this script to the Python module search path.
# This ensures `import cudalearn` works even if we run the script from
# a different working directory.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    import cudalearn
    # PyTorch equivalent: import torch
except ImportError as e:
    print("=" * 60)
    print("ERROR: Could not import 'cudalearn'.")
    print()
    print("The C++ extension module has not been built yet.")
    print("Build it first with one of these commands:")
    print()
    print("  cd", script_dir)
    print("  make                                # using Makefile")
    print("  python setup.py build_ext --inplace # using setuptools")
    print()
    print("Original error:", e)
    print("=" * 60)
    sys.exit(1)


# =============================================================================
# Step 1: Basic tensor operations
# =============================================================================
# Demonstrate creating tensors, checking shapes, and converting to/from numpy.

print("=" * 60)
print("  cudalearn Python API Demo")
print("=" * 60)
print()

# ---- Create tensors with factory functions ----

# cudalearn.randn(d0, d1, d2, d3) -- random normal values on GPU
# PyTorch equivalent: x = torch.randn(4, 3, 8, 8)
x = cudalearn.randn(4, 3, 8, 8)
print(f"1. Random tensor: {x}")
print(f"   shape = {x.shape}, size = {x.size}, ndim = {x.ndim}")
# PyTorch equivalent: print(x.shape, x.numel(), x.dim())

# cudalearn.zeros(d0, d1) -- all-zeros tensor on GPU
# PyTorch equivalent: z = torch.zeros(10, 5)
z = cudalearn.zeros(10, 5)
print(f"2. Zeros tensor:  {z}")

# cudalearn.ones(d0) -- all-ones tensor on GPU
# PyTorch equivalent: o = torch.ones(3)
o = cudalearn.ones(3)
print(f"3. Ones tensor:   {o}")

# ---- Convert numpy -> GradTensor (CPU -> GPU) ----
# cudalearn.from_numpy(arr) copies a numpy array to GPU memory
# PyTorch equivalent: t = torch.from_numpy(arr).cuda()
arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
t = cudalearn.from_numpy(arr)
print(f"4. From numpy:    {t}")

# ---- Convert GradTensor -> numpy (GPU -> CPU) ----
# tensor.to_numpy() copies GPU data back to a numpy array
# PyTorch equivalent: arr_back = t.cpu().numpy()
arr_back = t.to_numpy()
print(f"5. Back to numpy: shape={arr_back.shape}, values=\n{arr_back}")
print()


# =============================================================================
# Step 2: Create a neural network model
# =============================================================================
# We build a 2-layer MLP (Multi-Layer Perceptron) for a 4-class quadrant
# classification task. The network architecture:
#
#   Input (2 features) --> Linear(2, 32) --> ReLU --> Linear(32, 4) --> Output
#
# This is a simple but complete example showing:
#   - Sequential container (like nn.Sequential)
#   - Linear layers (like nn.Linear)
#   - ReLU activation (like nn.ReLU)

print("-" * 60)
print("Building model...")
print("-" * 60)

# Create a Sequential container -- chains layers in order
# PyTorch equivalent: model = nn.Sequential()
model = cudalearn.Sequential()

# Add layers with names (like PyTorch's add_module)
# PyTorch equivalent: model.add_module('fc1', nn.Linear(2, 32))
model.add("fc1",  cudalearn.Linear(2, 32))

# PyTorch equivalent: model.add_module('relu1', nn.ReLU())
model.add("relu1", cudalearn.ReLU())

# PyTorch equivalent: model.add_module('fc2', nn.Linear(32, 4))
model.add("fc2",  cudalearn.Linear(32, 4))

# Print the model architecture
# PyTorch equivalent: print(model)
print(model)
print()

# Count parameters
# PyTorch equivalent: sum(p.numel() for p in model.parameters())
params = model.parameters()
total_params = sum(p.size for p in params)
print(f"Total parameters: {total_params}")
# fc1: 2*32 weights + 32 biases = 96
# fc2: 32*4 weights + 4 biases = 132
# Total: 228
print(f"  fc1: {2*32} weights + 32 biases = {2*32 + 32}")
print(f"  fc2: {32*4} weights + 4 biases = {32*4 + 4}")
print()


# =============================================================================
# Step 3: Generate synthetic training data
# =============================================================================
# Task: classify 2D points into 4 quadrants.
#
#        Q1 (label=1)  |  Q0 (label=0)
#       (x<0, y>0)     |  (x>0, y>0)
#   --------------------+--------------------
#       Q2 (label=2)   |  Q3 (label=3)
#       (x<0, y<0)     |  (x>0, y<0)
#
# This is a simple non-linear classification problem that a neural network
# can learn easily. We use it because:
#   - It is easy to visualize and verify
#   - A linear model cannot solve it (need at least one hidden layer)
#   - Small enough to train in a few hundred epochs

print("-" * 60)
print("Generating training data (4-quadrant classification)...")
print("-" * 60)

np.random.seed(42)  # For reproducibility

# Number of training samples
N = 64

# Generate random 2D points in [-2, 2] x [-2, 2]
# Shape: [N, 2] -- N points, each with 2 features (x, y)
X_np = (np.random.rand(N, 2).astype(np.float32) - 0.5) * 4.0

# Assign quadrant labels based on sign of (x, y)
# Label 0: x>0, y>0 (top-right)
# Label 1: x<0, y>0 (top-left)
# Label 2: x<0, y<0 (bottom-left)
# Label 3: x>0, y<0 (bottom-right)
labels_np = np.zeros(N, dtype=np.int32)
for i in range(N):
    if X_np[i, 0] > 0 and X_np[i, 1] > 0:
        labels_np[i] = 0
    elif X_np[i, 0] < 0 and X_np[i, 1] > 0:
        labels_np[i] = 1
    elif X_np[i, 0] < 0 and X_np[i, 1] < 0:
        labels_np[i] = 2
    else:
        labels_np[i] = 3

print(f"  X shape:      {X_np.shape}  (N={N} points, 2 features each)")
print(f"  Labels shape: {labels_np.shape}  (4 classes: quadrants)")
print(f"  Label counts: {[int((labels_np == c).sum()) for c in range(4)]}")
print()

# Convert training data from numpy (CPU) to GradTensor (GPU)
# PyTorch equivalent: X = torch.from_numpy(X_np).cuda()
X_gpu = cudalearn.from_numpy(X_np)
print(f"  X on GPU: {X_gpu}")
# Labels stay as numpy -- CrossEntropyLoss.forward() accepts numpy int32
# PyTorch equivalent: labels = torch.from_numpy(labels_np).cuda()
print()


# =============================================================================
# Step 4: Set up optimizer and loss function
# =============================================================================

# Create Adam optimizer with learning rate 0.01
# PyTorch equivalent: optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = cudalearn.Adam(model.parameters(), lr=0.01)

# Create cross-entropy loss function
# PyTorch equivalent: criterion = nn.CrossEntropyLoss()
criterion = cudalearn.CrossEntropyLoss()

print("-" * 60)
print("Optimizer: Adam (lr=0.01)")
print("Loss:      CrossEntropyLoss")
print("-" * 60)
print()


# =============================================================================
# Step 5: Training loop
# =============================================================================
# This is the core deep learning training pattern:
#   1. Forward pass: compute predictions
#   2. Compute loss: compare predictions to ground truth
#   3. Backward pass: compute gradients via automatic differentiation
#   4. Optimizer step: update parameters using gradients
#   5. Zero gradients: reset for the next iteration
#
# This pattern is IDENTICAL to PyTorch -- only the module name differs.

print("=" * 60)
print("  Training Loop")
print("=" * 60)
print(f"{'Epoch':>6s}  {'Loss':>10s}")
print("-" * 20)

num_epochs = 200
losses = []  # Track loss values to verify convergence

for epoch in range(num_epochs):

    # ---- Step 5a: Forward pass ----
    # Pass input through the model to get predictions (logits)
    # PyTorch equivalent: logits = model(X)  (or model.forward(X))
    logits = model.forward(X_gpu)

    # ---- Step 5b: Compute loss ----
    # Compare predictions to ground truth labels
    # PyTorch equivalent: loss = criterion(logits, labels)
    loss = criterion.forward(logits, labels_np)

    # ---- Step 5c: Backward pass ----
    # Compute gradients of loss with respect to all parameters
    # This traverses the computation graph built during the forward pass
    # PyTorch equivalent: loss.backward()
    loss.backward()

    # ---- Step 5d: Optimizer step ----
    # Update all parameters using their gradients (Adam update rule)
    # PyTorch equivalent: optimizer.step()
    optimizer.step()

    # ---- Step 5e: Zero gradients ----
    # Reset gradients to zero for the next iteration.
    # Without this, gradients would accumulate across iterations.
    # PyTorch equivalent: optimizer.zero_grad()
    optimizer.zero_grad()

    # ---- Logging ----
    # Extract the scalar loss value from the GPU tensor
    # PyTorch equivalent: loss_val = loss.item()
    loss_val = loss.item()
    losses.append(loss_val)

    # Print every 20 epochs to avoid flooding the terminal
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        print(f"{epoch:>6d}  {loss_val:>10.4f}")

print("-" * 20)
print()


# =============================================================================
# Step 6: Verify training results
# =============================================================================

print("=" * 60)
print("  Training Results")
print("=" * 60)

# Check that loss decreased over training
initial_loss = losses[0]
final_loss = losses[-1]
print(f"  Initial loss: {initial_loss:.4f}")
print(f"  Final loss:   {final_loss:.4f}")
print(f"  Reduction:    {initial_loss - final_loss:.4f} "
      f"({(1 - final_loss/initial_loss)*100:.1f}%)")
print()

if final_loss < initial_loss:
    print("  [OK] Loss decreased -- training is working!")
else:
    print("  [!!] Loss did not decrease -- something may be wrong.")
print()

# ---- Run inference and check accuracy ----
# Forward pass one more time to get final predictions
# PyTorch equivalent: with torch.no_grad(): logits = model(X)
final_logits = model.forward(X_gpu)

# Convert logits from GPU to numpy for analysis
# PyTorch equivalent: logits_np = logits.cpu().numpy()
logits_np = final_logits.to_numpy()

# Get predicted classes by taking argmax of logits
# PyTorch equivalent: preds = logits.argmax(dim=1)
preds = np.argmax(logits_np, axis=1)

# Compute accuracy
# PyTorch equivalent: accuracy = (preds == labels).float().mean().item()
accuracy = np.mean(preds == labels_np)
print(f"  Accuracy: {accuracy*100:.1f}% ({int(accuracy*N)}/{N} correct)")
print()

# Show a few sample predictions
print("  Sample predictions (first 10):")
print(f"    True labels:  {labels_np[:10].tolist()}")
print(f"    Predictions:  {preds[:10].tolist()}")
print()


# =============================================================================
# Step 7: Demonstrate other API features
# =============================================================================

print("=" * 60)
print("  Additional API Features")
print("=" * 60)
print()

# ---- item() on a scalar tensor ----
# PyTorch equivalent: scalar_val = loss.item()
print(f"  loss.item() = {loss.item():.4f}")

# ---- len() on a tensor ----
# PyTorch equivalent: batch_size = len(X)   (or X.shape[0])
print(f"  len(X_gpu) = {len(X_gpu)}  (batch size)")

# ---- Inspect parameter shapes ----
print(f"\n  Model parameters:")
for i, p in enumerate(model.parameters()):
    print(f"    param[{i}]: {p}")

# ---- Module repr ----
print(f"\n  Model architecture:")
print(f"  {model}")

# ---- Gradient inspection ----
# After backward(), parameters have gradients we can inspect
# PyTorch equivalent: print(list(model.parameters())[0].grad)
# Note: we already called zero_grad(), so gradients are zero.
# Let's do one more forward-backward to show non-zero gradients.
logits = model.forward(X_gpu)
loss = criterion.forward(logits, labels_np)
loss.backward()

first_param = model.parameters()[0]
grad_np = first_param.grad_to_numpy()
print(f"\n  Gradient of first parameter (fc1.weight):")
print(f"    shape: {grad_np.shape}")
print(f"    mean:  {grad_np.mean():.6f}")
print(f"    std:   {grad_np.std():.6f}")

# Clean up the extra backward pass
optimizer.zero_grad()

print()
print("=" * 60)
print("  Demo complete!")
print("  Every operation above ran on the GPU using our custom")
print("  CUDA kernels, exposed to Python via pybind11.")
print("=" * 60)
