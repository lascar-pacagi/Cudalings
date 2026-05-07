# ===========================================================================
# Chapter 15: Backward-Pass Operations -- Makefile
# ===========================================================================
# Target: Quadro P4200 (Compute Capability 6.1, Pascal architecture)
# CUDA 11.7, using g++-11 as host compiler
#
# Builds four test executables:
#   conv2d_bwd_test      -- Conv2D backward pass test (grad_input + grad_weight)
#   batchnorm_bwd_test   -- BatchNorm2D backward pass test
#   simple_bwd_test      -- ReLU, Linear, GAP, CrossEntropy, Residual backward
#   gradient_check       -- Full end-to-end gradient checking (forward+backward chain)
#
# All files depend on the Tensor class from Chapter 13, included via
# relative path: ../13_tensor_class/tensor.cuh
# ===========================================================================

NVCC        := nvcc
NVCC_FLAGS  := -arch=sm_61 -O2 -lineinfo -ccbin g++-11

# cuRAND is needed by Tensor::randn()
LIBS        := -lcurand

# Include path so the compiler can find the Chapter 13 headers
INCLUDES    := -I../13_tensor_class

# Chapter 13 header dependencies
TENSOR_DEPS := ../13_tensor_class/tensor.cuh \
               ../13_tensor_class/tensor.cu  \
               ../13_tensor_class/tensor_ops.cuh

TARGETS     := conv2d_bwd_test batchnorm_bwd_test simple_bwd_test gradient_check

all: $(TARGETS)

# ---------------------------------------------------------------------------
# Conv2D backward pass test
# ---------------------------------------------------------------------------
conv2d_bwd_test: conv2d_backward.cu $(TENSOR_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ conv2d_backward.cu $(LIBS)

# ---------------------------------------------------------------------------
# BatchNorm2D backward pass test
# ---------------------------------------------------------------------------
batchnorm_bwd_test: batchnorm_backward.cu $(TENSOR_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ batchnorm_backward.cu $(LIBS)

# ---------------------------------------------------------------------------
# Simple backward ops test (ReLU, Linear, GAP, CrossEntropy, Residual)
# ---------------------------------------------------------------------------
simple_bwd_test: simple_backward.cu $(TENSOR_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ simple_backward.cu $(LIBS)

# ---------------------------------------------------------------------------
# Full gradient check (entire forward+backward chain)
# ---------------------------------------------------------------------------
gradient_check: gradient_check.cu $(TENSOR_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ gradient_check.cu $(LIBS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
