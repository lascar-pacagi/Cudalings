# ===========================================================================
# Chapter 14: Forward-Pass Operations -- Makefile
# ===========================================================================
# Target: Quadro P4200 (Compute Capability 6.1, Pascal architecture)
# CUDA 11.7, using g++-11 as host compiler
#
# Builds four test executables:
#   conv2d_test    -- Conv2D forward pass test
#   batchnorm_test -- BatchNorm2D forward pass test
#   lrp_test       -- Linear, ReLU, Global Average Pooling, Cross-Entropy test
#   forward_test   -- Full ResNet forward pass integration test
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

TARGETS     := conv2d_test batchnorm_test lrp_test forward_test

all: $(TARGETS)

# ---------------------------------------------------------------------------
# Conv2D forward pass test
# ---------------------------------------------------------------------------
conv2d_test: conv2d_forward.cu $(TENSOR_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ conv2d_forward.cu $(LIBS)

# ---------------------------------------------------------------------------
# BatchNorm2D forward pass test
# ---------------------------------------------------------------------------
batchnorm_test: batchnorm_forward.cu $(TENSOR_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ batchnorm_forward.cu $(LIBS)

# ---------------------------------------------------------------------------
# Linear, ReLU, Global Average Pooling, Cross-Entropy test
# ---------------------------------------------------------------------------
lrp_test: linear_relu_pool.cu $(TENSOR_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ linear_relu_pool.cu $(LIBS)

# ---------------------------------------------------------------------------
# Full ResNet forward pass integration test
# ---------------------------------------------------------------------------
forward_test: forward_test.cu $(TENSOR_DEPS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ forward_test.cu $(LIBS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
