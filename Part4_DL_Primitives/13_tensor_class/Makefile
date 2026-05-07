# ===========================================================================
# Chapter 13: The Tensor Class -- Makefile
# ===========================================================================
# Target: Quadro P4200 (Compute Capability 6.1, Pascal architecture)
# CUDA 11.7, using g++-11 as host compiler
#
# Note: We link -lcurand because Tensor::randn() uses the cuRAND library
# for GPU random number generation. cuRAND generates random numbers
# entirely on the GPU, much faster than generating on CPU and transferring.
# ===========================================================================

NVCC        := nvcc
NVCC_FLAGS  := -arch=sm_61 -O2 -lineinfo -ccbin g++-11

# Link against cuRAND (CUDA Random Number Generation library)
LIBS        := -lcurand

# All targets
TARGETS     := tensor_test

all: $(TARGETS)

# ---------------------------------------------------------------------------
# tensor_test.cu -- Comprehensive test suite for the Tensor class
# ---------------------------------------------------------------------------
# This depends on tensor.cuh (header), tensor.cu (included by header),
# and tensor_ops.cuh (operations). Since tensor.cu is #included by
# tensor.cuh (template class pattern), we only compile tensor_test.cu.
# ---------------------------------------------------------------------------
tensor_test: tensor_test.cu tensor.cuh tensor.cu tensor_ops.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ tensor_test.cu $(LIBS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
