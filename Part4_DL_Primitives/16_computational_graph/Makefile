# ===========================================================================
# Chapter 16: Computational Graph (Autograd Engine) -- Makefile
# ===========================================================================
# Target: Quadro P4200 (Compute Capability 6.1, Pascal architecture)
# CUDA 11.7, using g++-11 as host compiler
#
# Builds two executables:
#   autograd_test  -- Gradient correctness tests (polynomial, linear, conv, residual)
#   graph_viz      -- Computational graph visualization (ASCII + DOT)
#
# This chapter is SELF-CONTAINED: all CUDA kernels are implemented directly
# in the header files (grad_tensor.cuh and autograd_ops.cuh). No dependencies
# on other chapters.
# ===========================================================================

NVCC        := nvcc
NVCC_FLAGS  := -arch=sm_61 -O2 -lineinfo -ccbin g++-11

# --extended-lambda is needed for device lambdas (not used here, but safe to include)
# -std=c++14 is needed for std::function and shared_ptr in device code
NVCC_FLAGS  += -std=c++14

# cuRAND is needed by make_parameter() for random initialization
LIBS        := -lcurand

# Header dependencies
HEADERS     := grad_tensor.cuh autograd_ops.cuh

TARGETS     := autograd_test graph_viz

all: $(TARGETS)

# ---------------------------------------------------------------------------
# Autograd test suite
# ---------------------------------------------------------------------------
autograd_test: autograd_test.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ autograd_test.cu $(LIBS)

# ---------------------------------------------------------------------------
# Computational graph visualization
# ---------------------------------------------------------------------------
graph_viz: graph_viz.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ graph_viz.cu $(LIBS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
