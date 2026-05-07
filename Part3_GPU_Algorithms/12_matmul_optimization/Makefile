# ===========================================================================
# Chapter 12: Matrix Multiplication Optimization -- Makefile
# ===========================================================================
# Target: Quadro P4200 (Compute Capability 6.1, Pascal architecture)
# CUDA 11.7, using g++-11 as host compiler
# Links with cuBLAS for reference comparison
# ===========================================================================

NVCC        := nvcc
NVCC_FLAGS  := -arch=sm_61 -O2 -lineinfo -ccbin g++-11
LIBS        := -lcublas

# All targets
TARGETS     := matmul_naive matmul_optimized matmul_double_buffer

all: $(TARGETS)

# ---------------------------------------------------------------------------
# matmul_naive.cu -- Kernels 1-3: naive, coalesced, shared memory tiled
# ---------------------------------------------------------------------------
matmul_naive: matmul_naive.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# matmul_optimized.cu -- Kernels 4-6: 1D tiling, 2D tiling, vectorized loads
# ---------------------------------------------------------------------------
matmul_optimized: matmul_optimized.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# matmul_double_buffer.cu -- Kernel 7: double-buffered prefetching
# ---------------------------------------------------------------------------
matmul_double_buffer: matmul_double_buffer.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS)

.PHONY: all clean
