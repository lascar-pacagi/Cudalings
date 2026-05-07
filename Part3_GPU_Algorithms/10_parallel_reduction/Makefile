# ===========================================================================
# Chapter 10: Parallel Reduction -- Makefile
# ===========================================================================
# Target: Quadro P4200 (Compute Capability 6.1, Pascal architecture)
# CUDA 11.7, using g++-11 as host compiler
# ===========================================================================

NVCC        := nvcc
NVCC_FLAGS  := -arch=sm_61 -O2 -lineinfo -ccbin g++-11

# All targets
TARGETS     := reduction reduce_max multi_block_reduction

all: $(TARGETS)

# ---------------------------------------------------------------------------
# reduction.cu -- All 7 optimization levels benchmarked
# ---------------------------------------------------------------------------
reduction: reduction.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# ---------------------------------------------------------------------------
# reduce_max.cu -- Max/argmax reduction with generic template
# ---------------------------------------------------------------------------
reduce_max: reduce_max.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# ---------------------------------------------------------------------------
# multi_block_reduction.cu -- Two-pass reduction for very large arrays
# ---------------------------------------------------------------------------
multi_block_reduction: multi_block_reduction.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
