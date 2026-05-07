# ==============================================================================
# Chapter 03: Thread Hierarchy -- Makefile
# ==============================================================================
#
# Targets:
#   make              Build all programs
#   make run          Build and run all programs
#   make indexing     Build indexing_2d
#   make divergence   Build warp_divergence
#   make blur         Build image_blur
#   make clean        Remove all binaries
#
# Hardware: Quadro P4200 (CC 6.1)
# ==============================================================================

# Compiler and flags
NVCC      = nvcc
ARCH      = -arch=sm_61          # Compute Capability 6.1 (Pascal)
OPT       = -O2                  # Optimization level
NVCCFLAGS = $(ARCH) $(OPT)

# Source files and their corresponding binaries
SOURCES = indexing_2d.cu warp_divergence.cu image_blur.cu
TARGETS = indexing_2d warp_divergence image_blur

# ==============================================================================
# Default target: build everything
# ==============================================================================
all: $(TARGETS)

# ==============================================================================
# Individual build targets
# ==============================================================================

indexing_2d: indexing_2d.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

warp_divergence: warp_divergence.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

image_blur: image_blur.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ==============================================================================
# Convenience aliases
# ==============================================================================
indexing: indexing_2d
divergence: warp_divergence
blur: image_blur

# ==============================================================================
# Run targets
# ==============================================================================
run: all
	@echo ""
	@echo "============================================"
	@echo "  Running indexing_2d"
	@echo "============================================"
	./indexing_2d
	@echo ""
	@echo "============================================"
	@echo "  Running warp_divergence"
	@echo "============================================"
	./warp_divergence
	@echo ""
	@echo "============================================"
	@echo "  Running image_blur"
	@echo "============================================"
	./image_blur

run-indexing: indexing_2d
	./indexing_2d

run-divergence: warp_divergence
	./warp_divergence

run-blur: image_blur
	./image_blur

# ==============================================================================
# Clean
# ==============================================================================
clean:
	rm -f $(TARGETS)

.PHONY: all run run-indexing run-divergence run-blur clean indexing divergence blur
