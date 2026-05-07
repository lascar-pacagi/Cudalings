# Makefile for Chapter 05: Memory Coalescing
#
# Your GPU: Quadro P4200 (Compute Capability 6.1, 18 SMs)
# CUDA Toolkit: 11.7
#
# Usage:
#   make                    -- build all programs
#   make coalescing_patterns -- build coalescing patterns benchmark
#   make aos_vs_soa         -- build AoS vs SoA comparison
#   make matrix_transpose   -- build matrix transpose case study
#   make clean              -- remove all binaries
#   make run                -- build and run all programs

# Compiler
NVCC = nvcc

# Flags:
#   -arch=sm_61   Target Compute Capability 6.1 (Pascal, Quadro P4200)
#   -O2           Optimization level 2 (good balance of speed and compile time)
#   -lineinfo     Embed source line info for profiling (nsight, nvprof)
#                 This adds NO runtime overhead but makes profiler output
#                 much more useful -- you can see which source lines
#                 cause which memory transactions.
NVCC_FLAGS = -arch=sm_61 -O3 -lineinfo

# List of all programs to build
TARGETS = coalescing_patterns aos_vs_soa matrix_transpose

# Default target: build everything
all: $(TARGETS)

# Build rules
coalescing_patterns: coalescing_patterns.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

aos_vs_soa: aos_vs_soa.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

matrix_transpose: matrix_transpose.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run all programs in sequence
run: all
	@echo "========================================="
	@echo "Running coalescing_patterns"
	@echo "========================================="
	./coalescing_patterns
	@echo ""
	@echo "========================================="
	@echo "Running aos_vs_soa"
	@echo "========================================="
	./aos_vs_soa
	@echo ""
	@echo "========================================="
	@echo "Running matrix_transpose"
	@echo "========================================="
	./matrix_transpose

# Remove all compiled binaries
clean:
	rm -f $(TARGETS)

.PHONY: all clean run
