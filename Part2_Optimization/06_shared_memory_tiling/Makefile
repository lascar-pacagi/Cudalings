# Makefile for Chapter 06: Shared Memory and Tiling
#
# Your GPU: Quadro P4200 (Compute Capability 6.1, 18 SMs)
# CUDA Toolkit: 11.7
#
# Shared memory on CC 6.1:
#   - 48 KB per SM (configurable split with L1 cache)
#   - 96 KB total per SM (shared between shared mem and L1)
#   - 32 banks, 4 bytes per bank
#
# Usage:
#   make                  -- build all programs
#   make bank_conflicts   -- build bank conflicts demo
#   make tiled_transpose  -- build tiled transpose benchmark
#   make tiled_matmul_intro -- build tiled matmul preview
#   make clean            -- remove all binaries
#   make run              -- build and run all programs

# Compiler
NVCC = nvcc

# Flags:
#   -arch=sm_61   Target Compute Capability 6.1 (Pascal, Quadro P4200)
#   -O2           Optimization level 2 (good balance of speed and compile time)
#   -lineinfo     Embed source line info for profiling (nsight, nvprof)
#                 This adds NO runtime overhead but makes profiler output
#                 much more useful -- you can see which source lines
#                 cause which memory transactions.
NVCC_FLAGS = -arch=sm_61 -O2 -lineinfo -ccbin g++-11

# List of all programs to build
TARGETS = bank_conflicts tiled_transpose tiled_matmul_intro

# Default target: build everything
all: $(TARGETS)

# Build rules
bank_conflicts: bank_conflicts.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

tiled_transpose: tiled_transpose.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

tiled_matmul_intro: tiled_matmul_intro.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run all programs in sequence
run: all
	@echo "========================================="
	@echo "Running bank_conflicts"
	@echo "========================================="
	./bank_conflicts
	@echo ""
	@echo "========================================="
	@echo "Running tiled_transpose"
	@echo "========================================="
	./tiled_transpose
	@echo ""
	@echo "========================================="
	@echo "Running tiled_matmul_intro"
	@echo "========================================="
	./tiled_matmul_intro

# Remove all compiled binaries
clean:
	rm -f $(TARGETS)

.PHONY: all clean run
