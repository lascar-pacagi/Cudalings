# Makefile for Chapter 02: The CUDA Memory Model
#
# Your GPU: Quadro P4000 (Compute Capability 6.1)
# CUDA Toolkit: 11.7
#
# Usage:
#   make                -- build all programs
#   make memory_types   -- build just memory_types
#   make stencil_1d     -- build just stencil_1d
#   make bandwidth_test -- build just bandwidth_test
#   make clean          -- remove all binaries
#   make run            -- build and run all programs

# Compiler
NVCC = nvcc

# Flags:
#   -arch=sm_61  Target Compute Capability 6.1 (Pascal, Quadro P4000)
#   -O2          Optimization level 2 (good balance of speed and compile time)
NVCC_FLAGS = -arch=sm_61 -O2

# List of all programs to build
TARGETS = memory_types stencil_1d bandwidth_test

# Default target: build everything
all: $(TARGETS)

# Build rules -- each .cu file produces one executable
memory_types: memory_types.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

stencil_1d: stencil_1d.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

bandwidth_test: bandwidth_test.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run all programs in sequence
run: all
	@echo "========================================="
	@echo "Running memory_types"
	@echo "========================================="
	./memory_types
	@echo ""
	@echo "========================================="
	@echo "Running stencil_1d"
	@echo "========================================="
	./stencil_1d
	@echo ""
	@echo "========================================="
	@echo "Running bandwidth_test"
	@echo "========================================="
	./bandwidth_test

# Remove all compiled binaries
clean:
	rm -f $(TARGETS)

.PHONY: all clean run
