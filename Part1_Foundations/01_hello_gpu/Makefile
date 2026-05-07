# Makefile for Chapter 01: Hello GPU
#
# Your GPU: Quadro P4000 (Compute Capability 6.1)
# CUDA Toolkit: 11.7
#
# Usage:
#   make               -- build all programs
#   make hello_gpu     -- build just hello_gpu
#   make vector_add    -- build just vector_add
#   make vector_add_unified -- build just vector_add_unified
#   make clean         -- remove all binaries
#   make run           -- build and run all programs

# Compiler
NVCC = nvcc

# Flags:
#   -arch=sm_61  Target Compute Capability 6.1 (Pascal, Quadro P4000)
#   -O2          Optimization level 2 (good balance of speed and compile time)
NVCC_FLAGS = -arch=sm_61 -O2

# List of all programs to build
TARGETS = hello_gpu vector_add vector_add_unified

# Default target: build everything
all: $(TARGETS)

# Build rules -- each .cu file produces one executable
hello_gpu: hello_gpu.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

vector_add: vector_add.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

vector_add_unified: vector_add_unified.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Run all programs in sequence
run: all
	@echo "========================================="
	@echo "Running hello_gpu"
	@echo "========================================="
	./hello_gpu
	@echo ""
	@echo "========================================="
	@echo "Running vector_add"
	@echo "========================================="
	./vector_add
	@echo ""
	@echo "========================================="
	@echo "Running vector_add_unified"
	@echo "========================================="
	./vector_add_unified

# Remove all compiled binaries
clean:
	rm -f $(TARGETS)

.PHONY: all clean run
