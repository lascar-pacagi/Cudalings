# ==============================================================================
# Makefile — Chapter 17: cudalearn Library Architecture
# ==============================================================================
#
# Build targets:
#   make              — build the library integration test
#   make run          — build and run
#   make clean        — remove compiled binary
#
# Compiler settings:
#   - NVCC:       NVIDIA CUDA Compiler
#   - arch=sm_61: Quadro P4200 (Compute Capability 6.1, Pascal architecture)
#   - ccbin:      g++-11 (host compiler for C++14 support)
#   - std=c++14:  C++14 standard (needed for auto return types, lambdas, etc.)
#   - O2:         Optimization level 2 (good balance of speed and compile time)
#   - lcurand:    Link against cuRAND library (for Kaiming initialization)
# ==============================================================================

# CUDA compiler
NVCC = nvcc

# Compilation flags:
#   -arch=sm_61:     target Quadro P4200 (Pascal, CC 6.1)
#   -O2:             optimize for speed without aggressive transformations
#   -ccbin g++-11:   use GCC 11 as the host compiler (compatible with CUDA 11.7)
#   -std=c++14:      enable C++14 features (auto, lambdas, etc.)
NVCCFLAGS = -arch=sm_61 -O2 -ccbin g++-11 -std=c++14

# Libraries to link:
#   -lcurand: NVIDIA cuRAND for random number generation (used in weight init)
LIBS = -lcurand

# Source and target
SRC = library_test.cu
TARGET = library_test

# Header dependencies — rebuild if any header changes
HEADERS = cudalearn.cuh module.cuh layers.cuh optimizer.cuh loss.cuh dataloader.cuh

# ==============================================================================
# Build rules
# ==============================================================================

# Default target: build the test binary
all: $(TARGET)

# Compile the test program
# The single .cu file includes cudalearn.cuh, which pulls in all headers.
# NVCC compiles both host (C++) and device (CUDA) code in one pass.
$(TARGET): $(SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET) $(LIBS)

# Build and run
run: $(TARGET)
	./$(TARGET)

# Remove compiled binary
clean:
	rm -f $(TARGET)

.PHONY: all run clean
