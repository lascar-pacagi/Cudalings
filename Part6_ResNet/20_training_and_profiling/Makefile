# ==============================================================================
# Makefile -- Chapter 20: Training, Profiling, and the Full Journey
#
# The FINAL chapter of the CUDA Deep Learning Course.
#
# Targets:
#   all             - Build everything
#   train_mnist     - Train ResNet on MNIST (needs MNIST data files)
#   profile_kernels - Profile each kernel individually
#   optimization_demo - Compare conv2d implementations
#   course_summary  - Print course summary and run benchmark
#   clean           - Remove binaries
#
# Hardware: Quadro P4200 (Compute Capability 6.1), CUDA 11.7
# ==============================================================================

# Compiler and flags
NVCC        = nvcc
ARCH        = -arch=sm_61
CCBIN       = -ccbin g++-11
STD         = -std=c++14
OPT         = -O2
LIBS        = -lcurand
DEBUG       = -lineinfo

# Combined flags:
#   -arch=sm_61   : Target Quadro P4200 (Pascal, CC 6.1)
#   -O2           : Optimization level 2 (good balance of speed vs compile time)
#   -ccbin g++-11 : Use g++-11 as the host compiler
#   -std=c++14    : C++14 for lambda captures in autograd closures
#   -lcurand      : Link cuRAND for random number generation
#   -lineinfo     : Include source line info for profiling (nvprof/nsys)
NVCCFLAGS   = $(ARCH) $(OPT) $(CCBIN) $(STD) $(LIBS) $(DEBUG)

# The ResNet header from Chapter 19 -- all programs depend on it
RESNET_HDR  = ../19_resnet_from_scratch/resnet.cuh

# ==============================================================================
# Targets
# ==============================================================================

# Default: build everything
all: train_mnist profile_kernels optimization_demo course_summary

# Train ResNet on MNIST
# Requires MNIST data files in the current directory.
# See train_mnist.cu header for download instructions.
train_mnist: train_mnist.cu $(RESNET_HDR)
	$(NVCC) $(NVCCFLAGS) train_mnist.cu -o train_mnist

# Profile each kernel individually
# Run this to identify bottlenecks before optimizing.
profile_kernels: profile_kernels.cu $(RESNET_HDR)
	$(NVCC) $(NVCCFLAGS) profile_kernels.cu -o profile_kernels

# Conv2d optimization demo
# Compares naive, shared memory, and im2col+GEMM implementations.
optimization_demo: optimization_demo.cu $(RESNET_HDR)
	$(NVCC) $(NVCCFLAGS) optimization_demo.cu -o optimization_demo

# Course summary and celebration
# Prints what we built and runs a quick benchmark.
course_summary: course_summary.cu $(RESNET_HDR)
	$(NVCC) $(NVCCFLAGS) course_summary.cu -o course_summary

# Clean up
clean:
	rm -f train_mnist profile_kernels optimization_demo course_summary

# ==============================================================================
# Convenience targets for profiling
# ==============================================================================

# Run nvprof on the profiling program
nvprof: profile_kernels
	nvprof ./profile_kernels

# Run nvprof with detailed GPU trace
nvprof-trace: profile_kernels
	nvprof --print-gpu-trace ./profile_kernels

# Run nvprof with memory bandwidth metrics
nvprof-bw: profile_kernels
	nvprof --metrics gld_throughput,gst_throughput,dram_read_throughput,dram_write_throughput ./profile_kernels

# Run nvprof with occupancy metrics
nvprof-occ: profile_kernels
	nvprof --metrics achieved_occupancy,sm_efficiency ./profile_kernels

# Run nsys (Nsight Systems) profiling
nsys: profile_kernels
	nsys profile --stats=true -o ch20_profile ./profile_kernels

# ==============================================================================
# Help
# ==============================================================================

help:
	@echo "Chapter 20: Training, Profiling, and the Full Journey"
	@echo ""
	@echo "Build targets:"
	@echo "  make all               - Build all programs"
	@echo "  make train_mnist       - Build MNIST training program"
	@echo "  make profile_kernels   - Build kernel profiler"
	@echo "  make optimization_demo - Build conv2d optimization demo"
	@echo "  make course_summary    - Build course summary program"
	@echo "  make clean             - Remove all binaries"
	@echo ""
	@echo "Profiling targets:"
	@echo "  make nvprof            - Run nvprof on profile_kernels"
	@echo "  make nvprof-trace      - Run nvprof with GPU trace"
	@echo "  make nvprof-bw         - Run nvprof with bandwidth metrics"
	@echo "  make nvprof-occ        - Run nvprof with occupancy metrics"
	@echo "  make nsys              - Run Nsight Systems profiling"
	@echo ""
	@echo "Before running train_mnist, download MNIST data:"
	@echo "  wget https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
	@echo "  wget https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
	@echo "  wget https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
	@echo "  wget https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
	@echo "  gunzip *.gz"

.PHONY: all clean help nvprof nvprof-trace nvprof-bw nvprof-occ nsys
