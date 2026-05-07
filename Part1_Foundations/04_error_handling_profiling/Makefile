# ==============================================================================
# Chapter 04: Error Handling and Profiling -- Makefile
# ==============================================================================
#
# Targets:
#   make              Build all programs
#   make run          Build and run all programs
#   make errors       Build error_handling
#   make profiling    Build profiling_basics
#   make harness      Build benchmark_harness
#   make profile      Run profiling_basics under nvprof
#   make clean        Remove all binaries
#
# Hardware: Quadro P4200 (CC 6.1)
#
# Notes:
#   -lineinfo preserves source-line mapping in the binary, which allows
#   profiling tools (nvprof, nsight) to correlate metrics back to your
#   source code. Always use this flag when profiling.
#
#   -O2 enables optimizations. We keep this even for profiling so that
#   the profile reflects realistic (optimized) performance, not debug-build
#   slowdowns.
# ==============================================================================

# Compiler and flags
NVCC      = nvcc
ARCH      = -arch=sm_61          # Compute Capability 6.1 (Pascal)
OPT       = -O2                  # Optimization level
LINEINFO  = -lineinfo            # Source mapping for profiling tools
NVCCFLAGS = $(ARCH) $(OPT) $(LINEINFO)

# Source files and their corresponding binaries
SOURCES = error_handling.cu profiling_basics.cu benchmark_harness.cu
TARGETS = error_handling profiling_basics benchmark_harness

# ==============================================================================
# Default target: build everything
# ==============================================================================
all: $(TARGETS)

# ==============================================================================
# Individual build targets
# ==============================================================================

error_handling: error_handling.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

profiling_basics: profiling_basics.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

benchmark_harness: benchmark_harness.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# ==============================================================================
# Convenience aliases
# ==============================================================================
errors: error_handling
profiling: profiling_basics
harness: benchmark_harness

# ==============================================================================
# Run targets
# ==============================================================================
run: all
	@echo ""
	@echo "============================================"
	@echo "  Running error_handling"
	@echo "============================================"
	./error_handling
	@echo ""
	@echo "============================================"
	@echo "  Running profiling_basics"
	@echo "============================================"
	./profiling_basics
	@echo ""
	@echo "============================================"
	@echo "  Running benchmark_harness"
	@echo "============================================"
	./benchmark_harness

run-errors: error_handling
	./error_handling

run-profiling: profiling_basics
	./profiling_basics

run-harness: benchmark_harness
	./benchmark_harness

# ==============================================================================
# Profiling targets
# ==============================================================================
#
# These targets run the programs under nvprof with various options.
# You can also run these commands manually for more control.
#

# Quick summary of GPU activity and API calls
profile: profiling_basics
	nvprof ./profiling_basics

# Detailed trace showing every kernel launch and memory transfer
profile-trace: profiling_basics
	nvprof --print-gpu-trace ./profiling_basics

# Memory access efficiency metrics (coalescing quality)
profile-memory: profiling_basics
	nvprof --metrics gld_efficiency,gst_efficiency,gld_throughput,gst_throughput ./profiling_basics

# Occupancy and SM utilization
profile-occupancy: profiling_basics
	nvprof --metrics achieved_occupancy,sm_efficiency ./profiling_basics

# Profile the benchmark harness
profile-harness: benchmark_harness
	nvprof ./benchmark_harness

# ==============================================================================
# Debug build (with -DDEBUG for full synchronization checking)
# ==============================================================================
debug-errors: error_handling.cu
	$(NVCC) $(ARCH) -G -g -DDEBUG -o error_handling_debug $<
	./error_handling_debug
	rm -f error_handling_debug

# ==============================================================================
# Clean
# ==============================================================================
clean:
	rm -f $(TARGETS)

.PHONY: all run run-errors run-profiling run-harness clean \
        errors profiling harness \
        profile profile-trace profile-memory profile-occupancy \
        profile-harness debug-errors
