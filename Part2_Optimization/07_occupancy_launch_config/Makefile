# ══════════════════════════════════════════════════════════════════════════════
# Makefile for Chapter 07: Occupancy and Launch Configuration
# ══════════════════════════════════════════════════════════════════════════════
#
# Target GPU: Quadro P4200 (Compute Capability 6.1, Pascal)
# CUDA: 11.7
# Note: -ccbin g++-11 is required because CUDA 11.7 does not support GCC 13+
#
# Build all:   make all
# Run all:     make run
# Clean:       make clean
# Verbose:     make verbose   (shows register usage at compile time)
# ══════════════════════════════════════════════════════════════════════════════

NVCC       := nvcc
ARCH       := -arch=sm_61
CCBIN      := -ccbin g++-11
NVCC_FLAGS := $(ARCH) -O2 -lineinfo $(CCBIN)

# Targets
TARGETS := occupancy_demo launch_config register_pressure

# ── Default: build all ──
all: $(TARGETS)

# ── Individual targets ──
occupancy_demo: occupancy_demo.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

launch_config: launch_config.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

register_pressure: register_pressure.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# ── Verbose build: show register usage at compile time ──
# -Xptxas -v makes the PTX assembler print per-kernel resource usage:
#   - Registers per thread
#   - Shared memory (static)
#   - Constant memory
#   - Local memory (spilled registers)
#
# Example output:
#   ptxas info    : Compiling entry function '_Z15kernel_low_regsPKfPfi'
#   ptxas info    : Used 16 registers, 0 bytes smem, 380 bytes cmem[0]
#
# This is invaluable for understanding register pressure without running.
verbose: NVCC_FLAGS += -Xptxas -v
verbose: clean all

# ── Run all demos ──
run: all
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Running occupancy_demo"
	@echo "════════════════════════════════════════════════════════════════"
	./occupancy_demo
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Running launch_config"
	@echo "════════════════════════════════════════════════════════════════"
	./launch_config
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Running register_pressure"
	@echo "════════════════════════════════════════════════════════════════"
	./register_pressure

# ── Clean ──
clean:
	rm -f $(TARGETS)

.PHONY: all clean run verbose
