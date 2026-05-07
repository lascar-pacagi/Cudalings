# ══════════════════════════════════════════════════════════════════════════════
# Makefile for Chapter 09: Warp-Level Primitives
# ══════════════════════════════════════════════════════════════════════════════
#
# Target GPU: Quadro P4200 (Compute Capability 6.1, Pascal)
# CUDA: 11.7
# Note: -ccbin g++-11 is required because CUDA 11.7 does not support GCC 13+
#
# Build all:   make all
# Run all:     make run
# Clean:       make clean
# ══════════════════════════════════════════════════════════════════════════════

NVCC       := nvcc
ARCH       := -arch=sm_61
CCBIN      := -ccbin g++-11
NVCC_FLAGS := $(ARCH) -O2 -lineinfo $(CCBIN)

# Targets
TARGETS := shuffle_demo warp_reduction vote_functions

# ── Default: build all ──
all: $(TARGETS)

# ── Individual targets ──
shuffle_demo: shuffle_demo.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

warp_reduction: warp_reduction.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

vote_functions: vote_functions.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# ── Run all demos ──
run: all
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Running shuffle_demo"
	@echo "════════════════════════════════════════════════════════════════"
	./shuffle_demo
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Running warp_reduction"
	@echo "════════════════════════════════════════════════════════════════"
	./warp_reduction
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Running vote_functions"
	@echo "════════════════════════════════════════════════════════════════"
	./vote_functions

# ── Clean ──
clean:
	rm -f $(TARGETS)

.PHONY: all clean run
