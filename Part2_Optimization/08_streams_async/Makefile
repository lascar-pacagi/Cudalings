# ══════════════════════════════════════════════════════════════════════════════
# Makefile for Chapter 08: CUDA Streams and Asynchronous Operations
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
TARGETS := streams_basics overlap_transfer event_sync

# ── Default: build all ──
all: $(TARGETS)

# ── Individual targets ──
streams_basics: streams_basics.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

overlap_transfer: overlap_transfer.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

event_sync: event_sync.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# ── Run all demos ──
run: all
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Running streams_basics"
	@echo "════════════════════════════════════════════════════════════════"
	./streams_basics
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Running overlap_transfer"
	@echo "════════════════════════════════════════════════════════════════"
	./overlap_transfer
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  Running event_sync"
	@echo "════════════════════════════════════════════════════════════════"
	./event_sync

# ── Clean ──
clean:
	rm -f $(TARGETS)

.PHONY: all clean run
