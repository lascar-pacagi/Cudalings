# ==============================================================================
# Chapter 19: ResNet from Scratch — Makefile
#
# Target: Quadro P4200 (Compute Capability 6.1)
# Compiler: nvcc with g++-11 as host compiler
# ==============================================================================

NVCC      = nvcc
ARCH      = -arch=sm_61
CCBIN     = -ccbin g++-11
STD       = -std=c++14
OPT       = -O2
LIBS      = -lcurand
NVFLAGS   = $(ARCH) $(OPT) $(CCBIN) $(STD) $(LIBS)

# Targets
TEST_BIN  = resnet_test
TRAIN_BIN = resnet_train

# Header dependency
HEADER    = resnet.cuh

.PHONY: all clean run_test run_train

all: $(TEST_BIN) $(TRAIN_BIN)

$(TEST_BIN): resnet_test.cu $(HEADER)
	$(NVCC) $(NVFLAGS) resnet_test.cu -o $(TEST_BIN)

$(TRAIN_BIN): resnet_train.cu $(HEADER)
	$(NVCC) $(NVFLAGS) resnet_train.cu -o $(TRAIN_BIN)

run_test: $(TEST_BIN)
	./$(TEST_BIN)

run_train: $(TRAIN_BIN)
	./$(TRAIN_BIN)

clean:
	rm -f $(TEST_BIN) $(TRAIN_BIN)
