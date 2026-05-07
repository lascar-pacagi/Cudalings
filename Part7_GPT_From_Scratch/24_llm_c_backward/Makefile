NVCC = nvcc
NVCC_FLAGS = -arch=sm_61 -O2 -std=c++17

all: gpt2_backward

gpt2_backward: gpt2_backward.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f gpt2_backward

.PHONY: all clean
