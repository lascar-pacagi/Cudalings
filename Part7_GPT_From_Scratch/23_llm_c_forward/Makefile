# Makefile for chapter 23: llm.c-style GPT-2 forward in pure CUDA.
NVCC      = nvcc
NVCC_FLAGS = -arch=sm_61 -O2 -std=c++17 -Xcompiler -Wall

TARGETS = gpt2_forward

all: $(TARGETS)

gpt2_forward: gpt2_forward.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: gpt2_forward
	./gpt2_forward

clean:
	rm -f $(TARGETS)

.PHONY: all run clean
