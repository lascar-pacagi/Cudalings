NVCC = nvcc
NVCC_FLAGS = -arch=sm_61 -O2 -std=c++17

all: adamw

adamw: adamw.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: adamw
	./adamw

clean:
	rm -f adamw

.PHONY: all run clean
