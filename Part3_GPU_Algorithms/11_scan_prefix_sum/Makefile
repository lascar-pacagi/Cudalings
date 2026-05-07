NVCC = nvcc
NVCC_FLAGS = -arch=sm_61 -O2 -lineinfo -ccbin g++-11

all: scan_algorithms scan_large stream_compaction

scan_algorithms: scan_algorithms.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

scan_large: scan_large.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

stream_compaction: stream_compaction.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: all
	./scan_algorithms
	./scan_large
	./stream_compaction

clean:
	rm -f scan_algorithms scan_large stream_compaction

.PHONY: all run clean
