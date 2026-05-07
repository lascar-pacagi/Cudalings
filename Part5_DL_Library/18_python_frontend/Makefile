# ==============================================================================
# Makefile -- Build the cudalearn Python extension module using nvcc directly
# ==============================================================================
#
# Usage:
#   make          Build the cudalearn shared library (.so)
#   make test     Build and run the demo script
#   make clean    Remove build artifacts
#
# This Makefile compiles bindings.cu into a shared library that Python can
# import as a module. It is an alternative to setup.py for those who prefer
# direct control over the build process.
#
# The output filename must match what Python expects:
#   cudalearn.<EXT_SUFFIX>
# where EXT_SUFFIX is something like .cpython-310-x86_64-linux-gnu.so
# This is queried from Python's sysconfig module at build time.
#
# Prerequisites:
#   - CUDA toolkit (nvcc) on PATH
#   - pybind11 installed (pip install pybind11)
#   - g++-11 (host compiler for nvcc)
#   - Python 3.x with development headers
# ==============================================================================


# ==============================================================================
# Compiler and architecture settings
# ==============================================================================

# NVCC: the NVIDIA CUDA compiler
# Compiles .cu files containing both host (CPU) and device (GPU) code
NVCC := nvcc

# GPU architecture flag:
# -arch=sm_61 targets Compute Capability 6.1 (Pascal architecture)
# This matches the Quadro P4200 GPU on this system.
# sm_61 generates both PTX (portable) and SASS (native) code for this GPU.
#
# Common architecture values:
#   sm_50: Maxwell (GTX 750, 9xx)
#   sm_61: Pascal  (GTX 10xx, Quadro P-series)  <-- our target
#   sm_70: Volta   (V100)
#   sm_75: Turing  (RTX 20xx)
#   sm_80: Ampere  (A100, RTX 30xx)
#   sm_86: Ampere  (RTX 30xx laptop)
#   sm_89: Ada     (RTX 40xx)
#   sm_90: Hopper  (H100)
ARCH := -arch=sm_61

# Host compiler: nvcc delegates host (CPU) code compilation to this compiler.
# We use g++-11 for compatibility with CUDA 11.7.
# If your system uses a different g++ version, change this accordingly.
HOSTCC := -ccbin g++-11


# ==============================================================================
# Python and pybind11 configuration (auto-detected)
# ==============================================================================

# Python executable -- used to query sysconfig and pybind11
PYTHON := python3

# PYBIND11_INCLUDES: include paths for pybind11 and Python headers
# Running `python3 -m pybind11 --includes` produces flags like:
#   -I/path/to/python/include -I/path/to/pybind11/include
# These are needed so nvcc can find:
#   <pybind11/pybind11.h>  -- the binding library header
#   <Python.h>             -- the CPython API header
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

# EXT_SUFFIX: the platform-specific extension for Python modules
# On Linux with CPython 3.10, this is: .cpython-310-x86_64-linux-gnu.so
# This ensures Python's import machinery recognizes our .so file.
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Module name: the Python module name (import cudalearn)
MODULE_NAME := cudalearn

# Output filename: combines module name with the platform suffix
# e.g., cudalearn.cpython-310-x86_64-linux-gnu.so
TARGET := $(MODULE_NAME)$(EXT_SUFFIX)


# ==============================================================================
# CUDA paths
# ==============================================================================

# CUDA_HOME: root of the CUDA toolkit installation
# We try to auto-detect by finding nvcc on PATH and going up two directories.
# Override with: make CUDA_HOME=/usr/local/cuda
CUDA_HOME ?= $(shell dirname $$(dirname $$(which nvcc)))

# CUDA include directory (for cuda_runtime.h, curand.h, etc.)
CUDA_INCLUDE := -I$(CUDA_HOME)/include

# CUDA library directory (for -lcurand)
# conda installs put libs in lib/ not lib64/, so we check both
CUDA_LIB_DIR := $(shell [ -d "$(CUDA_HOME)/lib64" ] && echo "$(CUDA_HOME)/lib64" || echo "$(CUDA_HOME)/lib")
CUDA_LIBS := -L$(CUDA_LIB_DIR) -lcurand


# ==============================================================================
# Compiler flags
# ==============================================================================

# -shared:           produce a shared library (.so), not an executable
# -Xcompiler -fPIC:  pass -fPIC (Position Independent Code) to the host compiler
#                     Required for all code that goes into a shared library.
#                     -Xcompiler is nvcc's way of forwarding flags to g++.
# -std=c++14:        use C++14 standard (pybind11 needs at least C++11)
# --expt-relaxed-constexpr: allow constexpr functions in device code
#                     pybind11 headers use constexpr that nvcc is strict about;
#                     this flag relaxes those restrictions.
NVCCFLAGS := -shared -Xcompiler -fPIC -std=c++14 --expt-relaxed-constexpr

# Source file
SRC := bindings.cu


# ==============================================================================
# Build rules
# ==============================================================================

# Default target: build the shared library
# The .PHONY declaration tells make that 'all' is not a file -- it is always
# a valid target to run, even if a file named 'all' existed.
.PHONY: all test clean info

all: $(TARGET)
	@echo ""
	@echo "=========================================="
	@echo "  Build successful!"
	@echo "  Output: $(TARGET)"
	@echo ""
	@echo "  Usage in Python:"
	@echo "    import cudalearn"
	@echo "=========================================="

# Main compilation rule:
# Compile bindings.cu into the shared library using nvcc.
#
# The full command expands to something like:
#   nvcc -arch=sm_61 -ccbin g++-11 \
#        -shared -Xcompiler -fPIC -std=c++14 --expt-relaxed-constexpr \
#        -I/path/to/python/include -I/path/to/pybind11/include \
#        -I/path/to/cuda/include \
#        bindings.cu \
#        -o cudalearn.cpython-310-x86_64-linux-gnu.so \
#        -L/path/to/cuda/lib -lcurand
#
# What nvcc does internally:
#   1. Preprocesses the .cu file
#   2. Separates device code (__global__, __device__) from host code
#   3. Compiles device code to PTX, then to SASS (GPU machine code)
#   4. Compiles host code using g++-11
#   5. Links everything together into the .so shared library
$(TARGET): $(SRC)
	@echo "Compiling $(SRC) -> $(TARGET)..."
	@echo "  GPU arch:     $(ARCH)"
	@echo "  Host compiler: g++-11"
	@echo "  pybind11:     $(PYBIND11_INCLUDES)"
	@echo ""
	$(NVCC) $(ARCH) $(HOSTCC) \
		$(NVCCFLAGS) \
		$(PYBIND11_INCLUDES) \
		$(CUDA_INCLUDE) \
		$(SRC) \
		-o $(TARGET) \
		$(CUDA_LIBS)

# Test target: build and run the demo script
test: $(TARGET)
	@echo ""
	@echo "Running cudalearn_demo.py..."
	@echo "=========================================="
	$(PYTHON) cudalearn_demo.py

# Clean target: remove all build artifacts
# - $(TARGET):      the compiled .so file
# - build/:         directory created by setup.py
# - *.egg-info/:    metadata directory from setuptools
# - __pycache__/:   Python bytecode cache
# - *.o:            object files (if any intermediate compilation)
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET)
	rm -rf build/ dist/ *.egg-info
	rm -rf __pycache__
	rm -f *.o
	@echo "Clean complete."

# Info target: display detected configuration (useful for debugging)
info:
	@echo "Configuration:"
	@echo "  NVCC:             $(NVCC)"
	@echo "  ARCH:             $(ARCH)"
	@echo "  CUDA_HOME:        $(CUDA_HOME)"
	@echo "  CUDA_LIB_DIR:     $(CUDA_LIB_DIR)"
	@echo "  PYTHON:           $(PYTHON)"
	@echo "  PYBIND11_INCLUDES:$(PYBIND11_INCLUDES)"
	@echo "  EXT_SUFFIX:       $(EXT_SUFFIX)"
	@echo "  TARGET:           $(TARGET)"
