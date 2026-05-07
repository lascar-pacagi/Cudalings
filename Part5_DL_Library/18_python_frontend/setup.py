"""
setup.py -- Build script for the cudalearn Python extension module.

This script compiles bindings.cu (our pybind11 CUDA bindings) into a Python
extension module (.so shared library) that can be imported with:

    import cudalearn

Build with:
    python setup.py build_ext --inplace

What happens during the build:
    1. setuptools discovers bindings.cu as a source file
    2. We invoke nvcc (the NVIDIA CUDA compiler) instead of gcc/g++
    3. nvcc compiles the .cu file with pybind11 and Python include paths
    4. The result is linked with -lcurand and produces cudalearn.cpython-*.so
    5. --inplace puts the .so in the current directory (not in build/)

Why not use setuptools' built-in CExtension?
    Because setuptools does not know how to compile .cu files. CUDA code
    requires nvcc, which handles __global__ kernels, <<<>>> syntax, and
    device code compilation. We customize the build_ext command to call
    nvcc directly with the right flags.

Dependencies:
    - CUDA toolkit (nvcc, cuda headers, libcurand)
    - pybind11 (pip install pybind11)
    - Python development headers
    - g++-11 (host compiler for nvcc)
"""

import os
import sys
import subprocess
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


# =============================================================================
# Step 1: Locate CUDA toolkit
# =============================================================================
# We need to find:
#   - nvcc:           the CUDA compiler (compiles .cu files)
#   - CUDA includes:  cuda_runtime.h, curand.h, etc.
#   - CUDA libraries: libcurand.so (for random number generation)
#
# Strategy: check CUDA_HOME env var, then common install locations.
# On this system, CUDA is installed via conda in the miniconda3 prefix.

def find_cuda():
    """
    Locate the CUDA toolkit installation directory.

    Returns:
        dict with keys 'home', 'nvcc', 'include', 'lib64'

    Search order:
        1. CUDA_HOME or CUDA_PATH environment variable
        2. Directory containing the nvcc binary found on PATH
        3. Common system paths: /usr/local/cuda, /usr/local/cuda-11.7
    """
    # -----------------------------------------------------------------
    # Try environment variables first (most explicit way to specify)
    # -----------------------------------------------------------------
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')

    if cuda_home is None:
        # -----------------------------------------------------------------
        # Try to find nvcc on PATH and derive CUDA_HOME from it.
        # If nvcc is at /path/to/bin/nvcc, then CUDA_HOME = /path/to
        # -----------------------------------------------------------------
        try:
            nvcc_path = subprocess.check_output(
                ['which', 'nvcc'], stderr=subprocess.DEVNULL
            ).decode().strip()
            # nvcc is at <CUDA_HOME>/bin/nvcc, go up two levels
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if cuda_home is None:
        # -----------------------------------------------------------------
        # Fall back to common system installation paths
        # -----------------------------------------------------------------
        candidates = [
            '/usr/local/cuda',
            '/usr/local/cuda-11.7',
            '/usr/local/cuda-11',
        ]
        for path in candidates:
            if os.path.isdir(path):
                cuda_home = path
                break

    if cuda_home is None:
        raise EnvironmentError(
            "Cannot find CUDA installation. Set CUDA_HOME environment variable "
            "or ensure nvcc is on your PATH.\n"
            "  export CUDA_HOME=/usr/local/cuda"
        )

    # Build the paths dict
    # 'nvcc':    the compiler binary
    # 'include': directory with cuda_runtime.h, curand.h
    # 'lib64':   directory with libcurand.so
    cuda = {
        'home':    cuda_home,
        'nvcc':    os.path.join(cuda_home, 'bin', 'nvcc'),
        'include': os.path.join(cuda_home, 'include'),
        'lib64':   os.path.join(cuda_home, 'lib64'),
    }

    # On conda installations, libraries are in lib/ not lib64/
    if not os.path.isdir(cuda['lib64']):
        cuda['lib64'] = os.path.join(cuda_home, 'lib')

    # Verify nvcc exists; if not, try the system PATH directly
    if not os.path.isfile(cuda['nvcc']):
        try:
            nvcc_path = subprocess.check_output(
                ['which', 'nvcc'], stderr=subprocess.DEVNULL
            ).decode().strip()
            cuda['nvcc'] = nvcc_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise EnvironmentError(
                f"nvcc not found at {cuda['nvcc']} or on PATH. "
                "Please install the CUDA toolkit."
            )

    return cuda


# =============================================================================
# Step 2: Get pybind11 include paths
# =============================================================================
# pybind11 is a header-only library. We need its include directory so nvcc
# can find <pybind11/pybind11.h>. The command `python3 -m pybind11 --includes`
# returns flags like: -I/path/to/python/include -I/path/to/pybind11/include
#
# We also need the Python include directory (for Python.h), which pybind11
# conveniently includes in its output.

def get_pybind11_includes():
    """
    Get pybind11 and Python include directories.

    Returns:
        list of strings: include directory paths (without -I prefix)

    Uses `python3 -m pybind11 --includes` which returns something like:
        -I/home/user/miniconda3/include/python3.10
        -I/home/user/miniconda3/lib/python3.10/site-packages/pybind11/include
    """
    try:
        # Run `python -m pybind11 --includes` to get the include flags
        output = subprocess.check_output(
            [sys.executable, '-m', 'pybind11', '--includes'],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Parse the -I flags into a list of directories
        # Input:  "-I/path/one -I/path/two"
        # Output: ["/path/one", "/path/two"]
        includes = []
        for flag in output.split():
            if flag.startswith('-I'):
                includes.append(flag[2:])  # Strip the -I prefix
        return includes

    except (subprocess.CalledProcessError, FileNotFoundError):
        raise EnvironmentError(
            "pybind11 not found. Install it with:\n"
            "  pip install pybind11\n"
            "Then re-run: python setup.py build_ext --inplace"
        )


# =============================================================================
# Step 3: Custom build_ext command that uses nvcc
# =============================================================================
# The default build_ext uses the system C compiler (gcc/g++). We override it
# to use nvcc for .cu files. This is necessary because:
#   - .cu files contain CUDA kernel code (__global__ functions)
#   - Only nvcc understands <<<blocks, threads>>> launch syntax
#   - nvcc splits code into device code (GPU) and host code (CPU)
#   - nvcc calls the host compiler (g++-11) for the CPU portions

class NvccBuildExt(build_ext):
    """
    Custom build_ext that compiles .cu files with nvcc.

    Instead of using the default compiler (gcc), we invoke nvcc directly
    with all necessary flags for CUDA compilation, pybind11 compatibility,
    and shared library generation.
    """

    def build_extensions(self):
        """
        Override the build process to use nvcc for .cu source files.

        For each extension, we:
            1. Collect all include directories (CUDA, pybind11, Python)
            2. Build the nvcc command with appropriate flags
            3. Compile to a shared library (.so)
        """
        # Locate CUDA and pybind11
        cuda = find_cuda()
        pybind_includes = get_pybind11_includes()

        for ext in self.extensions:
            # Determine the output filename
            # e.g., cudalearn.cpython-310-x86_64-linux-gnu.so
            ext_filename = self.get_ext_filename(ext.name)
            ext_fullpath = os.path.join(
                self.build_lib if not self.inplace else os.path.dirname(__file__) or '.',
                ext_filename
            )

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(ext_fullpath) or '.', exist_ok=True)

            # =================================================================
            # Build the nvcc command
            # =================================================================
            # The command structure is:
            #   nvcc [arch flags] [compiler flags] [includes] [sources]
            #        -o output.so [linker flags] [libraries]
            cmd = [cuda['nvcc']]

            # ----- GPU architecture -----
            # -arch=sm_61: compile for Compute Capability 6.1 (Quadro P4200)
            # This generates PTX and SASS code for the Pascal architecture.
            # Using the exact CC of the target GPU gives optimal performance.
            cmd += ['-arch=sm_61']

            # ----- Host compiler -----
            # -ccbin g++-11: use g++ 11 as the host compiler
            # nvcc compiles device code itself, but delegates host code to g++.
            # We pin to g++-11 for compatibility with CUDA 11.7.
            cmd += ['-ccbin', 'g++-11']

            # ----- Shared library flags -----
            # -shared:  produce a shared library (.so), not an executable
            # -Xcompiler -fPIC: pass -fPIC to the host compiler (g++)
            #   -fPIC = Position Independent Code, required for shared libraries
            #   -Xcompiler passes the next flag through to g++ (not nvcc itself)
            cmd += ['-shared', '-Xcompiler', '-fPIC']

            # ----- C++ standard -----
            # -std=c++14: use C++14 standard
            # pybind11 requires at least C++11; C++14 provides useful extras
            cmd += ['-std=c++14']

            # ----- Include directories -----
            # We need includes for:
            #   1. CUDA headers (cuda_runtime.h, curand.h)
            #   2. pybind11 headers (pybind11/pybind11.h)
            #   3. Python headers (Python.h)
            #   4. Our library headers (../17_library_architecture/)
            for inc in pybind_includes:
                cmd += ['-I', inc]
            cmd += ['-I', cuda['include']]

            # Also add the parent directory so #include "../17_library_architecture/..."
            # resolves correctly
            src_dir = os.path.dirname(os.path.abspath(ext.sources[0]))
            cmd += ['-I', src_dir]

            # ----- Source files -----
            cmd += ext.sources

            # ----- Output file -----
            cmd += ['-o', ext_fullpath]

            # ----- Library directories and libraries -----
            # -L: add library search path
            # -lcurand: link against the cuRAND library (GPU random numbers)
            cmd += ['-L', cuda['lib64']]
            cmd += ['-lcurand']

            # ----- Suppress warnings -----
            # pybind11 generates some harmless warnings with nvcc; quiet them
            cmd += ['--expt-relaxed-constexpr']

            # ----- Display the command for transparency -----
            print("\n" + "=" * 70)
            print("NVCC COMPILE COMMAND:")
            print("=" * 70)
            print(' '.join(cmd))
            print("=" * 70 + "\n")

            # ----- Execute the compilation -----
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"nvcc compilation failed with exit code {e.returncode}.\n"
                    "Check the error messages above for details."
                ) from e

            print(f"\nSuccessfully built: {ext_fullpath}")
            print(f"You can now: import cudalearn")


# =============================================================================
# Step 4: Define the extension module
# =============================================================================
# We define a single extension module called "cudalearn" with bindings.cu
# as its source file. The include_dirs and libraries specified here are
# mostly informational -- the actual compilation is handled by our custom
# NvccBuildExt above, which sets all flags directly on the nvcc command line.

# Get the directory of this setup.py
HERE = os.path.dirname(os.path.abspath(__file__))

# The source file for our extension
sources = [os.path.join(HERE, 'bindings.cu')]

# Locate CUDA for library paths
try:
    cuda_info = find_cuda()
    cuda_lib_dir = cuda_info['lib64']
    cuda_include_dir = cuda_info['include']
except EnvironmentError:
    cuda_lib_dir = '/usr/local/cuda/lib64'
    cuda_include_dir = '/usr/local/cuda/include'

cudalearn_ext = Extension(
    name='cudalearn',                     # Module name (import cudalearn)
    sources=sources,                      # Source files to compile
    include_dirs=[cuda_include_dir],      # Include paths (supplementary)
    library_dirs=[cuda_lib_dir],          # Library search paths
    libraries=['curand'],                 # Libraries to link (-lcurand)
    language='c++',                       # Treat as C++ (for setuptools metadata)
)


# =============================================================================
# Step 5: Call setup()
# =============================================================================
# This is the main entry point for setuptools. It defines the package
# metadata and triggers the build when invoked with:
#   python setup.py build_ext --inplace

setup(
    # ---- Package metadata ----
    name='cudalearn',
    version='0.1.0',
    description='CUDA deep learning library with a PyTorch-like Python API',

    # ---- Extension module(s) to build ----
    ext_modules=[cudalearn_ext],

    # ---- Use our custom build command that calls nvcc ----
    cmdclass={
        'build_ext': NvccBuildExt,
    },

    # ---- Python version requirement ----
    python_requires='>=3.8',

    # ---- Build dependencies ----
    setup_requires=['pybind11>=2.6'],
    install_requires=['pybind11>=2.6', 'numpy'],
)
