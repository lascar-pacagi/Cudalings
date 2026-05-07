// =============================================================================
// Chapter 04: Error Handling in CUDA
// =============================================================================
//
// This program demonstrates:
//   1. The CUDA_CHECK macro pattern for catching errors
//   2. Intentionally triggering different types of errors
//   3. The difference between cudaGetLastError and cudaPeekAtLastError
//   4. Debug synchronization pattern for catching execution errors
//
// Compile:  nvcc -arch=sm_61 -O2 -lineinfo -o error_handling error_handling.cu
// Run:      ./error_handling
//
// Hardware: Quadro P4200 (CC 6.1)
// =============================================================================

#include <cstdio>
#include <cstdlib>

// =============================================================================
// THE CUDA_CHECK MACRO
// =============================================================================
//
// This is the standard error-checking macro used in virtually all CUDA code.
// It wraps any cudaError_t-returning function and:
//   1. Captures the error code
//   2. Checks if it indicates failure
//   3. Prints the file, line number, and human-readable error string
//   4. Exits the program
//
// The do { ... } while(0) wrapper is a C/C++ idiom that ensures the macro
// behaves correctly when used in if/else blocks without braces:
//
//   if (condition)
//       CUDA_CHECK(cudaMalloc(...));   // Works correctly with do-while(0)
//   else
//       something_else();
//
// Without do-while(0), the semicolon after the macro could break the if/else.
//
#define CUDA_CHECK(err) do {                                           \
    cudaError_t err_ = (err);                                          \
    if (err_ != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d -- %s (%s)\n",           \
                __FILE__, __LINE__,                                    \
                cudaGetErrorString(err_),                              \
                cudaGetErrorName(err_));                                \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

// =============================================================================
// A variant macro that checks kernel launches.
//
// After a kernel launch, we need TWO checks:
//   1. cudaGetLastError() -- catches LAUNCH errors (bad config, etc.)
//   2. cudaDeviceSynchronize() -- catches EXECUTION errors (out of bounds, etc.)
//
// In production code, you might skip the sync for performance.
// During development, always sync to catch errors at the right kernel.
// =============================================================================
#define CUDA_CHECK_KERNEL() do {                                       \
    CUDA_CHECK(cudaGetLastError());                                    \
    CUDA_CHECK(cudaDeviceSynchronize());                               \
} while(0)


// =============================================================================
// Simple kernels for testing
// =============================================================================

// A correct kernel -- just doubles every element
__global__ void double_elements(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

// A kernel that will cause an execution error if bounds checking is removed.
// We use this to demonstrate catching execution-time errors.
__global__ void kernel_with_potential_error(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Correct: bounds-checked access
    if (idx < n) {
        data[idx] = data[idx] + 1.0f;
    }
}


// =============================================================================
// DEMONSTRATION 1: Basic error checking with the macro
// =============================================================================
void demo_basic_error_checking() {
    printf("==========================================================\n");
    printf("  Demo 1: Basic CUDA_CHECK Usage\n");
    printf("==========================================================\n\n");

    // Every CUDA API call returns cudaError_t. Without checking, errors
    // are silently ignored. With CUDA_CHECK, we catch them immediately.

    const int N = 1024;
    const size_t bytes = N * sizeof(float);
    float *d_data = nullptr;

    // Step 1: Allocate device memory -- CUDA_CHECK catches allocation failure
    printf("  Allocating %zu bytes on GPU...\n", bytes);
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    printf("  OK: cudaMalloc succeeded (d_data = %p)\n", d_data);

    // Step 2: Initialize to zeros
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));
    printf("  OK: cudaMemset succeeded\n");

    // Step 3: Launch kernel with proper error checking
    //
    // Note the TWO-STEP check:
    //   - cudaGetLastError() catches launch config errors (synchronous)
    //   - cudaDeviceSynchronize() catches runtime errors (asynchronous)
    //
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("  Launching kernel: %d blocks x %d threads...\n",
           blocksPerGrid, threadsPerBlock);

    double_elements<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK_KERNEL();
    printf("  OK: Kernel launched and completed successfully\n");

    // Step 4: Clean up
    CUDA_CHECK(cudaFree(d_data));
    printf("  OK: cudaFree succeeded\n\n");
}


// =============================================================================
// DEMONSTRATION 2: Catching launch configuration errors
// =============================================================================
void demo_launch_config_error() {
    printf("==========================================================\n");
    printf("  Demo 2: Catching Launch Configuration Errors\n");
    printf("==========================================================\n\n");

    // The max threads per block for CC 6.1 is 1024.
    // Launching with more threads is a LAUNCH error -- caught immediately
    // by cudaGetLastError() without needing to synchronize.

    const int N = 1024;
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    // Try launching with 2048 threads per block (max is 1024)
    printf("  Attempting to launch with 2048 threads per block...\n");
    printf("  (Max for CC 6.1 is 1024 -- this SHOULD fail)\n\n");

    double_elements<<<1, 2048>>>(d_data, N);

    // cudaGetLastError() returns the error AND clears it.
    // We check manually here instead of using CUDA_CHECK so we can
    // continue the program after the expected error.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CAUGHT (expected): %s (%s)\n\n",
               cudaGetErrorString(err), cudaGetErrorName(err));
        printf("  This is a LAUNCH error -- caught immediately by\n");
        printf("  cudaGetLastError() without needing cudaDeviceSynchronize().\n");
        printf("  The kernel never started executing on the GPU.\n\n");
    } else {
        printf("  ERROR: Should have failed but didn't!\n\n");
    }

    // IMPORTANT: After cudaGetLastError(), the error state is cleared.
    // Subsequent operations can proceed normally.
    printf("  After clearing the error, we can launch correctly:\n");
    double_elements<<<1, 256>>>(d_data, N);
    CUDA_CHECK_KERNEL();
    printf("  OK: Follow-up kernel succeeded\n\n");

    CUDA_CHECK(cudaFree(d_data));
}


// =============================================================================
// DEMONSTRATION 3: cudaGetLastError vs cudaPeekAtLastError
// =============================================================================
void demo_get_vs_peek() {
    printf("==========================================================\n");
    printf("  Demo 3: cudaGetLastError vs cudaPeekAtLastError\n");
    printf("==========================================================\n\n");

    // Trigger an error intentionally
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)));

    printf("  Triggering an error (2048 threads per block)...\n\n");
    double_elements<<<1, 2048>>>(d_data, 1024);

    // --- cudaPeekAtLastError: reads error WITHOUT clearing ---
    cudaError_t err1 = cudaPeekAtLastError();
    printf("  1st cudaPeekAtLastError(): %s\n", cudaGetErrorString(err1));

    cudaError_t err2 = cudaPeekAtLastError();
    printf("  2nd cudaPeekAtLastError(): %s\n", cudaGetErrorString(err2));
    printf("  --> Same error both times (Peek does NOT clear)\n\n");

    // --- cudaGetLastError: reads error AND clears it ---
    cudaError_t err3 = cudaGetLastError();
    printf("  1st cudaGetLastError():    %s\n", cudaGetErrorString(err3));

    cudaError_t err4 = cudaGetLastError();
    printf("  2nd cudaGetLastError():    %s\n", cudaGetErrorString(err4));
    printf("  --> Error was cleared by first call (Get DOES clear)\n\n");

    // Practical takeaway:
    // - Use cudaGetLastError() as your default (check and clear)
    // - Use cudaPeekAtLastError() when debugging layered code where
    //   multiple levels might want to inspect the same error

    CUDA_CHECK(cudaFree(d_data));
}


// =============================================================================
// DEMONSTRATION 4: Catching memory allocation failures
// =============================================================================
void demo_allocation_error() {
    printf("==========================================================\n");
    printf("  Demo 4: Catching Memory Allocation Failures\n");
    printf("==========================================================\n\n");

    // Try to allocate an absurdly large amount of memory.
    // The P4200 has 8 GB -- requesting 1 TB will fail.
    //
    // Without CUDA_CHECK, this would silently return a null pointer,
    // and any subsequent cudaMemcpy or kernel using this pointer
    // would produce mysterious errors far from the real cause.

    float *d_huge = nullptr;
    size_t huge_size = (size_t)1024 * 1024 * 1024 * 1024;  // 1 TB

    printf("  Attempting to allocate 1 TB of GPU memory...\n");
    printf("  (P4200 has 8 GB -- this SHOULD fail)\n\n");

    cudaError_t err = cudaMalloc(&d_huge, huge_size);
    if (err != cudaSuccess) {
        printf("  CAUGHT (expected): %s (%s)\n\n",
               cudaGetErrorString(err), cudaGetErrorName(err));
        printf("  Without CUDA_CHECK, d_huge would be NULL and any\n");
        printf("  subsequent operation would produce a confusing error.\n");
        printf("  With CUDA_CHECK, we catch it right at the source.\n\n");
    } else {
        // This won't happen, but if it did:
        printf("  Surprisingly succeeded! Freeing...\n");
        cudaFree(d_huge);
    }

    // Clear any lingering error state
    cudaGetLastError();
}


// =============================================================================
// DEMONSTRATION 5: Invalid device error
// =============================================================================
void demo_invalid_device() {
    printf("==========================================================\n");
    printf("  Demo 5: Catching Invalid Device Errors\n");
    printf("==========================================================\n\n");

    // Query how many CUDA devices are available
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("  System has %d CUDA device(s)\n", deviceCount);

    // Try to select a device that doesn't exist
    printf("  Attempting to select device #%d (doesn't exist)...\n\n",
           deviceCount);

    cudaError_t err = cudaSetDevice(deviceCount);  // One past the last valid
    if (err != cudaSuccess) {
        printf("  CAUGHT (expected): %s (%s)\n\n",
               cudaGetErrorString(err), cudaGetErrorName(err));
    }

    // Restore to device 0
    cudaGetLastError();  // Clear error
    CUDA_CHECK(cudaSetDevice(0));
    printf("  Restored to device 0 successfully\n\n");
}


// =============================================================================
// DEMONSTRATION 6: Error propagation -- why checking EVERY call matters
// =============================================================================
void demo_error_propagation() {
    printf("==========================================================\n");
    printf("  Demo 6: Error Propagation (Sticky Errors)\n");
    printf("==========================================================\n\n");

    // Some CUDA errors are "sticky" -- once they occur, ALL subsequent
    // CUDA calls will fail until the error is cleared with cudaGetLastError()
    // or the context is reset with cudaDeviceReset().
    //
    // This is why checking only the LAST call is not enough: the error
    // might have originated many calls earlier.

    printf("  Scenario: What happens if you DON'T check errors?\n\n");

    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)));

    // Step 1: Trigger an error (bad launch config)
    printf("  Step 1: Launch with bad config (2048 threads)...\n");
    double_elements<<<1, 2048>>>(d_data, 1024);
    // NOT checking the error here!

    // Step 2: Try another operation -- it might fail due to sticky error
    printf("  Step 2: Try cudaMemset (should work on its own)...\n");
    cudaError_t err = cudaMemset(d_data, 0, 1024 * sizeof(float));
    printf("         Result: %s\n", cudaGetErrorString(err));

    // Step 3: The error from step 1 is still in the error state
    printf("  Step 3: Check cudaGetLastError()...\n");
    err = cudaGetLastError();
    printf("         Result: %s\n\n", cudaGetErrorString(err));

    // Note: Launch configuration errors (cudaErrorInvalidConfiguration)
    // are non-sticky -- they don't corrupt the CUDA context. But other
    // errors (like illegal memory access) ARE sticky and will cause
    // ALL subsequent calls to fail.
    //
    // The lesson: always check EVERY call, so you catch errors at
    // their source rather than debugging downstream symptoms.

    printf("  LESSON: Always check errors at the point they occur.\n");
    printf("  A launch config error here is non-sticky, but illegal\n");
    printf("  memory access errors ARE sticky and will break all\n");
    printf("  subsequent CUDA calls.\n\n");

    // Clean up -- clear any error first
    cudaGetLastError();
    CUDA_CHECK(cudaFree(d_data));
}


// =============================================================================
// DEMONSTRATION 7: Debug synchronization pattern
// =============================================================================
void demo_debug_sync() {
    printf("==========================================================\n");
    printf("  Demo 7: Debug Synchronization Pattern\n");
    printf("==========================================================\n\n");

    // In production, synchronizing after every kernel kills performance.
    // But during development, it's essential for catching execution errors
    // at the exact kernel that caused them.
    //
    // The pattern: use a macro that synchronizes in debug builds only.
    //
    //   #ifdef DEBUG
    //     #define CHECK_KERNEL() do {
    //         CUDA_CHECK(cudaGetLastError());
    //         CUDA_CHECK(cudaDeviceSynchronize());
    //     } while(0)
    //   #else
    //     #define CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())
    //   #endif
    //
    // Compile with: nvcc -DDEBUG ... (for debug mode)
    // Compile with: nvcc ...          (for production mode)

    const int N = 1 << 20;  // 1M elements
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    printf("  Launching 3 kernels with full sync checking...\n");
    printf("  (In production, you'd remove the sync for speed)\n\n");

    // Kernel 1
    double_elements<<<blocks, threads>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());           // Check launch
    CUDA_CHECK(cudaDeviceSynchronize());      // Check execution
    printf("  Kernel 1: OK\n");

    // Kernel 2
    kernel_with_potential_error<<<blocks, threads>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Kernel 2: OK\n");

    // Kernel 3
    double_elements<<<blocks, threads>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Kernel 3: OK\n\n");

    printf("  All kernels completed without errors.\n");
    printf("  Without sync, errors from kernel 1 might only surface\n");
    printf("  at kernel 3 or even later, making debugging much harder.\n\n");

    CUDA_CHECK(cudaFree(d_data));
}


// =============================================================================
// MAIN
// =============================================================================
int main() {
    printf("\n");
    printf("##########################################################\n");
    printf("#                                                        #\n");
    printf("#        Chapter 04: CUDA Error Handling                 #\n");
    printf("#                                                        #\n");
    printf("##########################################################\n\n");

    // Print device info for context
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("  Total memory: %.1f GB\n\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Run all demonstrations
    demo_basic_error_checking();
    demo_launch_config_error();
    demo_get_vs_peek();
    demo_allocation_error();
    demo_invalid_device();
    demo_error_propagation();
    demo_debug_sync();

    printf("##########################################################\n");
    printf("#  All error handling demos completed successfully!      #\n");
    printf("##########################################################\n\n");

    return 0;
}
