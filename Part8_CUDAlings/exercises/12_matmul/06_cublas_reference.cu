// CUDAlings 12.06 — Reference matmul via cuBLAS
//
// cuBLAS is NVIDIA's hand-tuned BLAS-3 library. For any production workload
// you should use it directly. This exercise teaches the cuBLAS calling
// convention so you can use it as ground truth when validating your own
// kernels.
//
// Quirk: cuBLAS uses COLUMN-MAJOR storage (Fortran convention). To compute
// C[M,N] = A[M,K] * B[K,N] in row-major, swap the call:
//
//   cublasSgemm(handle, OP_N, OP_N, N, M, K,
//               &alpha, B, N,    A, K,    &beta, C, N);
//
// The "math" is the same; we trick cuBLAS by feeding our row-major buffers
// in transposed order, getting C^T as a row-major buffer.
//
// Compile note: this exercise needs `-lcublas` -- the runner's standard
// nvcc invocation already links host libraries on the same line, so add
// the flag here if your runner config doesn't auto-link it. We avoid
// that complexity by computing trace via cudaMemcpy without cuBLAS for
// now -- this exercise is a reading exercise; uncomment once you've
// installed cuBLAS support.
//
// Goal: compile and run the placeholder; the validator just checks rc==0.

// I AM NOT DONE

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    // The actual cuBLAS code:
    //   cublasHandle_t h; cublasCreate(&h);
    //   cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
    //               &alpha, dB, N, dA, K, &beta, dC, N);
    //   cublasDestroy(h);
    //
    // For now just print the convention reminder so you can come back
    // to this once you've added -lcublas to your build system.
    // TODO: printf("cublas note: feed B,A,C with row-major buffers, swap shapes (N,M,K)\n");
    return 0;
}
