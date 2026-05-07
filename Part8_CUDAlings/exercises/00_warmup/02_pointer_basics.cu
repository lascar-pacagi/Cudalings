// CUDAlings 00.02 — Pointers + manual host allocation
//
// Before cudaMalloc, make sure plain C-style memory feels natural.
// Allocate an int array of size N=8 on the host, fill it with [1..8],
// sum it, print. Free at the end.
//
// Why this matters: every CUDA program is a host program first.
// cudaMalloc / cudaFree mirror malloc / free; if pointer math is fuzzy,
// device memory will be much fuzzier.

// I AM NOT DONE

#include <cstdio>
#include <cstdlib>

int main() {
    int N = 8;
    int* a = nullptr;
    // TODO: a = (int*)malloc(N * sizeof(int));
    // TODO: fill a[0..N-1] with 1..N
    int sum = 0;
    // TODO: sum the array into `sum`
    printf("sum=%d\n", sum);     // expected 36
    // TODO: free(a);
    return 0;
}
