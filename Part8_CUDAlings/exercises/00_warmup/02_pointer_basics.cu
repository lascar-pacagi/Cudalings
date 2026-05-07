// CUDAlings 00.02 — Pointers + manual host allocation
//
// Before cudaMalloc, make sure plain C-style memory feels natural.
// Allocate an int array of size N=8 on the host, fill it with [1..8],
// sum it, print. Free at the end.
//
// Why this matters: every CUDA program is a host program first.
// cudaMalloc / cudaFree mirror malloc / free; if pointer math is fuzzy,
// device memory will be much fuzzier.



#include <cstdio>
#include <cstdlib>

int main() {
    int N = 8;
    int* a = nullptr;
    // TODO: allocate `a` on the heap so it can hold N ints
    a = (int*)malloc(N * sizeof(int));
    // TODO: fill a[0..N-1] with 1..N
    for (int i = 0; i < N; i++) {
        a[i] = i + 1;
    }
    int sum = 0;
    // TODO: sum the array into `sum`
    for (int i = 0; i < N; i++) {
        sum += a[i];
    }
    printf("sum=%d\n", sum);     // expected 36
    // TODO: release the heap allocation
    free(a);
    return 0;
}
