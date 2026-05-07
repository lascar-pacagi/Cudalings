#include <cstdio>
#include <cuda_runtime.h>
int main() {
    printf("cublas note: feed B,A,C with row-major buffers, swap shapes (N,M,K)\n");
    return 0;
}
