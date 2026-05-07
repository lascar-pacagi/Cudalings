#include <cstdio>
#include <cstdlib>
int main() {
    int N = 8;
    int* a = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) a[i] = i + 1;
    int sum = 0;
    for (int i = 0; i < N; ++i) sum += a[i];
    printf("sum=%d\n", sum);
    free(a);
    return 0;
}
