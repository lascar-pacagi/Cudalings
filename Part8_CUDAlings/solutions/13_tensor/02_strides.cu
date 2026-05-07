#include <cstdio>
int offset3d(int b, int t, int e, int sb, int st, int se) {
    return b*sb + t*st + e*se;
}
int main() {
    int data[64];
    for (int i = 0; i < 64; ++i) data[i] = i;
    int sum = 0;
    for (int i = 0; i < 4; ++i) sum += data[offset3d(i, i, i, 16, 4, 1)];
    printf("diag=%d\n", sum);
    return 0;
}
