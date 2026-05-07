#include <cstdio>
int main() {
    int flops_per_element = 2;
    int bytes_per_element = 12;
    printf("AI=%.3f\n", (double)flops_per_element / bytes_per_element);
    return 0;
}
