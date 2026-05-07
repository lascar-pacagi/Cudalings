// CUDAlings 00.01 — Just print "hello" from main()
//
// Before we touch the GPU at all, confirm your toolchain works. nvcc is
// happy to compile a .cu file that contains zero CUDA code -- the result
// is a normal host-only binary.
//
// Goal: make the program print exactly the line "hello".



#include <cstdio>

int main() {
    // TODO: print the string "hello" followed by a newline
    printf("hello\n");
    return 0;
}
