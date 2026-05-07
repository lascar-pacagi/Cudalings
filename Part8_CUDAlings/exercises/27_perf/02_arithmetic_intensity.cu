// CUDAlings 27.02 — Compute the arithmetic intensity of a kernel
//
// Arithmetic intensity = FLOPs / bytes_loaded_or_stored.
// Vector add (y = x + b for scalar b):
//   1 FMA per element (well, 1 ADD)        =  1 FLOP
//   8 bytes read (x, b broadcast) + 4 wr    = 12 bytes (b is constant; ignore)
//   AI = 1 / 8 = 0.125 FLOP/byte
//
// SAXPY (y = a*x + y):
//   2 FLOPs per element  (mul + add)       =  2 FLOP
//   8 bytes read (x, y) + 4 bytes write y  = 12 bytes
//   AI = 2 / 12 ≈ 0.167 FLOP/byte
//
// Goal: print "AI=0.167" for a SAXPY kernel by computing the ratio
// algebraically (no need to actually run the kernel).

// I AM NOT DONE

#include <cstdio>

int main() {
    int flops_per_element = 2;     // mul + add
    int bytes_per_element = 12;    // read x (4), read y (4), write y (4)
    // TODO: print AI = flops_per_element / bytes_per_element with 3 decimals
    printf("AI=0.000\n");
    return 0;
}
