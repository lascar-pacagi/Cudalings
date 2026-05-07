"""CUDAlings 27.06 -- Classify a kernel as memory-bound vs compute-bound.

Given (achieved_bw_gbps, peak_bw_gbps, achieved_flops, peak_flops),
return "memory" if the kernel is closer to its memory ceiling, else
"compute". The actual decision in production: check the section ncu
reports "Speed Of Light: Memory" vs "Speed Of Light: Compute (SM)".

Pascal P4200 peaks: ~192 GB/s, ~5300 GFLOPS (fp32).
"""

# I AM NOT DONE


def classify(bw_gbps, peak_bw, flops, peak_flops):
    bw_frac = bw_gbps / peak_bw
    flop_frac = flops / peak_flops
    # TODO: return "memory" if bw_frac > flop_frac else "compute"
    return "unknown"


if __name__ == "__main__":
    # Case 1: a copy kernel hits 150 GB/s (78% of 192) but only 50 GFLOPS
    # (~1% of 5300). Memory-bound.
    a = classify(150, 192, 50, 5300)
    # Case 2: a matmul kernel hits 30 GB/s (15%) and 4000 GFLOPS (75%).
    # Compute-bound.
    b = classify(30, 192, 4000, 5300)
    print("ok" if (a == "memory" and b == "compute") else f"FAIL {a} {b}")
