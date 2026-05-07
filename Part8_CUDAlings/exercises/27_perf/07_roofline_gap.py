"""CUDAlings 27.07 -- Roofline gap calculation.

The roofline model says:
    peak_for_kernel = min(peak_bandwidth * AI, peak_flops)
where AI = arithmetic intensity (FLOP / byte_loaded).

The "gap" is achieved_flops / peak_for_kernel, expressed as a percentage
(0% = nothing, 100% = at the roofline). A 30%+ gap on a critical kernel
is your signal to optimize.

Goal: implement roofline_pct(achieved_flops, ai, peak_bw, peak_flops).
"""

# I AM NOT DONE


def roofline_pct(achieved_flops, ai, peak_bw, peak_flops):
    """All values in matching units (FLOPS, FLOP/byte, bytes/sec).
    Returns achieved as a percentage of the kernel's roofline ceiling."""
    # TODO: derive the kernel's roofline ceiling, then return achieved as a % of it
    return 0.0


if __name__ == "__main__":
    # Vector add: AI = 0.083 FLOP/byte. Peak BW = 192e9 B/s, peak FLOPS = 5.3e12.
    # Memory-bound regime: peak_for_kernel = 192e9 * 0.083 ≈ 1.59e10 = 15.9 GFLOPS
    # If we achieve 12 GFLOPS, that's ~75% of the kernel's ceiling.
    pct = roofline_pct(achieved_flops=12e9, ai=0.083, peak_bw=192e9, peak_flops=5.3e12)
    print("ok" if 70.0 < pct < 80.0 else f"FAIL pct={pct:.1f}")
