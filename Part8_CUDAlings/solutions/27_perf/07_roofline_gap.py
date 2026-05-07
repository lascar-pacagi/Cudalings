def roofline_pct(achieved_flops, ai, peak_bw, peak_flops):
    peak_for_kernel = min(peak_bw * ai, peak_flops)
    return 100.0 * achieved_flops / peak_for_kernel
if __name__ == "__main__":
    pct = roofline_pct(12e9, 0.083, 192e9, 5.3e12)
    print("ok" if 70.0 < pct < 80.0 else f"FAIL pct={pct:.1f}")
