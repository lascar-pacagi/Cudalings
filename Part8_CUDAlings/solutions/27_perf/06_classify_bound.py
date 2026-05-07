def classify(bw_gbps, peak_bw, flops, peak_flops):
    bw_frac = bw_gbps / peak_bw
    flop_frac = flops / peak_flops
    return "memory" if bw_frac > flop_frac else "compute"
if __name__ == "__main__":
    a = classify(150, 192, 50, 5300)
    b = classify(30, 192, 4000, 5300)
    print("ok" if (a == "memory" and b == "compute") else f"FAIL {a} {b}")
