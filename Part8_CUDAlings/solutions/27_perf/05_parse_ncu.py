import re
NCU_OUTPUT = """
==PROF== Connected to process 12345
    dram__bytes.sum.per_second                            145.32 GB/s
    sm__inst_executed_pipe_fma_op.sum                       8.42M
    smsp__warps_active.avg.pct_of_peak_sustained_active     62.1 %
"""
def parse_metrics(text):
    out = {}
    m = re.search(r"dram__bytes\.sum\.per_second\s+([\d\.]+)\s*GB/s", text)
    if m: out["bw_gbps"] = float(m.group(1))
    m = re.search(r"smsp__warps_active.*?pct_of_peak_sustained_active\s+([\d\.]+)\s*%", text)
    if m: out["occ_pct"] = float(m.group(1))
    return out
if __name__ == "__main__":
    m = parse_metrics(NCU_OUTPUT)
    ok = (abs(m.get("bw_gbps", 0) - 145.32) < 1e-2
          and abs(m.get("occ_pct", 0) - 62.1) < 1e-2)
    print("ok" if ok else f"FAIL {m}")
