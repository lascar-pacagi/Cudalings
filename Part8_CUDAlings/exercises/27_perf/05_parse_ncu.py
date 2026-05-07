"""CUDAlings 27.05 -- Parse ncu output for two key metrics.

ncu prints metrics in this kind of format (CSV-ish, "metric: value unit"):

    dram__bytes.sum.per_second               145.32 GB/s
    sm__inst_executed_pipe_fma_op.sum         8.42M
    smsp__warps_active.avg.pct_of_peak_sustained_active   62.1 %

Goal: given the snippet below, parse out:
    - achieved memory bandwidth (GB/s)
    - achieved occupancy (%)
"""

# I AM NOT DONE

import re

NCU_OUTPUT = """
==PROF== Connected to process 12345
[... ncu kernel name banner ...]
    Section: GPU Speed Of Light
    --------------------------------------------------------------
    dram__bytes.sum.per_second                            145.32 GB/s
    sm__inst_executed_pipe_fma_op.sum                       8.42M
    smsp__warps_active.avg.pct_of_peak_sustained_active     62.1 %
"""


def parse_metrics(text):
    """Return {'bw_gbps': float, 'occ_pct': float}."""
    out = {}
    # TODO: regex for "dram__bytes.sum.per_second  <num> GB/s"  → bw_gbps
    # TODO: regex for "smsp__warps_active.*pct_of_peak_sustained_active  <num> %"  → occ_pct
    return out


if __name__ == "__main__":
    m = parse_metrics(NCU_OUTPUT)
    ok = (abs(m.get("bw_gbps", 0) - 145.32) < 1e-2
          and abs(m.get("occ_pct", 0) - 62.1) < 1e-2)
    print("ok" if ok else f"FAIL {m}")
