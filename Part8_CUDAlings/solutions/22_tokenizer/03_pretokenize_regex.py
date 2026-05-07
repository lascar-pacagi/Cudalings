import re
PAT = re.compile(r"""
    '(?:s|t|re|ve|m|ll|d) |
    \ ?[A-Za-z]+          |
    \ ?[0-9]+             |
    \ ?[^\sA-Za-z0-9]+    |
    \s+
""", re.VERBOSE)
def pretokenize(s): return PAT.findall(s)
if __name__ == "__main__":
    out = pretokenize("Hello, world!")
    expected = ["Hello", ",", " world", "!"]
    print("ok" if out == expected else f"FAIL {out}")
