"""CUDAlings 22.03 -- GPT-2's regex pretokenization.

Before BPE, GPT-2 splits text into chunks using this regex (loosely):
    contractions ('s, 't, 'd, ...) | letters | numbers | punctuation | whitespace

The ACTUAL OpenAI regex is gnarly with Unicode categories. This exercise
uses a simplified ASCII-only version. The trick: pretokenization prevents
BPE from learning merges that cross word boundaries, which keeps the
tokenizer aligned with linguistic units.

Goal: split "Hello, world! How are you?" into pretokens. The expected
list has 11 entries (each word, each punctuation mark, each space).
"""

# I AM NOT DONE

import re

PAT = re.compile(r"""
    '(?:s|t|re|ve|m|ll|d) |   # contractions
    \ ?[A-Za-z]+          |   # words optionally preceded by space
    \ ?[0-9]+             |   # numbers
    \ ?[^\sA-Za-z0-9]+    |   # punctuation chunks
    \s+                       # whitespace
""", re.VERBOSE)


def pretokenize(s):
    # TODO: return all non-overlapping matches of PAT in `s`
    return []


if __name__ == "__main__":
    out = pretokenize("Hello, world!")
    # Expected:        "Hello"  ","  " world"  "!"
    expected = ["Hello", ",", " world", "!"]
    print("ok" if out == expected else f"FAIL {out}")
