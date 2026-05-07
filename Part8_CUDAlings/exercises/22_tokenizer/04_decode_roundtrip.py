"""CUDAlings 22.04 -- Encode then decode must reproduce the input.

A tokenizer that doesn't roundtrip is a tokenizer with a bug. The
property to verify: for any input string s,
    decode(encode(s)) == s

Goal: implement decode() given the merges -> tokens table. We provide
encode() (chapter 22.02). For each token id, look up the byte sequence
it expands to (recursive expansion via the inverse of merges).
"""

# I AM NOT DONE


def merge_pair(ids, pair, new_id):
    out = []
    i, n = 0, len(ids)
    while i < n:
        if i + 1 < n and (ids[i], ids[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        best = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if best not in merges: break
        ids = merge_pair(ids, best, merges[best])
    return ids


def decode(ids, vocab):
    """vocab[i] = bytes that token i expands to."""
    # TODO: concatenate vocab lookups into bytes and decode them as UTF-8
    return ""


if __name__ == "__main__":
    # Build vocab matching merges
    merges = {(97, 98): 256, (256, 99): 257}
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = vocab[97] + vocab[98]                 # "ab"
    vocab[257] = vocab[256] + vocab[99]                # "abc"

    text = "abc abc"
    ids = encode(text, merges)
    out = decode(ids, vocab)
    print("ok" if out == text else f"FAIL {out!r}")
