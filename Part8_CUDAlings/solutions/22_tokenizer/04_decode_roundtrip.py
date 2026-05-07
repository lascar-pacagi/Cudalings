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
    b = b"".join(vocab[i] for i in ids)
    return b.decode("utf-8", errors="replace")
if __name__ == "__main__":
    merges = {(97, 98): 256, (256, 99): 257}
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = vocab[97] + vocab[98]
    vocab[257] = vocab[256] + vocab[99]
    text = "abc abc"
    ids = encode(text, merges)
    out = decode(ids, vocab)
    print("ok" if out == text else f"FAIL {out!r}")
