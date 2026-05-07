def merge_pair(ids, pair, new_id):
    out = []
    i, n = 0, len(ids)
    while i < n:
        if i + 1 < n and (ids[i], ids[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out
def bpe_encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pairs = set(zip(ids, ids[1:]))
        best = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if best not in merges: break
        ids = merge_pair(ids, best, merges[best])
    return ids
if __name__ == "__main__":
    merges = {(97, 98): 256, (256, 99): 257}
    out = bpe_encode("abc", merges)
    print("ok" if out == [257] else f"FAIL {out}")
