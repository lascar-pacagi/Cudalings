def merge(ids, pair, new_id):
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i + 1 < n and (ids[i], ids[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out
if __name__ == "__main__":
    ids = [1, 2, 3, 1, 2, 4, 1, 2]
    out = merge(ids, (1, 2), 99)
    print("ok" if out == [99, 3, 99, 4, 99] else f"FAIL {out}")
