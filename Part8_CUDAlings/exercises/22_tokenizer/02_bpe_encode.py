"""CUDAlings 22.02 -- Encode a string with a learned BPE merges table.

Given:
    merges = {(a_id, b_id): new_id, ...}    # in insertion order
    text   = "..."  (UTF-8)

Encode by:
    1. ids = list(text.encode("utf-8"))    # start from raw bytes
    2. while True:
         find the (a, b) adjacent pair whose merge_index is smallest
         if no pair is in `merges`: break
         apply the merge over the entire id list

Goal: implement bpe_encode given the small merges table below.
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


def bpe_encode(text, merges):
    ids = list(text.encode("utf-8"))
    # TODO: greedily apply the earliest-learned applicable merge until none fit
    return ids


if __name__ == "__main__":
    # Hand-crafted merges:
    # bytes for 'a' = 97, 'b' = 98, 'c' = 99
    # First merge: ('a','b') -> 256
    # Second merge: (256, 'c') -> 257
    merges = {
        (97, 98):  256,        # ab
        (256, 99): 257,        # (ab)c
    }
    out = bpe_encode("abc", merges)
    # encoded "abc" should collapse to single token id 257
    print("ok" if out == [257] else f"FAIL {out}")
