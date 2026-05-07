"""CUDAlings 22.01 -- One step of BPE merge.

Given a token-id list and a target pair, replace every adjacent
occurrence with a new id.
"""

# I AM NOT DONE


def merge(ids, pair, new_id):
    """Walk ids; replace every consecutive `pair` with `new_id`."""
    out = []
    # TODO: walk ids and emit the merged token whenever the pair matches
    return out


if __name__ == "__main__":
    ids = [1, 2, 3, 1, 2, 4, 1, 2]
    out = merge(ids, (1, 2), 99)
    expected = [99, 3, 99, 4, 99]
    print("ok" if out == expected else f"FAIL got {out}")
