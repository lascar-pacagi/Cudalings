"""CUDAlings 21.02 -- Causal mask.

Goal: produce a (T, T) lower-triangular mask matrix where positions
above the diagonal are -inf and on/below are 0. Adding this to the
attention logits before softmax kills any "look at the future" weight.
"""

# I AM NOT DONE

import torch


def causal_mask(T):
    """Return an (T, T) tensor of 0 on/below diag, -inf above."""
    # TODO: torch.triu(torch.full((T,T), float('-inf')), diagonal=1)
    return torch.zeros(T, T)


if __name__ == "__main__":
    m = causal_mask(4)
    ok = (m[0, 1] == float("-inf")
          and m[3, 3] == 0.0
          and m[1, 0] == 0.0)
    print("ok" if ok else "FAIL")
