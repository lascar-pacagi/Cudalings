"""CUDAlings 21.01 -- Multi-head attention reshape.

The single most-confused step in implementing a transformer is reshaping
q/k/v from (B, T, E) to (B, H, T, Dh). Get it right once, get it right
forever.

Goal: implement reshape_to_heads(x, H) so the test below passes.
"""

# I AM NOT DONE

import torch


def reshape_to_heads(x, n_head):
    """(B, T, E) -> (B, H, T, Dh) where Dh = E // H."""
    B, T, E = x.shape
    Dh = E // n_head
    # TODO: split E into (H, Dh), then move H next to B (no copy needed)
    return x


if __name__ == "__main__":
    x = torch.arange(2 * 3 * 8, dtype=torch.float32).view(2, 3, 8)
    y = reshape_to_heads(x, n_head=2)
    expected_shape = (2, 2, 3, 4)
    if y.shape == expected_shape and y[0, 0, 0, 0] == 0 and y[0, 0, 0, 3] == 3 and y[0, 1, 0, 0] == 4:
        print("ok")
    else:
        print(f"FAIL  shape={y.shape}  y[0,0,0,:]={y[0,0,0,:]}")
