"""CUDAlings 14.06 -- Conv2d as im2col + matmul (the cuDNN trick).

Convolution can be reshaped into a single matrix multiply:
    1. im2col: turn each KH*KW patch of the input into one column of a
       big matrix. Result shape: (IC*KH*KW, OH*OW).
    2. flatten weights: (OC, IC*KH*KW)
    3. matmul: (OC, IC*KH*KW) @ (IC*KH*KW, OH*OW) = (OC, OH*OW)
    4. reshape: (OC, OH*OW) -> (OC, OH, OW)

Why? Matmul kernels are HEAVILY tuned. Reformulating conv as matmul lets
you piggyback on cuBLAS / cuDNN's matmul. cuDNN does this internally for
Pascal-era GPUs that lack tensor cores.

Goal: implement im2col_then_matmul for a single image, single batch.
Verify against torch.nn.functional.conv2d.
"""

# I AM NOT DONE

import torch


def im2col_then_matmul(x, W):
    """x: (IC, IH, IW)  W: (OC, IC, KH, KW)  → y: (OC, OH, OW)"""
    IC, IH, IW = x.shape
    OC, _, KH, KW = W.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    # TODO: cols = torch.zeros(IC * KH * KW, OH * OW)
    # TODO: for kh, kw in [0..KH) x [0..KW), copy x[:, kh:kh+OH, kw:kw+OW] into cols[*, h*OW + w]
    cols = torch.zeros(IC * KH * KW, OH * OW)

    Wm = W.reshape(OC, IC * KH * KW)         # (OC, IC*KH*KW)
    y = (Wm @ cols).reshape(OC, OH, OW)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(3, 8, 8)
    W = torch.randn(4, 3, 3, 3)
    mine = im2col_then_matmul(x, W)
    ref = torch.nn.functional.conv2d(x.unsqueeze(0), W).squeeze(0)
    print("ok" if torch.allclose(mine, ref, atol=1e-5) else f"FAIL {(mine - ref).abs().max()}")
