"""CUDAlings 28.05 -- Conv2d backward verified against autograd.

Conv backward decomposes into two operations:
    dx = conv_transpose(dy, W)        -- a "transposed conv"
    dW = correlation(x, dy)           -- conv with x as kernel and dy as input

We use the elegant im2col formulation:
    forward:  cols = im2col(x, KH, KW)         (IC*KH*KW, OH*OW)
              y_flat = W_flat @ cols           (OC, OH*OW)
    backward: dy_flat is shape (OC, OH*OW)
              dW_flat = dy_flat @ cols^T       (OC, IC*KH*KW)
              dcols   = W_flat^T @ dy_flat     (IC*KH*KW, OH*OW)
              dx      = col2im(dcols)

Goal: implement my_conv2d_backward and verify (dx, dW) against autograd
on a small problem.
"""

# I AM NOT DONE

import torch
import torch.nn.functional as F


def my_conv2d_backward(x, W, dy):
    """x: (1, IC, IH, IW)  W: (OC, IC, KH, KW)  dy: (1, OC, OH, OW)
    Return (dx, dW).
    """
    _, IC, IH, IW = x.shape
    OC, _, KH, KW = W.shape
    _, _, OH, OW = dy.shape

    cols = F.unfold(x, (KH, KW))                     # (1, IC*KH*KW, OH*OW)
    cols0 = cols.squeeze(0)
    dy0 = dy.reshape(OC, OH * OW)
    Wm  = W.reshape(OC, IC * KH * KW)

    # TODO: derive dW from a matmul of dy and cols (then reshape to (OC,IC,KH,KW))
    # TODO: derive dcols from a matmul of W and dy, then fold it back into dx (col2im)
    dW = torch.zeros_like(W)
    dx = torch.zeros_like(x)
    return dx, dW


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 6, 6, requires_grad=True)
    W = torch.randn(4, 3, 3, 3, requires_grad=True)
    y = F.conv2d(x, W)
    dy = torch.randn_like(y)
    y.backward(dy)

    dx_m, dW_m = my_conv2d_backward(x.detach(), W.detach(), dy)
    ok = (torch.allclose(dx_m, x.grad, atol=1e-5)
          and torch.allclose(dW_m, W.grad, atol=1e-5))
    print("ok" if ok else "FAIL")
