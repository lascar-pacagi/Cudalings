"""CUDAlings 15.02 -- Backward of a Linear layer.

Forward (no bias):  y = x @ W^T              x: (B, IC), W: (OC, IC), y: (B, OC)
Backward:
    dx = dy @ W                              (B, OC) @ (OC, IC) = (B, IC)
    dW = dy^T @ x                            (OC, B) @ (B, IC)  = (OC, IC)
"""

# I AM NOT DONE

import torch


def linear_backward(x, W, dy):
    """Return (dx, dW) given the forward inputs and dy."""
    dx = None
    dW = None
    # TODO: implement the two matmuls for dx and dW per the docstring shapes
    return dx, dW


if __name__ == "__main__":
    torch.manual_seed(0)
    x  = torch.randn(4, 3, requires_grad=True)
    W  = torch.randn(2, 3, requires_grad=True)
    y  = x @ W.transpose(0, 1)        # (4, 2)
    dy = torch.ones_like(y)
    y.backward(dy)

    dx, dW = linear_backward(x.detach(), W.detach(), dy)
    if (dx is not None and dW is not None
        and torch.allclose(dx, x.grad, atol=1e-5)
        and torch.allclose(dW, W.grad, atol=1e-5)):
        print("ok")
    else:
        print(f"FAIL")
