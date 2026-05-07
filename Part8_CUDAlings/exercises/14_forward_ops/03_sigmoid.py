"""CUDAlings 14.03 -- Sigmoid forward + backward in PyTorch.

PyTorch already has torch.sigmoid; the point of this exercise is to write
the formula yourself so you have it in muscle memory:
    forward:  y = 1 / (1 + exp(-x))
    backward: dx = dy * y * (1 - y)
"""

# I AM NOT DONE

import torch


class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: compute y = sigmoid(x), save y for backward, return it
        return x

    @staticmethod
    def backward(ctx, dy):
        # TODO: pull saved y, return dy * d/dx sigmoid (using y, not x)
        return dy


if __name__ == "__main__":
    x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    y = MySigmoid.apply(x)
    y.sum().backward()
    fwd_ok = torch.allclose(y, torch.sigmoid(x.detach()), atol=1e-5)
    grad_ok = torch.allclose(x.grad, y.detach() * (1 - y.detach()), atol=1e-5)
    print("ok" if fwd_ok and grad_ok else f"FAIL fwd={y} grad={x.grad}")
