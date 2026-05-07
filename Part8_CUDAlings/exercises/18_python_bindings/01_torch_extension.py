"""CUDAlings 18.01 -- A custom PyTorch op that mimics ReLU.

Goal: implement `my_relu(x)` using *torch ops only* (no custom CUDA yet).
Then wire its backward via `torch.autograd.Function`. This is the bridge
between "I can write a kernel" and "PyTorch knows it's a layer with grads".

In Chapter 18 of the course, you'll replace the forward/backward bodies
with calls to your own CUDA kernels via the load_inline mechanism. Here
we just exercise the autograd wiring.
"""

# I AM NOT DONE

import torch


class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # TODO: ctx.save_for_backward(x) and return x.clamp(min=0)
        return x

    @staticmethod
    def backward(ctx, dy):
        # TODO: pull the saved x; return dy * (x > 0).float()
        return dy


def my_relu(x):
    return MyReLU.apply(x)


if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = my_relu(x)
    y.sum().backward()
    # Expected forward: [0, 0, 0, 1, 2]   sum = 3
    # Expected grad:    [0, 0, 0, 1, 1]   sum = 2
    fwd_sum = float(y.sum())
    grad_sum = float(x.grad.sum())
    if abs(fwd_sum - 3.0) < 1e-5 and abs(grad_sum - 2.0) < 1e-5:
        print("ok")
    else:
        print(f"FAIL fwd={fwd_sum} grad={grad_sum}")
