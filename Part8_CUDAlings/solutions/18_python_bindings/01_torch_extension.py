import torch


class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        return dy * (x > 0).to(dy.dtype)


def my_relu(x):
    return MyReLU.apply(x)


if __name__ == "__main__":
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = my_relu(x)
    y.sum().backward()
    fwd_sum = float(y.sum())
    grad_sum = float(x.grad.sum())
    if abs(fwd_sum - 3.0) < 1e-5 and abs(grad_sum - 2.0) < 1e-5:
        print("ok")
    else:
        print(f"FAIL fwd={fwd_sum} grad={grad_sum}")
