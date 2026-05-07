import torch
class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = 1.0 / (1.0 + torch.exp(-x))
        ctx.save_for_backward(y)
        return y
    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        return dy * y * (1.0 - y)
if __name__ == "__main__":
    x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    y = MySigmoid.apply(x)
    y.sum().backward()
    fwd_ok = torch.allclose(y, torch.sigmoid(x.detach()), atol=1e-5)
    grad_ok = torch.allclose(x.grad, y.detach() * (1 - y.detach()), atol=1e-5)
    print("ok" if fwd_ok and grad_ok else f"FAIL fwd={y} grad={x.grad}")
