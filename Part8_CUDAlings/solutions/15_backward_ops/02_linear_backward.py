import torch
def linear_backward(x, W, dy):
    dx = dy @ W
    dW = dy.transpose(0, 1) @ x
    return dx, dW
if __name__ == "__main__":
    torch.manual_seed(0)
    x  = torch.randn(4, 3, requires_grad=True)
    W  = torch.randn(2, 3, requires_grad=True)
    y  = x @ W.transpose(0, 1)
    dy = torch.ones_like(y)
    y.backward(dy)
    dx, dW = linear_backward(x.detach(), W.detach(), dy)
    ok = (torch.allclose(dx, x.grad, atol=1e-5)
          and torch.allclose(dW, W.grad, atol=1e-5))
    print("ok" if ok else "FAIL")
