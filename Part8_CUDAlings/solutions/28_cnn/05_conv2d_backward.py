import torch
import torch.nn.functional as F
def my_conv2d_backward(x, W, dy):
    _, IC, IH, IW = x.shape
    OC, _, KH, KW = W.shape
    _, _, OH, OW = dy.shape
    cols = F.unfold(x, (KH, KW))
    cols0 = cols.squeeze(0)
    dy0 = dy.reshape(OC, OH * OW)
    Wm  = W.reshape(OC, IC * KH * KW)
    dW_flat = dy0 @ cols0.transpose(0, 1)
    dW = dW_flat.reshape(OC, IC, KH, KW)
    dcols = Wm.transpose(0, 1) @ dy0
    dx = F.fold(dcols.unsqueeze(0), (IH, IW), (KH, KW))
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
