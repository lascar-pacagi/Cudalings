import torch
import torch.nn.functional as F
def my_conv2d(x, W, padding=0, stride=1):
    B, IC, IH, IW = x.shape
    OC, _, KH, KW = W.shape
    OH = (IH + 2 * padding - KH) // stride + 1
    OW = (IW + 2 * padding - KW) // stride + 1
    cols = F.unfold(x, (KH, KW), padding=padding, stride=stride)
    Wm = W.reshape(OC, IC * KH * KW)
    y = (Wm @ cols).reshape(B, OC, OH, OW)
    return y
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 8, 8)
    W = torch.randn(4, 3, 3, 3)
    for pad in (0, 1):
        for stride in (1, 2):
            mine = my_conv2d(x, W, padding=pad, stride=stride)
            ref = F.conv2d(x, W, padding=pad, stride=stride)
            if not torch.allclose(mine, ref, atol=1e-5):
                print(f"FAIL pad={pad} stride={stride}")
                break
    else:
        print("ok")
