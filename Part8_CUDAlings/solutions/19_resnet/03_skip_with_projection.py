import torch
import torch.nn as nn
import torch.nn.functional as F
class DownsampleBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride, bias=False)
        self.b1 = nn.BatchNorm2d(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(out_c)
        self.proj = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
        self.bp = nn.BatchNorm2d(out_c)
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        skip = self.bp(self.proj(x))
        return F.relu(h + skip)
if __name__ == "__main__":
    torch.manual_seed(0)
    m = DownsampleBlock(16, 32, stride=2); m.train(False)
    x = torch.randn(1, 16, 8, 8)
    y = m(x)
    print("ok" if y.shape == (1, 32, 4, 4) else f"FAIL {y.shape}")
