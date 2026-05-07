import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + x)
class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(16)
        self.s1 = nn.Sequential(ResidualBlock(16), ResidualBlock(16))
        self.s2 = nn.Sequential(ResidualBlock(16), ResidualBlock(16))
        self.head = nn.Linear(16, num_classes)
    def forward(self, x):
        x = F.relu(self.bn(self.stem(x)))
        x = self.s1(x); x = self.s2(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.head(x)
if __name__ == "__main__":
    torch.manual_seed(0)
    m = ResNet8(); m.train(False)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print("ok" if y.shape == (2, 10) else f"FAIL {y.shape}")
