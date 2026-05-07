import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)

if __name__ == "__main__":
    torch.manual_seed(0)
    block = ResidualBlock(8)
    block.train(False)
    x = torch.randn(2, 8, 4, 4)
    y = block(x)
    print("ok" if y.shape == x.shape else f"FAIL {y.shape}")
