"""CUDAlings 19.01 -- A residual block.

A ResBlock computes y = relu(x + F(x)), where F is two stacked Conv-BN-ReLU
layers. The "skip" lets gradients propagate freely past F, which is what
makes very deep networks trainable.

Goal: implement `forward` for the ResidualBlock below. Channels in == out
so no projection is needed.
"""

# I AM NOT DONE

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        # TODO: implement y = relu(x + F(x)) where F = (bn2 ∘ conv2 ∘ relu ∘ bn1 ∘ conv1)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    block = ResidualBlock(8)
    block.train(False)                   # inference mode (no BN running stats update)
    x = torch.randn(2, 8, 4, 4)
    y = block(x)
    if y.shape == x.shape:
        print("ok")
    else:
        print(f"FAIL: shape {y.shape}")
