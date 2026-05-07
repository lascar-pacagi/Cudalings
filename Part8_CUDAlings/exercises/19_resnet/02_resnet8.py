"""CUDAlings 19.02 -- A small ResNet-8 from primitive blocks.

Architecture:
    stem: Conv(3->16, 3x3, pad=1)
    stage 1: 2 ResidualBlocks at channels=16
    stage 2: 2 ResidualBlocks at channels=32  (first does down-sample)
    head: GlobalAvgPool -> Linear(32 -> 10)

Total: 1 stem conv + 2*2 = 4 residual blocks + head = ~8 conv layers.

Goal: assemble a ResNet8 model. We don't train -- just verify shape
(B, 10) for a CIFAR-like input (B, 3, 32, 32).
"""

# I AM NOT DONE

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
        # TODO: register the layers from the docstring (stem conv + bn, two
        #       2-block stages at 16 channels, final Linear head).
        pass

    def forward(self, x):
        # TODO: chain the layers per the docstring; finish with global-avg-pool
        #       + flatten + Linear head.
        return torch.zeros(x.size(0), 10)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = ResNet8()
    m.train(False)
    x = torch.randn(2, 3, 32, 32)
    y = m(x)
    print("ok" if y.shape == (2, 10) else f"FAIL {y.shape}")
