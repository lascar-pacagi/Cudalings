"""CUDAlings 28.04 -- A small CNN forward pass end-to-end.

Architecture (LeNet-like):
    input (B, 1, 28, 28)
        → Conv(1->8, 3x3, pad=1) → BN → ReLU → MaxPool(2)    -- (B, 8, 14, 14)
        → Conv(8->16, 3x3, pad=1) → BN → ReLU → MaxPool(2)   -- (B, 16, 7, 7)
        → Flatten → Linear(16*7*7 -> 10)                      -- (B, 10)

This is a real CNN you can train on MNIST. Goal: complete the `forward`
method using nn.Conv2d, nn.BatchNorm2d, F.relu, F.max_pool2d, and the
final Linear. Verify shapes.
"""

# I AM NOT DONE

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        # TODO: chain the two conv→bn→relu→pool stages, flatten, then run the FC head
        return torch.zeros(x.size(0), 10)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyCNN()
    model.train(False)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    if y.shape == (4, 10):
        print("ok")
    else:
        print(f"FAIL {y.shape}")
