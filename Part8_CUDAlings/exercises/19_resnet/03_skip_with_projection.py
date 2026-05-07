"""CUDAlings 19.03 -- Residual block where channels change.

When the residual path's channel count differs from the main path, you
add a 1x1 conv to project the skip:
    if in_c != out_c:
        skip = Conv1x1(in_c, out_c)(x)
    else:
        skip = x
    return relu(F + skip)

This is what every "down-sampling" block in a ResNet does: increase
channels via the main path, project the skip with 1x1 conv (and matching
stride if also down-sampling spatially).

Goal: implement DownsampleBlock taking 16 channels -> 32 channels with
stride 2 on the main path. Verify shape.
"""

# I AM NOT DONE

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
        # TODO: self.proj = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
        # TODO: self.bp = nn.BatchNorm2d(out_c)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        # TODO: skip = self.bp(self.proj(x))
        skip = x      # placeholder; will FAIL until you add the proj path
        return F.relu(h + skip)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = DownsampleBlock(16, 32, stride=2); m.train(False)
    x = torch.randn(1, 16, 8, 8)
    y = m(x)
    print("ok" if y.shape == (1, 32, 4, 4) else f"FAIL {y.shape}")
