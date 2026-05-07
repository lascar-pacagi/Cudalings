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
class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(8)
        self.blk  = ResidualBlock(8)
        self.head = nn.Linear(8, 4)
    def forward(self, x):
        x = F.relu(self.bn(self.stem(x)))
        x = self.blk(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return self.head(x)
def main():
    torch.manual_seed(0)
    x = torch.randn(16, 1, 4, 4)
    targets = (x.view(16, -1).sum(dim=1) > 0).long() * 2 + (torch.randn(16) > 0).long()
    model = TinyResNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    losses = []
    for step in range(50):
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    print("ok" if losses[-1] < losses[0] * 0.6 else f"FAIL {losses[0]:.3f}->{losses[-1]:.3f}")
if __name__ == "__main__":
    main()
