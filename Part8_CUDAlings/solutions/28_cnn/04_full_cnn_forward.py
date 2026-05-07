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
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)
if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyCNN()
    model.train(False)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    print("ok" if y.shape == (4, 10) else f"FAIL {y.shape}")
