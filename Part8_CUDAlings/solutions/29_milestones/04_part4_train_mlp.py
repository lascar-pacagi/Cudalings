import math
import torch
class MLP:
    def __init__(self, in_dim, hidden, out_dim):
        self.w1 = torch.zeros(in_dim, hidden, requires_grad=True)
        self.b1 = torch.zeros(hidden, requires_grad=True)
        self.w2 = torch.zeros(hidden, out_dim, requires_grad=True)
        self.b2 = torch.zeros(out_dim, requires_grad=True)
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
    def __call__(self, x):
        h = torch.relu(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2
    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]
def main():
    torch.manual_seed(0)
    x = torch.linspace(-math.pi, math.pi, 256).unsqueeze(1)
    y = torch.sin(x)
    model = MLP(1, 64, 1)
    lr = 0.01
    for step in range(2000):
        pred = model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad
                p.grad.zero_()
    final = float(((model(x) - y) ** 2).mean())
    print("ok" if final < 0.05 else f"FAIL final_mse={final:.4f}")
if __name__ == "__main__":
    main()
