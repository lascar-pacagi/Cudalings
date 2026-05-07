import torch
def sgd_step(params, lr):
    with torch.no_grad():
        for p in params:
            p.data -= lr * p.grad
            p.grad.zero_()

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 64).unsqueeze(1)
    y = 3 * x + 2
    w = torch.zeros(1, 1, requires_grad=True)
    b = torch.zeros(1, 1, requires_grad=True)
    losses = []
    for _ in range(200):
        pred = x @ w + b
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        sgd_step([w, b], lr=0.1)
        losses.append(float(loss))
    print("ok" if losses[-1] < 0.01 else f"FAIL {losses[-1]}")
