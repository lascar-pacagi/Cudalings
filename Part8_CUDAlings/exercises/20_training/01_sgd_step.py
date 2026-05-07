"""CUDAlings 20.01 -- Hand-written SGD step.

Goal: implement `sgd_step(params, lr)` so it does, for each parameter:
    p.data -= lr * p.grad
    p.grad.zero_()

Then drive a tiny linear regression and show that the loss goes down.
"""

# I AM NOT DONE

import torch


def sgd_step(params, lr):
    # TODO: with torch.no_grad(): for p in params: p.data -= lr * p.grad; p.grad.zero_()
    pass


if __name__ == "__main__":
    torch.manual_seed(0)
    # y = 3x + 2  (target weights)
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

    if losses[-1] < 0.01 and losses[-1] < losses[0]:
        print("ok")
    else:
        print(f"FAIL  start={losses[0]:.4f}  end={losses[-1]:.4f}")
