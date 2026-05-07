"""CUDAlings 17.02 -- SGD optimizer class.

PyTorch's optim.SGD with momentum:
    v = momentum * v + grad
    p = p - lr * v

Goal: implement the SGD class so it tracks per-parameter momentum buffers
internally and applies the update on .step(). Verify it minimizes a tiny
quadratic loss.
"""

# I AM NOT DONE

import torch


class SGD:
    def __init__(self, params, lr, momentum=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        # one momentum buffer per param (init to zeros)
        self.buf = [torch.zeros_like(p) for p in self.params]

    def step(self):
        with torch.no_grad():
            for p, v in zip(self.params, self.buf):
                # TODO: update the momentum buffer in place, then take the SGD step
                pass

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None: p.grad.zero_()


if __name__ == "__main__":
    torch.manual_seed(0)
    p = torch.zeros(1, requires_grad=True)            # target: 5
    opt = SGD([p], lr=0.1, momentum=0.9)
    for _ in range(200):
        loss = (p - 5.0).pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    print("ok" if abs(float(p) - 5.0) < 1e-3 else f"FAIL p={float(p)}")
