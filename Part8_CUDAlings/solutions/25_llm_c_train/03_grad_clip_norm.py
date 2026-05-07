import torch
def my_clip_grad_norm(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads: return 0.0
    total_sq = sum((g ** 2).sum() for g in grads)
    total_norm = total_sq.sqrt()
    if float(total_norm) > max_norm:
        scale = max_norm / (float(total_norm) + 1e-6)
        for g in grads: g.mul_(scale)
    return float(total_norm)
if __name__ == "__main__":
    torch.manual_seed(0)
    p1 = torch.zeros(3, requires_grad=True)
    p2 = torch.zeros(2, requires_grad=True)
    p1.grad = torch.tensor([3.0, 4.0, 0.0])
    p2.grad = torch.tensor([0.0, 0.0])
    n = my_clip_grad_norm([p1, p2], max_norm=1.0)
    new_norm = (p1.grad ** 2).sum().sqrt()
    if abs(n - 5.0) < 1e-3 and abs(float(new_norm) - 1.0) < 1e-3:
        print("ok")
    else:
        print(f"FAIL n={n} nn={float(new_norm)}")
