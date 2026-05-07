"""CUDAlings 25.03 -- Global gradient norm clipping.

Standard trick when training transformers (or anything else with
exploding gradients):
    total_norm = sqrt(sum_i ||g_i||^2)
    if total_norm > clip:
        for each g: g *= clip / total_norm

Goal: implement clip_grad_norm_(params, max_norm) in pure tensor ops.
We compare against torch.nn.utils.clip_grad_norm_.
"""

# I AM NOT DONE

import torch


def my_clip_grad_norm(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    # TODO: compute the global L2 norm across all grads (concatenated)
    total_norm = torch.tensor(0.0)
    # TODO: if it exceeds max_norm, rescale every grad in place to bring it to max_norm
    return float(total_norm)


if __name__ == "__main__":
    torch.manual_seed(0)
    p1 = torch.zeros(3, requires_grad=True)
    p2 = torch.zeros(2, requires_grad=True)
    p1.grad = torch.tensor([3.0, 4.0, 0.0])    # ||.|| = 5
    p2.grad = torch.tensor([0.0, 0.0])         # ||.|| = 0
    # Total norm = 5. Clip to 1: scale by 0.2.
    n = my_clip_grad_norm([p1, p2], max_norm=1.0)
    new_norm = (p1.grad ** 2).sum().sqrt()
    if abs(n - 5.0) < 1e-3 and abs(float(new_norm) - 1.0) < 1e-3:
        print("ok")
    else:
        print(f"FAIL n_returned={n}  new_norm={float(new_norm)}")
