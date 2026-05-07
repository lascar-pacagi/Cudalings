import math
def get_lr(step, lr_max, lr_min, warmup, max_steps):
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, progress)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
if __name__ == "__main__":
    a = get_lr(0,    1.0, 0.1, 100, 1000)
    b = get_lr(100,  1.0, 0.1, 100, 1000)
    c = get_lr(1000, 1.0, 0.1, 100, 1000)
    ok = (abs(a - 0.0) < 1e-6 and abs(b - 1.0) < 1e-6 and abs(c - 0.1) < 1e-3)
    print("ok" if ok else f"FAIL {a} {b} {c}")
