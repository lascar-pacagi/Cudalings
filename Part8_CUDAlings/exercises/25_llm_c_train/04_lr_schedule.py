"""CUDAlings 25.04 -- Cosine learning-rate schedule with linear warmup.

The schedule used by every modern LLM (GPT-2, Llama, Mistral, ...):

    if step < warmup:
        lr = lr_max * step / warmup           # linear warmup
    else:
        progress = (step - warmup) / (max_steps - warmup)   # 0..1
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))

Goal: implement get_lr(step) and verify the schedule has the right shape:
    step 0       -> 0  (warmup starts at 0)
    step warmup  -> lr_max
    step max_steps -> lr_min
"""

# I AM NOT DONE

import math


def get_lr(step, lr_max, lr_min, warmup, max_steps):
    if step < warmup:
        # TODO: return lr_max * step / warmup
        return 0.0
    # TODO: progress = (step - warmup) / max(1, max_steps - warmup)
    # TODO: progress = min(1.0, progress)
    # TODO: return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    return 0.0


if __name__ == "__main__":
    LR_MAX, LR_MIN, WARMUP, MAX = 1.0, 0.1, 100, 1000
    a = get_lr(0,    LR_MAX, LR_MIN, WARMUP, MAX)
    b = get_lr(100,  LR_MAX, LR_MIN, WARMUP, MAX)
    c = get_lr(1000, LR_MAX, LR_MIN, WARMUP, MAX)
    ok = (abs(a - 0.0) < 1e-6
          and abs(b - 1.0) < 1e-6
          and abs(c - 0.1) < 1e-3)
    print("ok" if ok else f"FAIL a={a} b={b} c={c}")
