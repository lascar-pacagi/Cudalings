"""CUDAlings 17.04 -- state_dict: save/load model weights as a dict.

PyTorch's state_dict is just an ordered dict of {name: tensor}. Saving
and loading by name is what makes "checkpoint then resume" work across
different code revisions, as long as parameter names are stable.

Goal: implement get_state_dict and load_state_dict so a roundtrip
preserves the parameter values.
"""

# I AM NOT DONE

import torch
import torch.nn as nn


def get_state_dict(model):
    """Return {name: tensor.clone()} for every parameter."""
    out = {}
    # TODO: for name, p in model.named_parameters(): out[name] = p.data.clone()
    return out


def load_state_dict(model, sd):
    """Copy each tensor in `sd` back into the model's parameters by name."""
    # TODO: with torch.no_grad(): for name, p in model.named_parameters(): p.copy_(sd[name])
    pass


if __name__ == "__main__":
    torch.manual_seed(0)
    m1 = nn.Linear(3, 2)
    sd = get_state_dict(m1)

    m2 = nn.Linear(3, 2)
    load_state_dict(m2, sd)

    eq = (torch.equal(m1.weight, m2.weight) and torch.equal(m1.bias, m2.bias))
    print("ok" if eq else "FAIL")
