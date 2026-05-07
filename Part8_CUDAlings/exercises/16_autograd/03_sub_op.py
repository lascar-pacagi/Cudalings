"""CUDAlings 16.03 -- Add a subtraction op to a micrograd-style autograd.

Forward:   c = a - b
Backward:  da += dc        (gradient flows through unchanged)
           db += -dc       (negated)

Goal: implement `sub` matching the pattern of `add` and `mul` (chapter 16.01).
"""

# I AM NOT DONE

import torch


class Value:
    def __init__(self, data, parents=(), _backward=lambda: None):
        self.data = float(data)
        self.grad = 0.0
        self.parents = list(parents)
        self._backward = _backward

    def backward(self):
        topo, seen = [], set()
        def build(n):
            if id(n) in seen: return
            seen.add(id(n))
            for p in n.parents: build(p)
            topo.append(n)
        build(self)
        self.grad = 1.0
        for n in reversed(topo): n._backward()


def add(a, b):
    out = Value(a.data + b.data, [a, b])
    def _bwd():
        a.grad += out.grad
        b.grad += out.grad
    out._backward = _bwd
    return out


def sub(a, b):
    # TODO: build the out Value and wire its _backward (gradients per the docstring)
    return Value(0)


if __name__ == "__main__":
    # L = a - b - c    (= a - (b + c) effectively, but we use only sub)
    # dL/da = 1; dL/db = -1; dL/dc = -1
    a, b, c = Value(10), Value(3), Value(2)
    L = sub(sub(a, b), c)
    L.backward()
    ok = (L.data == 5.0 and a.grad == 1.0 and b.grad == -1.0 and c.grad == -1.0)
    print("ok" if ok else f"FAIL L={L.data} a={a.grad} b={b.grad} c={c.grad}")
