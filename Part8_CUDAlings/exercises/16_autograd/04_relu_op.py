"""CUDAlings 16.04 -- Add a ReLU op to micrograd-style autograd.

Forward:   y = max(0, x)
Backward:  dx += dy if x > 0 else 0

This is the first NON-LINEAR op. The backward gates the upstream gradient
on the saved input's sign.

Goal: implement `relu(x)` so the test passes.
"""

# I AM NOT DONE


class Value:
    def __init__(self, data, parents=(), _backward=lambda: None):
        self.data = float(data); self.grad = 0.0
        self.parents = list(parents); self._backward = _backward
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


def relu(x):
    out = Value(x.data if x.data > 0 else 0.0, [x])
    def _bwd():
        # TODO: route out.grad through the ReLU mask into x.grad
        pass
    out._backward = _bwd
    return out


if __name__ == "__main__":
    # ReLU(-2) = 0  → dy = 1 → dx = 0
    a = Value(-2.0)
    L = relu(a); L.backward()
    if L.data != 0.0 or a.grad != 0.0: print(f"FAIL neg L={L.data} a.grad={a.grad}"); raise SystemExit

    # ReLU(3) = 3   → dy = 1 → dx = 1
    b = Value(3.0)
    L = relu(b); L.backward()
    if L.data != 3.0 or b.grad != 1.0: print(f"FAIL pos L={L.data} b.grad={b.grad}"); raise SystemExit

    print("ok")
