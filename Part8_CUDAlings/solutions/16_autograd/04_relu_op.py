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
    def _bwd(): x.grad += (1.0 if x.data > 0 else 0.0) * out.grad
    out._backward = _bwd
    return out
if __name__ == "__main__":
    a = Value(-2.0); L = relu(a); L.backward()
    if not (L.data == 0.0 and a.grad == 0.0): print("FAIL"); raise SystemExit
    b = Value(3.0); L = relu(b); L.backward()
    if not (L.data == 3.0 and b.grad == 1.0): print("FAIL"); raise SystemExit
    print("ok")
