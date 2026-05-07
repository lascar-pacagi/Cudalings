// CUDAlings 16.01 — Tiny scalar autograd (micrograd in C++)
//
// Build a Value class that records (data, grad, parents, _backward). The
// .backward() entry-point does a topological sort of the DAG and runs each
// node's _backward in reverse order. This is the seed of every autograd
// system, including the one in cudalearn that you'll build in Part 4-5.
//
// Goal: implement operator+ and operator* so the test below prints
// "ok" with the right gradients.

// I AM NOT DONE

#include <cstdio>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

struct Value : std::enable_shared_from_this<Value> {
    float data;
    float grad = 0.f;
    std::vector<std::shared_ptr<Value>> parents;
    std::function<void()> _backward = []{};

    explicit Value(float v) : data(v) {}

    static std::shared_ptr<Value> make(float v) { return std::make_shared<Value>(v); }

    void backward() {
        std::vector<std::shared_ptr<Value>> topo;
        std::unordered_set<Value*> seen;
        std::function<void(const std::shared_ptr<Value>&)> build =
            [&](const std::shared_ptr<Value>& v) {
                if (seen.insert(v.get()).second) {
                    for (auto& p : v->parents) build(p);
                    topo.push_back(v);
                }
            };
        build(shared_from_this());
        grad = 1.f;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) (*it)->_backward();
    }
};

using V = std::shared_ptr<Value>;

V add(const V& a, const V& b) {
    auto out = Value::make(a->data + b->data);
    out->parents = {a, b};
    // TODO: install out->_backward so addition's gradient flows to both
    //       parents unchanged (mind the ownership cycle: capture `out` weakly).
    return out;
}

V mul(const V& a, const V& b) {
    auto out = Value::make(a->data * b->data);
    out->parents = {a, b};
    // TODO: install out->_backward so multiplication's gradient routes to
    //       each parent scaled by the OTHER parent's value.
    return out;
}

int main() {
    // L = (a + b) * c  with a=2, b=3, c=4 ; L = 20
    // dL/da = c = 4 ; dL/db = c = 4 ; dL/dc = a+b = 5
    auto a = Value::make(2.f);
    auto b = Value::make(3.f);
    auto c = Value::make(4.f);
    auto L = mul(add(a, b), c);
    L->backward();
    bool pass = L->data == 20.f && a->grad == 4.f && b->grad == 4.f && c->grad == 5.f;
    printf(pass ? "ok\n" : "FAIL  L=%.1f a.grad=%.1f b.grad=%.1f c.grad=%.1f\n",
           L->data, a->grad, b->grad, c->grad);
    return pass ? 0 : 1;
}
