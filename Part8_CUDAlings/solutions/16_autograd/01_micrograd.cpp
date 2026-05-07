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
    Value* o = out.get();
    out->_backward = [a, b, o] { a->grad += o->grad; b->grad += o->grad; };
    return out;
}
V mul(const V& a, const V& b) {
    auto out = Value::make(a->data * b->data);
    out->parents = {a, b};
    Value* o = out.get();
    out->_backward = [a, b, o] {
        a->grad += b->data * o->grad;
        b->grad += a->data * o->grad;
    };
    return out;
}

int main() {
    auto a = Value::make(2.f);
    auto b = Value::make(3.f);
    auto c = Value::make(4.f);
    auto L = mul(add(a, b), c);
    L->backward();
    bool pass = L->data == 20.f && a->grad == 4.f && b->grad == 4.f && c->grad == 5.f;
    printf(pass ? "ok\n" : "FAIL\n");
    return pass ? 0 : 1;
}
