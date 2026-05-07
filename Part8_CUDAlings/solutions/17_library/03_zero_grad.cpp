#include <cstdio>
#include <vector>
struct Parameter { float data = 0.f; float grad = 0.f; };
struct Module {
    std::vector<Parameter*> own_params;
    std::vector<Module*>    children;
    void zero_grad() {
        for (auto* p : own_params) p->grad = 0.f;
        for (auto* c : children) c->zero_grad();
    }
};
struct Linear : Module {
    Parameter w, b;
    Linear() { own_params = {&w, &b}; }
};
struct MLP : Module {
    Linear l1, l2;
    MLP() { children = {&l1, &l2}; }
};
int main() {
    MLP m;
    m.l1.w.grad = 1.0f;  m.l1.b.grad = 2.0f;
    m.l2.w.grad = 3.0f;  m.l2.b.grad = 4.0f;
    m.zero_grad();
    float total = m.l1.w.grad + m.l1.b.grad + m.l2.w.grad + m.l2.b.grad;
    printf("total=%.1f\n", total);
    return 0;
}
