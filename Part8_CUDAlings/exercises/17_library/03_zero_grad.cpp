// CUDAlings 17.03 — Recursive zero_grad over a Module tree
//
// Build on 17.01: a Module owns Parameters and child Modules. Add
// `zero_grad()` that resets `.grad = 0` for every parameter in the
// subtree.

// I AM NOT DONE

#include <cstdio>
#include <vector>

struct Parameter { float data = 0.f; float grad = 0.f; };
struct Module {
    std::vector<Parameter*> own_params;
    std::vector<Module*>    children;

    void zero_grad() {
        // TODO: for each own_param: p->grad = 0.f
        // TODO: for each child: child->zero_grad()
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
    printf("total=%.1f\n", total);     // expected 0.0
    return 0;
}
