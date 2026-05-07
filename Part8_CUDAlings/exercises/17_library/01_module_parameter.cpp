// CUDAlings 17.01 — Module / Parameter base classes
//
// PyTorch-style: a Module owns Parameters and child Modules. .parameters()
// recursively flattens the tree so an Optimizer can iterate them all.
//
// Goal: implement Module::parameters() so it returns every Parameter from
// `this` plus every Parameter of every child Module, recursively.

// I AM NOT DONE

#include <cstdio>
#include <vector>
#include <memory>

struct Parameter { float data = 0.f; float grad = 0.f; };

struct Module {
    std::vector<Parameter*> own_params;
    std::vector<Module*>    children;

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> out;
        // TODO: append own_params to out
        // TODO: for each child, recursively append child->parameters() to out
        return out;
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
    auto p = m.parameters();
    printf("count=%zu\n", p.size());   // 2 (l1) + 2 (l2) = 4
    return 0;
}
