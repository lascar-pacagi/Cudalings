#include <cstdio>
#include <vector>
struct Parameter { float data = 0.f; float grad = 0.f; };
struct Module {
    std::vector<Parameter*> own_params;
    std::vector<Module*>    children;
    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> out(own_params.begin(), own_params.end());
        for (auto* c : children) {
            auto sub = c->parameters();
            out.insert(out.end(), sub.begin(), sub.end());
        }
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
    printf("count=%zu\n", p.size());
    return 0;
}
