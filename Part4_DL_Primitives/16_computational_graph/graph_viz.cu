// ===========================================================================
// Chapter 16: graph_viz.cu -- Computational Graph Visualization
// ===========================================================================
//
// This file provides tools to visualize the computational graph as:
//   1. ASCII art -- printed directly to the terminal
//   2. DOT format -- for Graphviz rendering (optional, pipe to `dot -Tpng`)
//
// The visualizer walks the DAG from a root tensor (typically the loss)
// and prints each node with its name, shape, and operation type.
//
// ===========================================================================

#include "autograd_ops.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <set>
#include <sstream>
#include <algorithm>

// ===========================================================================
// ASCII Graph Printer
// ===========================================================================
//
// Prints the computational graph as an indented tree, showing the DAG
// structure from output (root) to inputs (leaves).
//
// Example output:
//
//   [ce_loss] cross_entropy (1)
//     [linear_out] linear (2, 3)
//       [gap_out] global_avg_pool (2, 2)
//         [relu_out] relu (2, 2, 4, 4)
//           [conv2d_out] conv2d (2, 2, 4, 4)
//             [input] LEAF (2, 1, 4, 4)
//             [conv_w] PARAM (2, 1, 3, 3)
//       [lin_w] PARAM (3, 2)
//       [lin_b] PARAM (3)
//
// ===========================================================================

void print_ascii_graph(GradTensor* node, int depth = 0,
                        std::set<GradTensor*>* visited = nullptr) {
    // Create visited set on first call
    bool own_visited = false;
    if (!visited) {
        visited = new std::set<GradTensor*>();
        own_visited = true;
    }

    // Indentation
    for (int i = 0; i < depth; i++) {
        printf("  ");
    }

    // Print this node
    bool is_leaf = node->children.empty();
    bool is_param = is_leaf && node->requires_grad;

    if (is_param) {
        printf("[%s] PARAM %s\n", node->name.c_str(), node->shape_str().c_str());
    } else if (is_leaf) {
        printf("[%s] LEAF %s\n", node->name.c_str(), node->shape_str().c_str());
    } else {
        printf("[%s] %s %s\n", node->name.c_str(),
               node->op_name.empty() ? "op" : node->op_name.c_str(),
               node->shape_str().c_str());
    }

    // Check if already visited (prevent infinite loops, show sharing)
    if (visited->count(node)) {
        for (int i = 0; i < depth + 1; i++) printf("  ");
        printf("(... already shown above ...)\n");
        if (own_visited) delete visited;
        return;
    }
    visited->insert(node);

    // Recurse into children
    for (auto& child : node->children) {
        print_ascii_graph(child.get(), depth + 1, visited);
    }

    if (own_visited) delete visited;
}

// ===========================================================================
// DOT Format Generator (for Graphviz)
// ===========================================================================
//
// Generates a .dot file that can be rendered with:
//   ./graph_viz | dot -Tpng -o graph.png
//
// Nodes are colored by type:
//   - Parameters: lightblue
//   - Leaf inputs: lightyellow
//   - Operations: lightsalmon
//   - Loss: lightgreen
//
// ===========================================================================

void generate_dot(GradTensor* root, FILE* out = stdout) {
    fprintf(out, "digraph computational_graph {\n");
    fprintf(out, "  rankdir=BT;\n");  // Bottom-to-top (inputs at bottom)
    fprintf(out, "  node [shape=record, style=filled];\n\n");

    // Collect all nodes via DFS
    std::vector<GradTensor*> all_nodes;
    std::set<GradTensor*> visited;

    std::function<void(GradTensor*)> collect = [&](GradTensor* node) {
        if (visited.count(node)) return;
        visited.insert(node);
        all_nodes.push_back(node);
        for (auto& child : node->children) {
            collect(child.get());
        }
    };
    collect(root);

    // Print node definitions
    for (auto* node : all_nodes) {
        bool is_leaf = node->children.empty();
        bool is_param = is_leaf && node->requires_grad;
        bool is_root = (node == root);

        const char* color = "white";
        if (is_root) color = "lightgreen";
        else if (is_param) color = "lightblue";
        else if (is_leaf) color = "lightyellow";
        else color = "lightsalmon";

        std::string label = node->name;
        if (!node->op_name.empty()) {
            label += "\\n" + node->op_name;
        }
        label += "\\n" + node->shape_str();

        fprintf(out, "  node_%p [label=\"%s\", fillcolor=%s];\n",
                (void*)node, label.c_str(), color);
    }

    fprintf(out, "\n");

    // Print edges (child -> parent, since data flows up)
    visited.clear();
    std::function<void(GradTensor*)> print_edges = [&](GradTensor* node) {
        if (visited.count(node)) return;
        visited.insert(node);
        for (auto& child : node->children) {
            fprintf(out, "  node_%p -> node_%p;\n",
                    (void*)child.get(), (void*)node);
            print_edges(child.get());
        }
    };
    print_edges(root);

    fprintf(out, "}\n");
}

// ===========================================================================
// Build a mini-ResNet-like graph for visualization
// ===========================================================================
//
// Architecture (inspired by a single ResNet basic block):
//
//   input (1, 8, 8, 8)
//     |
//     +--------> conv1 (8->16, 3x3, pad=1) -> BN1 -> ReLU
//     |                                                  |
//     |                                                  v
//     +--------> conv_skip (8->16, 1x1, pad=0)   ->   ADD
//                                                        |
//                                                        v
//                                               GAP -> Linear -> CE Loss
//
// Note: We use conv_skip to match channel dimensions (8->16).
//
// ===========================================================================

void build_and_show_resnet_block() {
    printf("\n================================================================\n");
    printf("  Mini-ResNet Block: Computational Graph\n");
    printf("================================================================\n\n");

    int B = 1, C = 4, H = 8, W = 8;
    int OC = 4, KH = 3, KW = 3;
    int num_classes = 3;

    // Create leaf tensors (parameters and input)
    auto input = make_grad_tensor({B, C, H, W}, false, "input");
    auto conv1_w = make_grad_tensor({OC, C, KH, KW}, true, "conv1.weight");
    auto bn1_gamma = make_ones({OC}, true, "bn1.gamma");
    auto bn1_beta = make_zeros({OC}, true, "bn1.beta");
    auto bn1_rmean = make_zeros({OC}, false, "bn1.running_mean");
    auto bn1_rvar = make_ones({OC}, false, "bn1.running_var");
    auto fc_w = make_grad_tensor({num_classes, OC}, true, "fc.weight");
    auto fc_b = make_zeros({num_classes}, true, "fc.bias");
    auto labels = make_grad_tensor({B}, false, "labels");

    // Initialize with small random values
    std::vector<float> h_input(B * C * H * W);
    for (int i = 0; i < (int)h_input.size(); i++) h_input[i] = 0.1f * ((i * 7) % 11 - 5);
    input->set_data_from_host(h_input.data());

    std::vector<float> h_conv1(OC * C * KH * KW);
    for (int i = 0; i < (int)h_conv1.size(); i++) h_conv1[i] = 0.05f * ((i * 3) % 9 - 4);
    conv1_w->set_data_from_host(h_conv1.data());

    std::vector<float> h_fc(num_classes * OC);
    for (int i = 0; i < (int)h_fc.size(); i++) h_fc[i] = 0.1f * ((i * 5) % 7 - 3);
    fc_w->set_data_from_host(h_fc.data());

    float h_labels[] = {1.0f};
    labels->set_data_from_host(h_labels);

    // ---- Build the graph ----
    printf("Building computational graph...\n\n");

    // Main path: conv -> bn -> relu
    auto conv1_out = autograd::conv2d(input, conv1_w, 1);
    auto bn1_out = autograd::batchnorm(conv1_out, bn1_gamma, bn1_beta,
                                        bn1_rmean, bn1_rvar, true);
    auto relu1_out = autograd::relu(bn1_out);

    // Residual connection: add input (same channels, so direct addition)
    auto res_out = autograd::add(relu1_out, input);

    // Classification head: GAP -> Linear -> Cross-Entropy
    auto gap_out = autograd::global_avg_pool(res_out);
    auto logits = autograd::linear(gap_out, fc_w, fc_b);
    auto loss = autograd::cross_entropy(logits, labels);

    // ---- Print ASCII graph ----
    printf("--- ASCII Graph (root = loss, leaves = inputs/parameters) ---\n\n");
    print_ascii_graph(loss.get());

    // ---- Print DOT format ----
    printf("\n--- DOT Format (for Graphviz) ---\n");
    printf("--- Save to file and run: dot -Tpng graph.dot -o graph.png ---\n\n");
    generate_dot(loss.get());

    // ---- Run backward and show gradient info ----
    printf("\n--- Running backward pass ---\n");
    loss->backward();

    float h_loss;
    loss->get_data_to_host(&h_loss);
    printf("  Loss value: %.6f\n", h_loss);

    // Show gradient norms for parameters
    auto show_grad_norm = [](GradTensorPtr param) {
        std::vector<float> h_grad(param->size);
        param->get_grad_to_host(h_grad.data());
        float norm = 0.0f;
        for (float g : h_grad) norm += g * g;
        norm = sqrtf(norm);
        printf("  %-20s  grad_norm = %.6f\n", param->name.c_str(), norm);
    };

    printf("\n  Parameter gradient norms:\n");
    show_grad_norm(conv1_w);
    show_grad_norm(bn1_gamma);
    show_grad_norm(bn1_beta);
    show_grad_norm(fc_w);
    show_grad_norm(fc_b);
}

// ===========================================================================
//  MAIN
// ===========================================================================
int main() {
    printf("================================================================\n");
    printf("  Chapter 16: Computational Graph Visualization\n");
    printf("================================================================\n");

    build_and_show_resnet_block();

    printf("\n================================================================\n");
    printf("  Done.\n");
    printf("================================================================\n");

    return 0;
}
