#pragma once
// sym_eval.h ─ KAN symbolic layer evaluator
//
// Each SymLayer encodes a fully-symbolised KAN layer:
//   input  (n_samp × in_dim)  →  raw symbolic sum  →  linear calibration
//   → output (n_samp × out_dim)
//
// Edges are grouped by equation type for vectorised Eigen operations.

#include <Eigen/Dense>
#include <string>
#include <vector>

// One group of edges that all share the same symbolic function type.
struct SymGroup {
    std::string             sym_type;
    std::vector<int>        col_indices;           // input feature index per edge
    std::vector<int>        out_indices;           // output node index per edge (replaces routing matrix)
    std::vector<std::vector<double>> params;       // params[edge_k] = {a, b, c, …}
    Eigen::VectorXd         x_mins;                // normalization: min per edge
    Eigen::VectorXd         x_spans;               // normalization: span per edge
    bool                    normalized;
    // Pre-extracted parameter row vectors for batched Eigen evaluation.
    // p0[k], p1[k], p2[k] = params[k][0..2] for all edges k in this group.
    Eigen::RowVectorXd      p0, p1, p2;
};

// One symbolised KAN layer (all edges, grouped by type, + calibration).
struct SymLayer {
    int                      in_dim;
    int                      out_dim;
    bool                     normalized;           // whether to normalize inputs
    std::vector<SymGroup>    groups;
    Eigen::VectorXd          scale;                // out_dim calibration scales
    Eigen::VectorXd          bias;                 // out_dim calibration biases
};

// Apply one symbolic layer: X_in (n_samp × in_dim) → (n_samp × out_dim).
// Handles normalization internally; applies calibration at the end.
Eigen::MatrixXd apply_sym_layer(const SymLayer& layer,
                                const Eigen::MatrixXd& X_in);

// Load a SymLayer from the JSON file produced by export_for_benchmark.py.
SymLayer load_sym_layer(const std::string& json_path);
