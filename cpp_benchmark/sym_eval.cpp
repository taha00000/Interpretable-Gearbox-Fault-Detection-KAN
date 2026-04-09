// sym_eval.cpp ─ KAN symbolic layer evaluator (batched + sparse-routing)
//
// Implements all 13 primitives from benchmark_inference.py.
//
// Optimisations vs. original per-column loop:
//
//   1. Batched primitive evaluation:
//      Parameters for all edges in a group are stored as row vectors (p0,p1,p2).
//      Each primitive is applied to the whole (n_samp × n_group) matrix in one
//      Eigen array expression — the compiler can auto-vectorize across all edges.
//
//   2. Sparse routing → scatter-add:
//      Each edge connects exactly one input to one output, so the routing
//      matrix has one 1 per row.  Instead of a dense GEMM
//        out += Y_group × routing   [n_samp × n_group × out_dim multiply-adds]
//      we scatter-add each column of Y_group directly to the correct output column:
//        out.col(out_j[k]) += Y_group.col(k)   [n_samp adds, one per edge]
//      For layer 0 (905 edges, out_dim=22) this is ~22× fewer flops.
//
//   3. Fused gather + normalise:
//      X_norm is built in a single pass from X_in, avoiding one temporary matrix.
//
// For each group:
//   1. Gather + optionally normalise → X_norm  (n_samp × n_group)
//   2. Apply primitive to whole matrix          → Y_group (n_samp × n_group)
//   3. Scatter-add: out.col(out_j[k]) += Y_group.col(k)  for each k
// Finally: out = out * scale + bias  (element-wise calibration)

#include "sym_eval.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using json = nlohmann::json;

// ── Batched primitive dispatch ────────────────────────────────────────────────
// X  : (n_samp × n_group) — already normalised
// grp: group with pre-built p0/p1/p2 row vectors (1 × n_group)
// Returns Y : (n_samp × n_group)

static Eigen::MatrixXd apply_group(const SymGroup& grp,
                                    const Eigen::MatrixXd& X)
{
    const int n_samp  = static_cast<int>(X.rows());
    const int n_group = static_cast<int>(X.cols());
    Eigen::MatrixXd Y(n_samp, n_group);

    const auto& sym = grp.sym_type;
    const auto& p0  = grp.p0;   // (1 × n_group)
    const auto& p1  = grp.p1;
    const auto& p2  = grp.p2;

    if (sym == "linear") {
        // a*x + b
        Y = (X.array().rowwise() * p0.array()).rowwise() + p1.array();
    }
    else if (sym == "quadratic") {
        // a*x² + b*x + c
        Y = (X.array().square().rowwise() * p0.array()
           + X.array().rowwise() * p1.array()).rowwise() + p2.array();
    }
    else if (sym == "sigmoid") {
        // a / (1 + exp(-clip(b*(x-c), -100, 100)))
        auto Z = ((X.array().rowwise() - p2.array()).rowwise() * p1.array())
                     .min(100.0).max(-100.0);
        Y = (1.0 / (1.0 + (-Z).exp())).rowwise() * p0.array();
    }
    else if (sym == "tanh") {
        // a * tanh(clip(b*(x-c), -50, 50))
        auto Z = ((X.array().rowwise() - p2.array()).rowwise() * p1.array())
                     .min(50.0).max(-50.0);
        Y = Z.tanh().rowwise() * p0.array();
    }
    else if (sym == "gaussian") {
        // a * exp(-clip(((x-c)/(|b|+1e-6))², 0, 100))
        Eigen::RowVectorXd b_safe = p1.array().abs() + 1e-6;
        auto Z = ((X.array().rowwise() - p2.array()).rowwise() / b_safe.array())
                     .square().min(100.0).max(0.0);
        Y = (-Z).exp().rowwise() * p0.array();
    }
    else if (sym == "hinge") {
        // a * max(0, x-c) + b
        Y = ((X.array().rowwise() - p2.array()).max(0.0).rowwise() * p0.array())
                .rowwise() + p1.array();
    }
    else if (sym == "sqrt") {
        // a * sqrt(|x| + 1e-8) + b
        Y = ((X.array().abs() + 1e-8).sqrt().rowwise() * p0.array())
                .rowwise() + p1.array();
    }
    else if (sym == "log") {
        // a * log(|x| + 1e-8) + b
        Y = ((X.array().abs() + 1e-8).log().rowwise() * p0.array())
                .rowwise() + p1.array();
    }
    else if (sym == "constant") {
        // scalar constant a per edge (independent of x)
        Y = p0.replicate(n_samp, 1);
    }
    else if (sym == "exp") {
        // a * exp(clip(b*x, -50, 50))
        auto Z = (X.array().rowwise() * p1.array()).min(50.0).max(-50.0);
        Y = Z.exp().rowwise() * p0.array();
    }
    else if (sym == "power") {
        // a * (|x| + 1e-8)^clip(b,-5,5)
        // = a * exp(clip(b,-5,5) * log(|x|+1e-8))
        Eigen::RowVectorXd b_clamped = p1.array().max(-5.0).min(5.0);
        auto log_abs = (X.array().abs() + 1e-8).log();
        Y = (log_abs.rowwise() * b_clamped.array()).exp().rowwise() * p0.array();
    }
    else if (sym == "rational") {
        // a / (|b| + |x| + 1e-8)
        Eigen::RowVectorXd b_abs = p1.array().abs();
        Y = (1.0 / (X.array().abs().rowwise() + b_abs.array() + 1e-8))
                .rowwise() * p0.array();
    }
    else if (sym == "sin") {
        // a * sin(clip(b*x + c, -100, 100))
        auto Z = ((X.array().rowwise() * p1.array()).rowwise() + p2.array())
                     .min(100.0).max(-100.0);
        Y = Z.sin().rowwise() * p0.array();
    }
    else {
        Y.setZero();  // unknown type → zero contribution
    }
    return Y;
}

// ── apply_sym_layer ───────────────────────────────────────────────────────────
Eigen::MatrixXd apply_sym_layer(const SymLayer& layer,
                                const Eigen::MatrixXd& X_in)
{
    const int n_samp  = static_cast<int>(X_in.rows());
    const int out_dim = layer.out_dim;

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_samp, out_dim);

    for (const auto& grp : layer.groups) {
        const int n_group = static_cast<int>(grp.col_indices.size());

        // Fused gather + optional per-edge normalisation in one pass.
        // Avoids allocating a separate X_group intermediate.
        Eigen::MatrixXd X_norm(n_samp, n_group);
        if (grp.normalized) {
            for (int k = 0; k < n_group; ++k)
                X_norm.col(k) = (X_in.col(grp.col_indices[k]).array()
                                 - grp.x_mins(k))
                                / (grp.x_spans(k) + 1e-12);
        } else {
            for (int k = 0; k < n_group; ++k)
                X_norm.col(k) = X_in.col(grp.col_indices[k]);
        }

        // Batched primitive evaluation: (n_samp × n_group) → (n_samp × n_group)
        const Eigen::MatrixXd Y_group = apply_group(grp, X_norm);

        // Sparse routing: scatter-add each edge column to its output column.
        // Replaces the dense GEMM  out += Y_group @ routing
        // (routing has exactly one 1 per row → out_dim× fewer ops).
        for (int k = 0; k < n_group; ++k)
            out.col(grp.out_indices[k]) += Y_group.col(k);
    }

    // Linear calibration: out[i, j] = out[i, j] * scale[j] + bias[j]
    out.array().rowwise() *= layer.scale.transpose().array();
    out.array().rowwise() += layer.bias.transpose().array();

    return out;
}

// ── load_sym_layer ────────────────────────────────────────────────────────────
SymLayer load_sym_layer(const std::string& json_path)
{
    std::ifstream f(json_path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + json_path);
    json doc = json::parse(f);

    SymLayer layer;
    layer.out_dim    = doc["out_dim"].get<int>();
    layer.normalized = doc["normalized"].get<bool>();

    // Calibration vectors
    auto sc = doc["scale"].get<std::vector<double>>();
    auto bs = doc["bias"].get<std::vector<double>>();
    layer.scale = Eigen::Map<Eigen::VectorXd>(sc.data(), sc.size());
    layer.bias  = Eigen::Map<Eigen::VectorXd>(bs.data(), bs.size());

    // Build groups from edge list (group by sym type)
    std::map<std::string, std::vector<json>> buckets;
    for (auto& e : doc["edges"])
        buckets[e["sym"].get<std::string>()].push_back(e);

    for (auto& [sym_type, edge_list] : buckets) {
        SymGroup grp;
        grp.sym_type   = sym_type;
        grp.normalized = layer.normalized;
        int n_group    = static_cast<int>(edge_list.size());

        grp.col_indices.resize(n_group);
        grp.out_indices.resize(n_group);
        grp.params.resize(n_group);
        grp.x_mins  = Eigen::VectorXd(n_group);
        grp.x_spans = Eigen::VectorXd(n_group);

        // Pre-allocate parameter row vectors (all primitives use at most 3)
        grp.p0 = Eigen::RowVectorXd::Zero(n_group);
        grp.p1 = Eigen::RowVectorXd::Zero(n_group);
        grp.p2 = Eigen::RowVectorXd::Zero(n_group);

        for (int k = 0; k < n_group; ++k) {
            auto& e = edge_list[k];
            grp.col_indices[k] = e["in_i"].get<int>();
            grp.out_indices[k] = e["out_j"].get<int>();
            grp.params[k]      = e["params"].get<std::vector<double>>();
            grp.x_mins(k)      = e["x_min"].get<double>();
            grp.x_spans(k)     = e["x_span"].get<double>();

            // Extract into flat row vectors for batched eval
            const auto& p = grp.params[k];
            if (p.size() >= 1) grp.p0(k) = p[0];
            if (p.size() >= 2) grp.p1(k) = p[1];
            if (p.size() >= 3) grp.p2(k) = p[2];
        }
        layer.groups.push_back(std::move(grp));
    }
    return layer;
}
