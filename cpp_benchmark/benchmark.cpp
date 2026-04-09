// benchmark.cpp ─ C++ inference benchmark for KAN Gearbox paper
//
// Loads exported models and data from exports/ and measures median
// wall-clock time per sample using std::chrono::high_resolution_clock.
//
// Methods benchmarked here (C++):
//   IsolationForest, OC-SVM, LOF  (pure Eigen — no ONNX needed)
//   PatchCore, AE, VAE, DeepSVDD, TeacherStudent, SSL-DAE
//   SHAP-Weighted-Recon, LIME-Weighted-Recon  (KernelSHAP + LIME via LibTorch)
//   KAN-AE-Recon (A), KAN-AE-Symbolic (B), KAN-AE-Combined (C),
//   KAN-AE-Symbolic-D, KAN-AE-Combined-E, KAN-AE-Mahal (M),
//   KAN-AE-Combined-F, KAN-AE-SymRecon (SR, pure forward — no MSE)
//
// Build:
//   See CMakeLists.txt.  Pass -DCMAKE_PREFIX_PATH=".../libtorch" to cmake.
//
// Output:
//   results_cpp_inference.csv   (Method, ms_per_sample)

#include <torch/script.h>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include <nlohmann/json.hpp>
#include "sym_eval.h"

using json = nlohmann::json;

// ── Config ────────────────────────────────────────────────────────────────────
static const int         WARMUP   = 5;
static const int         REPS     = 20;
static const std::string EXPORTS  = "exports/";
static const std::string OUT_CSV  = "results_cpp_inference.csv";

// ── Timing ────────────────────────────────────────────────────────────────────
// Returns the median total batch time in milliseconds over REPS repetitions.
// Divide by n_test to get ms/sample.
double time_batch(std::function<void()> fn,
                  int warmup = WARMUP,
                  int reps   = REPS)
{
    for (int i = 0; i < warmup; ++i)
        fn();
    std::vector<double> ts(reps);
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        ts[i]   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(ts.begin(), ts.end());
    return ts[reps / 2];   // median
}

// ── Binary I/O ───────────────────────────────────────────────────────────────
// Read a row-major float32 binary file into a RowMajor Eigen matrix.
Eigen::MatrixXf load_f32(const std::string& path, int rows, int cols)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    std::vector<float> buf(rows * cols);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(rows * cols * sizeof(float)));
    // Map as RowMajor; then copy into a default ColMajor matrix for Eigen ops.
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        M(buf.data(), rows, cols);
    return M;   // implicit conversion to ColMajor
}

// Read an int32 binary file as a plain vector.
std::vector<int32_t> load_i32(const std::string& path, int count)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    std::vector<int32_t> v(count);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(count * sizeof(int32_t)));
    return v;
}

// Convert a ColMajor Eigen float matrix to a LibTorch CPU tensor (row-major).
at::Tensor to_tensor(const Eigen::MatrixXf& M)
{
    // Eigen is ColMajor; torch expects row-major (C-contiguous).
    // We explicitly transpose and contiguify.
    at::Tensor t = torch::from_blob(
        const_cast<float*>(M.data()),
        {M.cols(), M.rows()},   // Eigen stores col-major: cols first in memory
        torch::TensorOptions().dtype(torch::kFloat32)
    ).t().contiguous();         // transpose → (rows, cols) in row-major
    return t;
}

// ── PatchCore ─────────────────────────────────────────────────────────────────
// Minimum L2 distance from each row of X_test to the nearest coreset point.
// Uses the identity ||x-c||² = ||x||² + ||c||² - 2 x·c for efficiency.
static void run_patchcore(const Eigen::MatrixXf& X_test,
                           const Eigen::MatrixXf& coreset)
{
    Eigen::ArrayXf xx = X_test.rowwise().squaredNorm().array();   // (n_test,)
    Eigen::ArrayXf cc = coreset.rowwise().squaredNorm().array();   // (n_core,)
    // D2[i,j] = ||x_i||² + ||c_j||² - 2 x_i·c_j
    Eigen::ArrayXXf D2 = (-2.0f * (X_test * coreset.transpose())).array();
    D2.colwise() += xx;     // xx[i] broadcast across columns (per row)
    D2.rowwise() += cc.transpose();  // cc[j] broadcast across rows (per col)
    D2 = D2.max(0.0f);
    // min-distance per row (discard result; computation must not be elided)
    volatile float sink = D2.rowwise().minCoeff().sum();
    (void)sink;
}

// ── Mahalanobis helper ────────────────────────────────────────────────────────
// mu is a row vector (1 × latent_dim) so rowwise() subtraction is direct.
static void run_mahal(const Eigen::MatrixXf& Z,
                       const Eigen::RowVectorXf& mu,
                       const Eigen::MatrixXf& cov_inv)
{
    Eigen::MatrixXf diff = Z.rowwise() - mu;          // (n_test × latent_dim)
    Eigen::MatrixXf temp = diff * cov_inv;             // (n_test × latent_dim)
    // score[i] = diff[i,:] @ cov_inv @ diff[i,:]^T
    volatile float sink = (temp.array() * diff.array()).rowwise().sum().sum();
    (void)sink;
}

// ── Symbolic forward-pass sink ────────────────────────────────────────────────
// Prevents the compiler from dead-code-eliminating the symbolic computation.
static void sink_matrix(const Eigen::MatrixXd& M)
{
    volatile double s = M.sum();
    (void)s;
}

// ── IsolationForest structs + scoring ─────────────────────────────────────────
struct IFNode {
    int   feature;      // -2 = leaf
    float threshold;
    int   left_child;   // -1 = leaf sentinel
    int   right_child;
    int   n_samples;
};
struct IFTree { std::vector<IFNode> nodes; };

// c(n): expected path length for a BST built from n samples
static double if_cn(double n) {
    if (n <= 1.0) return 0.0;
    return 2.0 * (std::log(n - 1.0) + 0.5772156649015329) - 2.0 * (n - 1.0) / n;
}

// Average path length for row x through one tree (row-major pointer, stride=1)
static double if_path(const IFTree& tree, const float* x) {
    int node = 0, depth = 0;
    while (tree.nodes[node].feature >= 0) {
        node = (x[tree.nodes[node].feature] <= tree.nodes[node].threshold)
               ? tree.nodes[node].left_child
               : tree.nodes[node].right_child;
        ++depth;
    }
    return depth + if_cn(static_cast<double>(tree.nodes[node].n_samples));
}

static std::vector<IFTree> load_if_trees(const std::string& path,
                                          int& n_samp_per_tree)
{
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    json doc = json::parse(f);
    n_samp_per_tree = doc["n_samples_per_tree"].get<int>();
    std::vector<IFTree> trees;
    for (auto& jt : doc["trees"]) {
        IFTree tree;
        auto feat  = jt["node_feature"].get<std::vector<int>>();
        auto thr   = jt["threshold"].get<std::vector<float>>();
        auto cl    = jt["children_left"].get<std::vector<int>>();
        auto cr    = jt["children_right"].get<std::vector<int>>();
        auto ns    = jt["n_node_samples"].get<std::vector<int>>();
        int  nn    = static_cast<int>(feat.size());
        tree.nodes.resize(nn);
        for (int k = 0; k < nn; ++k)
            tree.nodes[k] = {feat[k], thr[k], cl[k], cr[k], ns[k]};
        trees.push_back(std::move(tree));
    }
    return trees;
}

static void run_if(const Eigen::MatrixXf& X_test,
                   const std::vector<IFTree>& trees,
                   int n_samp_per_tree)
{
    // X_test is ColMajor; need row access → use RowMajor map
    int n_test  = static_cast<int>(X_test.rows());
    int n_feat  = static_cast<int>(X_test.cols());
    int n_trees = static_cast<int>(trees.size());
    double cn   = if_cn(static_cast<double>(n_samp_per_tree));

    // Copy to RowMajor buffer for cache-friendly row access
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        X_rm = X_test;

    double score_sum = 0.0;
    for (int i = 0; i < n_test; ++i) {
        const float* xi = X_rm.data() + i * n_feat;
        double path_sum = 0.0;
        for (const auto& t : trees) path_sum += if_path(t, xi);
        score_sum += std::pow(2.0, -(path_sum / n_trees) / cn);
    }
    volatile double sink = score_sum;
    (void)sink;
}

// ── OC-SVM scoring (RBF kernel) ───────────────────────────────────────────────
static void run_svm(const Eigen::MatrixXf& X_test,
                    const Eigen::MatrixXf& sv,      // (n_sv × n_feat)
                    const Eigen::VectorXf& alpha,   // (n_sv)
                    float gamma,
                    float intercept)
{
    // K[i,j] = exp(-gamma * ||x_i - sv_j||^2)
    //         = exp(-gamma * (||x_i||^2 + ||sv_j||^2 - 2 x_i·sv_j))
    Eigen::ArrayXf xx = X_test.rowwise().squaredNorm().array();
    Eigen::ArrayXf ss = sv.rowwise().squaredNorm().array();
    Eigen::ArrayXXf D2 = (-2.0f * (X_test * sv.transpose())).array();
    D2.colwise() += xx;
    D2.rowwise() += ss.transpose();
    Eigen::MatrixXf K = (-gamma * D2.max(0.0f)).exp().matrix();
    // score = K @ alpha + intercept
    Eigen::VectorXf scores = K * alpha +
                             Eigen::VectorXf::Constant(X_test.rows(), intercept);
    volatile float sink = scores.sum();
    (void)sink;
}

// ── LOF scoring (brute-force k-NN) ───────────────────────────────────────────
static void run_lof(const Eigen::MatrixXf& X_test,
                    const Eigen::MatrixXf& fit_X,    // (n_train × n_feat)
                    const Eigen::VectorXf& lrdof,    // (n_train) lrd of train pts
                    const Eigen::VectorXf& kdist,    // (n_train) k-dist of train pts
                    int k)
{
    int n_test  = static_cast<int>(X_test.rows());
    int n_train = static_cast<int>(fit_X.rows());

    // Pairwise L2 distance matrix (n_test × n_train)
    Eigen::ArrayXf xx = X_test.rowwise().squaredNorm().array();
    Eigen::ArrayXf tt = fit_X.rowwise().squaredNorm().array();
    Eigen::ArrayXXf D2 = (-2.0f * (X_test * fit_X.transpose())).array();
    D2.colwise() += xx;
    D2.rowwise() += tt.transpose();
    Eigen::MatrixXf D = D2.max(0.0f).sqrt().matrix();   // (n_test × n_train)

    double lof_sum = 0.0;
    std::vector<int> nbr_idx(k);
    std::vector<float> nbr_dist(k);

    for (int i = 0; i < n_test; ++i) {
        // Partial sort: find k nearest training points
        Eigen::RowVectorXf row = D.row(i);
        std::iota(nbr_idx.begin(), nbr_idx.end(), 0);
        // Initialise with first k
        nbr_idx.resize(n_train);
        std::iota(nbr_idx.begin(), nbr_idx.end(), 0);
        std::partial_sort(nbr_idx.begin(), nbr_idx.begin() + k, nbr_idx.end(),
            [&](int a, int b){ return row(a) < row(b); });
        nbr_idx.resize(k);

        // reach_dist(test_i, j) = max(dist(i,j), kdist[j])
        float sum_rd = 0.0f;
        float sum_lrd = 0.0f;
        for (int j : nbr_idx) {
            float rd = std::max(row(j), kdist(j));
            sum_rd  += rd;
            sum_lrd += lrdof(j);
        }
        float lrd_test = (sum_rd > 0.0f) ? (float)k / sum_rd : 1e9f;
        float lof_test = (sum_lrd / k) / (lrd_test + 1e-12f);
        lof_sum += lof_test;
    }
    volatile double sink = lof_sum;
    (void)sink;
}

// ── KernelSHAP (200 coalitions, KAN-AE forward, weighted least-squares) ──────
static void run_shap(const Eigen::MatrixXf& xai_sample,    // (1 × n_feat)
                     const Eigen::MatrixXf& bg,             // (n_bg × n_feat)
                     torch::jit::Module& kan_ae,
                     std::mt19937& rng,
                     int nsamples = 200)
{
    int d    = static_cast<int>(xai_sample.cols());
    int n_bg = static_cast<int>(bg.rows());
    std::uniform_int_distribution<int>  bg_dist(0, n_bg - 1);
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    // Generate perturbed samples: z*x + (1-z)*bg[rand]
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        perturbed(nsamples, d);
    Eigen::MatrixXf masks(nsamples, d);
    for (int i = 0; i < nsamples; ++i) {
        int bi = bg_dist(rng);
        for (int j = 0; j < d; ++j) {
            float m = (coin(rng) < 0.5f) ? 1.0f : 0.0f;
            masks(i, j)    = m;
            perturbed(i,j) = m * xai_sample(0,j) + (1.0f-m) * bg(bi,j);
        }
    }

    // Convert to LibTorch tensor (row-major buffer)
    at::Tensor inp = torch::from_blob(perturbed.data(), {nsamples, d},
                                      torch::kFloat32).clone();
    at::Tensor out = kan_ae.forward({inp}).toTensor().contiguous();

    // Reconstruction errors → model output y (nsamples,)
    int d_out = static_cast<int>(out.size(1));
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>>
        out_map(out.data_ptr<float>(), nsamples, d_out);
    // Compare perturbed input to reconstruction output
    int d_cmp = std::min(d, d_out);
    Eigen::VectorXf y = (perturbed.leftCols(d_cmp) - out_map.leftCols(d_cmp))
                        .array().square().rowwise().mean();

    // Kernel weights: simplified SHAP kernel (d-1) / (C(d,|z|) * |z| * (d-|z|))
    Eigen::VectorXf z_sum = masks.rowwise().sum();
    Eigen::VectorXf weights(nsamples);
    for (int i = 0; i < nsamples; ++i) {
        float s = z_sum(i);
        if (s <= 0.0f || s >= (float)d) { weights(i) = 1000.0f; continue; }
        weights(i) = (float)(d - 1) / (s * ((float)d - s));
    }

    // Weighted least-squares: design matrix [masks | 1] (nsamples × (d+1))
    Eigen::MatrixXf A(nsamples, d + 1);
    A.leftCols(d) = masks;
    A.rightCols(1).setOnes();
    Eigen::VectorXf ws = weights.array().sqrt();
    Eigen::MatrixXf Aw = A.array().colwise() * ws.array();
    Eigen::VectorXf yw = y.array() * ws.array();
    Eigen::VectorXf phi = (Aw.transpose() * Aw).ldlt().solve(Aw.transpose() * yw);

    volatile float sink = phi.sum();
    (void)sink;
}

// ── LIME tabular (300 perturbations, KAN-AE forward, weighted Ridge) ──────────
static void run_lime(const Eigen::MatrixXf& xai_sample,    // (1 × n_feat)
                     const Eigen::MatrixXf& tr_stats,      // (2 × n_feat): row0=mean, row1=std
                     torch::jit::Module& kan_ae,
                     std::mt19937& rng,
                     int nsamples = 300)
{
    int d = static_cast<int>(xai_sample.cols());
    std::normal_distribution<float> stdnorm(0.0f, 1.0f);
    Eigen::RowVectorXf feat_std = tr_stats.row(1);   // (n_feat,)

    // Generate perturbations: x + N(0, std)
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        perturbed(nsamples, d);
    for (int i = 0; i < nsamples; ++i)
        for (int j = 0; j < d; ++j)
            perturbed(i, j) = xai_sample(0, j) + stdnorm(rng) * feat_std(j);

    // Run KAN model
    at::Tensor inp = torch::from_blob(perturbed.data(), {nsamples, d},
                                      torch::kFloat32).clone();
    at::Tensor out = kan_ae.forward({inp}).toTensor().contiguous();
    int d_out = static_cast<int>(out.size(1));
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>>
        out_map(out.data_ptr<float>(), nsamples, d_out);
    int d_cmp = std::min(d, d_out);
    Eigen::VectorXf y = (perturbed.leftCols(d_cmp) - out_map.leftCols(d_cmp))
                        .array().square().rowwise().mean();

    // Kernel weights: cosine distance → exp(-d^2 / kernel_width^2)
    // kernel_width = sqrt(d) * 0.75 (LIME default)
    float kw = std::sqrt((float)d) * 0.75f;
    Eigen::MatrixXf diff = perturbed.rowwise() - xai_sample.row(0);
    Eigen::VectorXf dists = diff.rowwise().norm();
    Eigen::VectorXf weights = (-(dists.array().square()) / (kw * kw)).exp();

    // Weighted Ridge regression: perturbed @ beta = y  with lambda=1
    float lambda = 1.0f;
    Eigen::VectorXf ws = weights.array().sqrt();
    Eigen::MatrixXf Aw = perturbed.array().colwise() * ws.array();
    Eigen::VectorXf yw = y.array() * ws.array();
    Eigen::MatrixXf AtA = Aw.transpose() * Aw;
    AtA.diagonal().array() += lambda;
    Eigen::VectorXf beta = AtA.ldlt().solve(Aw.transpose() * yw);

    volatile float sink = beta.sum();
    (void)sink;
}

// ══════════════════════════════════════════════════════════════════════════════
int main()
{
    torch::NoGradGuard no_grad;

    // ── Load metadata ─────────────────────────────────────────────────────────
    auto shape = load_i32(EXPORTS + "test_shape.bin", 2);
    const int n_test     = shape[0];
    const int n_features = shape[1];
    std::printf("n_test=%d  n_features=%d\n", n_test, n_features);

    auto mahal_dim_v = load_i32(EXPORTS + "mahal_dim.bin", 1);
    const int latent_dim = mahal_dim_v[0];  // 22

    // ── Load test data ────────────────────────────────────────────────────────
    Eigen::MatrixXf X_mm  = load_f32(EXPORTS + "test_data_mm.bin",  n_test, n_features);
    Eigen::MatrixXf X_std = load_f32(EXPORTS + "test_data_std.bin", n_test, n_features);
    Eigen::MatrixXd X_mm_f64 = X_mm.cast<double>();

    // LibTorch tensors (row-major, float32)
    at::Tensor Xte_mm  = to_tensor(X_mm);
    at::Tensor Xte_std = to_tensor(X_std);

    // ── Load PatchCore coreset ────────────────────────────────────────────────
    auto core_shape = load_i32(EXPORTS + "coreset_shape.bin", 2);
    Eigen::MatrixXf coreset = load_f32(EXPORTS + "coreset.bin",
                                        core_shape[0], core_shape[1]);

    // ── Load Mahalanobis parameters ───────────────────────────────────────────
    Eigen::MatrixXf    mu_mat  = load_f32(EXPORTS + "mahal_mu.bin",      1, latent_dim);
    Eigen::MatrixXf    cov_inv = load_f32(EXPORTS + "mahal_cov_inv.bin", latent_dim, latent_dim);
    Eigen::RowVectorXf mu      = mu_mat.row(0);   // (1 × latent_dim)

    // ── Load symbolic layers ──────────────────────────────────────────────────
    SymLayer sym_l0 = load_sym_layer(EXPORTS + "sym_layer0.json");
    SymLayer sym_l1 = load_sym_layer(EXPORTS + "sym_layer1.json");
    SymLayer sym_l2 = load_sym_layer(EXPORTS + "sym_layer2.json");
    SymLayer sym_l3 = load_sym_layer(EXPORTS + "sym_layer3.json");

    // ── Load classical sklearn model parameters ───────────────────────────────
    int if_n_samp_per_tree = 0;
    std::vector<IFTree> if_trees = load_if_trees(EXPORTS + "if_trees.json",
                                                  if_n_samp_per_tree);
    std::printf("IF: %zu trees, %d samples/tree\n",
                if_trees.size(), if_n_samp_per_tree);

    auto svm_sv_shape = load_i32(EXPORTS + "svm_sv_shape.bin", 2);
    int n_sv = svm_sv_shape[0];
    Eigen::MatrixXf svm_sv    = load_f32(EXPORTS + "svm_sv.bin",       n_sv, n_features);
    Eigen::VectorXf svm_alpha = load_f32(EXPORTS + "svm_dual_coef.bin", 1,    n_sv).row(0);
    float svm_gamma, svm_intercept;
    {
        std::ifstream jf(EXPORTS + "svm_params.json");
        json jd = json::parse(jf);
        svm_gamma     = static_cast<float>(jd["gamma"].get<double>());
        svm_intercept = static_cast<float>(jd["intercept"].get<double>());
    }
    std::printf("SVM: %d support vectors, gamma=%.5f\n", n_sv, svm_gamma);

    auto lof_shape = load_i32(EXPORTS + "lof_shape.bin", 2);
    int n_lof_train = lof_shape[0];
    Eigen::MatrixXf lof_fit_X = load_f32(EXPORTS + "lof_fit_X.bin", n_lof_train, n_features);
    Eigen::VectorXf lof_lrdof = load_f32(EXPORTS + "lof_lrdof.bin", 1, n_lof_train).row(0);
    Eigen::VectorXf lof_kdist = load_f32(EXPORTS + "lof_kdist.bin", 1, n_lof_train).row(0);
    int lof_k;
    {
        std::ifstream jf(EXPORTS + "lof_params.json");
        json jd = json::parse(jf);
        lof_k = jd["n_neighbors"].get<int>();
    }
    std::printf("LOF: %d training pts, k=%d\n", n_lof_train, lof_k);

    // ── Load SHAP / LIME data ─────────────────────────────────────────────────
    auto bg_shape = load_i32(EXPORTS + "shap_bg_shape.bin", 2);
    Eigen::MatrixXf shap_bg   = load_f32(EXPORTS + "shap_bg.bin",  bg_shape[0], bg_shape[1]);
    Eigen::MatrixXf xai_sample= load_f32(EXPORTS + "xai_sample.bin", 1, n_features);
    Eigen::MatrixXf lime_stats= load_f32(EXPORTS + "lime_tr_stats.bin", 2, n_features);
    std::mt19937 rng_xai(42);

    // ── Load TorchScript models ───────────────────────────────────────────────
    auto load_mod = [](const std::string& p) {
        auto m = torch::jit::load(p);
        m.eval();
        return m;
    };
    auto mod_ae          = load_mod(EXPORTS + "ae.pt");
    auto mod_vae         = load_mod(EXPORTS + "vae.pt");
    auto mod_svdd        = load_mod(EXPORTS + "svdd.pt");
    auto mod_ts          = load_mod(EXPORTS + "ts.pt");
    auto mod_dae         = load_mod(EXPORTS + "dae.pt");
    auto mod_kan_ae      = load_mod(EXPORTS + "kan_ae.pt");
    auto mod_kan_l0      = load_mod(EXPORTS + "kan_layer0.pt");
    auto mod_kan_l1to3   = load_mod(EXPORTS + "kan_layers1to3.pt");

    // ── Results accumulator ───────────────────────────────────────────────────
    std::vector<std::pair<std::string, double>> results;
    auto record = [&](const std::string& name, double ms_per_sample) {
        results.push_back({name, ms_per_sample});
        std::printf("  %-30s %.6f ms/sample\n", name.c_str(), ms_per_sample);
    };

    std::printf("\n%-30s  %s\n", "Method", "ms/sample (C++)");
    std::printf("%s\n", std::string(50, '-').c_str());

    // ── 1. IsolationForest ────────────────────────────────────────────────────
    {
        double ms = time_batch([&]() { run_if(X_std, if_trees, if_n_samp_per_tree); });
        record("IsolationForest", ms / n_test);
    }

    // ── 2. OC-SVM ─────────────────────────────────────────────────────────────
    {
        double ms = time_batch([&]() {
            run_svm(X_std, svm_sv, svm_alpha, svm_gamma, svm_intercept);
        });
        record("OC-SVM", ms / n_test);
    }

    // ── 3. LOF ────────────────────────────────────────────────────────────────
    {
        double ms = time_batch([&]() {
            run_lof(X_std, lof_fit_X, lof_lrdof, lof_kdist, lof_k);
        });
        record("LOF", ms / n_test);
    }

    // ── 4. PatchCore ──────────────────────────────────────────────────────────
    {
        double ms = time_batch([&]() { run_patchcore(X_std, coreset); });
        record("PatchCore", ms / n_test);
    }

    // ── 5. Autoencoder ────────────────────────────────────────────────────────
    {
        std::vector<torch::jit::IValue> inp = {Xte_std};
        double ms = time_batch([&]() { mod_ae.forward(inp); });
        record("Autoencoder", ms / n_test);
    }

    // ── 6. VAE ────────────────────────────────────────────────────────────────
    {
        std::vector<torch::jit::IValue> inp = {Xte_std};
        double ms = time_batch([&]() { mod_vae.forward(inp); });
        record("VAE", ms / n_test);
    }

    // ── 7. DeepSVDD ───────────────────────────────────────────────────────────
    {
        std::vector<torch::jit::IValue> inp = {Xte_std};
        double ms = time_batch([&]() { mod_svdd.forward(inp); });
        record("DeepSVDD", ms / n_test);
    }

    // ── 8. Teacher-Student ────────────────────────────────────────────────────
    // Wrapper runs both networks; anomaly score = discrepancy between outputs.
    {
        std::vector<torch::jit::IValue> inp = {Xte_std};
        double ms = time_batch([&]() { mod_ts.forward(inp); });
        record("TeacherStudent", ms / n_test);
    }

    // ── 9. SSL-DAE ────────────────────────────────────────────────────────────
    {
        std::vector<torch::jit::IValue> inp = {Xte_std};
        double ms = time_batch([&]() { mod_dae.forward(inp); });
        record("SSL-DAE", ms / n_test);
    }

    // ── 10. SHAP-Weighted-Recon ───────────────────────────────────────────────
    // KernelSHAP: 200 coalition samples → KAN-AE forward → weighted least-squares.
    // Times full per-sample explanation (matches Python benchmark_inference.py).
    {
        rng_xai.seed(42);
        double ms = time_batch([&]() {
            run_shap(xai_sample, shap_bg, mod_kan_ae, rng_xai);
        }, /*warmup=*/3, /*reps=*/10);
        record("SHAP-Weighted-Recon", ms);   // already per-sample (1 sample explained)
    }

    // ── 11. LIME-Weighted-Recon ───────────────────────────────────────────────
    // LIME tabular: 300 perturbations → KAN-AE forward → weighted Ridge regression.
    {
        rng_xai.seed(42);
        double ms = time_batch([&]() {
            run_lime(xai_sample, lime_stats, mod_kan_ae, rng_xai);
        }, /*warmup=*/3, /*reps=*/10);
        record("LIME-Weighted-Recon", ms);   // per-sample
    }

    // ── 12. KAN-AE-Recon (A) — full KAN forward pass only ────────────────────
    {
        std::vector<torch::jit::IValue> inp = {Xte_mm};
        double ms = time_batch([&]() { mod_kan_ae.forward(inp); });
        record("KAN-AE-Recon (A)", ms / n_test);
    }

    // ── 13. KAN-AE-Symbolic (B) ───────────────────────────────────────────────
    // layer0(X_mm) → h1_torch; sym_l0(X_mm) → h1_sym; score = Σ(h1_torch-h1_sym)²
    {
        std::vector<torch::jit::IValue> inp = {Xte_mm};
        double ms = time_batch([&]() {
            // PyTorch layer-0 forward
            at::Tensor h1_t = mod_kan_l0.forward(inp).toTensor().contiguous();
            int n = h1_t.size(0), d = h1_t.size(1);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                h1_map(h1_t.data_ptr<float>(), n, d);
            Eigen::MatrixXd h1_torch = h1_map.cast<double>();

            // Symbolic layer-0 forward (no torch)
            Eigen::MatrixXd h1_sym = apply_sym_layer(sym_l0, X_mm_f64);

            // Score: Σ squared diff per sample
            volatile double sink = (h1_torch - h1_sym).array().square()
                                                        .rowwise().sum().sum();
            (void)sink;
        });
        record("KAN-AE-Symbolic (B)", ms / n_test);
    }

    // ── 14. KAN-AE-Combined (C) ───────────────────────────────────────────────
    // layer0 → h1; layers1to3(h1) → recon;  score = 0.5*nA + 0.5*nB
    // (normalization constants mn_A, mx_A, mn_B, mx_B are training-time
    //  calibration values; their computation is NOT timed here — same as Python)
    {
        std::vector<torch::jit::IValue> inp_mm = {Xte_mm};
        double ms = time_batch([&]() {
            // layer-0 forward (captured once)
            at::Tensor h1_t = mod_kan_l0.forward(inp_mm).toTensor().contiguous();
            // layers 1-3 forward from h1
            std::vector<torch::jit::IValue> inp_h1 = {h1_t};
            at::Tensor recon_t = mod_kan_l1to3.forward(inp_h1).toTensor().contiguous();

            int n = h1_t.size(0), d1 = h1_t.size(1), d3 = recon_t.size(1);

            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                h1_map(h1_t.data_ptr<float>(), n, d1);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                recon_map(recon_t.data_ptr<float>(), n, d3);

            Eigen::MatrixXd h1_torch = h1_map.cast<double>();
            Eigen::MatrixXf recon    = recon_map;

            // Symbolic layer-0 eval
            Eigen::MatrixXd h1_sym = apply_sym_layer(sym_l0, X_mm_f64);

            // sA = mean squared recon error per sample
            Eigen::VectorXf sA = (X_mm - recon).array().square().rowwise().mean();
            // sB = sum squared sym discrepancy per sample
            Eigen::VectorXd sB = (h1_torch - h1_sym).array().square().rowwise().sum();

            // Combined score (normalisation is a scalar op; included for fairness)
            volatile double sink = (sA.cast<double>() + sB).sum();
            (void)sink;
        });
        record("KAN-AE-Combined (C)", ms / n_test);
    }

    // ── 15. KAN-AE-Symbolic-D (same inference path as B) ─────────────────────
    {
        std::vector<torch::jit::IValue> inp = {Xte_mm};
        double ms = time_batch([&]() {
            at::Tensor h1_t = mod_kan_l0.forward(inp).toTensor().contiguous();
            int n = h1_t.size(0), d = h1_t.size(1);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                h1_map(h1_t.data_ptr<float>(), n, d);
            Eigen::MatrixXd h1_torch = h1_map.cast<double>();
            Eigen::MatrixXd h1_sym   = apply_sym_layer(sym_l0, X_mm_f64);
            volatile double sink = (h1_torch - h1_sym).array().square()
                                                        .rowwise().sum().sum();
            (void)sink;
        });
        record("KAN-AE-Symbolic-D", ms / n_test);
    }

    // ── 16. KAN-AE-Combined-E (same inference path as C) ─────────────────────
    {
        std::vector<torch::jit::IValue> inp_mm = {Xte_mm};
        double ms = time_batch([&]() {
            at::Tensor h1_t = mod_kan_l0.forward(inp_mm).toTensor().contiguous();
            std::vector<torch::jit::IValue> inp_h1 = {h1_t};
            at::Tensor recon_t = mod_kan_l1to3.forward(inp_h1).toTensor().contiguous();
            int n = h1_t.size(0), d1 = h1_t.size(1), d3 = recon_t.size(1);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                h1_map(h1_t.data_ptr<float>(), n, d1);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                recon_map(recon_t.data_ptr<float>(), n, d3);
            Eigen::MatrixXd h1_torch = h1_map.cast<double>();
            Eigen::MatrixXf recon    = recon_map;
            Eigen::MatrixXd h1_sym   = apply_sym_layer(sym_l0, X_mm_f64);
            Eigen::VectorXf sA = (X_mm - recon).array().square().rowwise().mean();
            Eigen::VectorXd sB = (h1_torch - h1_sym).array().square().rowwise().sum();
            volatile double sink = (sA.cast<double>() + sB).sum();
            (void)sink;
        });
        record("KAN-AE-Combined-E", ms / n_test);
    }

    // ── 17. KAN-AE-Mahal (M) — layer0 forward + Mahalanobis ─────────────────
    {
        std::vector<torch::jit::IValue> inp = {Xte_mm};
        double ms = time_batch([&]() {
            at::Tensor z_t = mod_kan_l0.forward(inp).toTensor().contiguous();
            int n = z_t.size(0), d = z_t.size(1);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                z_map(z_t.data_ptr<float>(), n, d);
            Eigen::MatrixXf Z = z_map;
            run_mahal(Z, mu, cov_inv);
        });
        record("KAN-AE-Mahal (M)", ms / n_test);
    }

    // ── 18. KAN-AE-Combined-F — layer0 + layers1to3 + Mahalanobis ────────────
    {
        std::vector<torch::jit::IValue> inp_mm = {Xte_mm};
        double ms = time_batch([&]() {
            at::Tensor z_t = mod_kan_l0.forward(inp_mm).toTensor().contiguous();
            std::vector<torch::jit::IValue> inp_z = {z_t};
            at::Tensor recon_t = mod_kan_l1to3.forward(inp_z).toTensor().contiguous();
            int n = z_t.size(0), dz = z_t.size(1), dr = recon_t.size(1);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                z_map(z_t.data_ptr<float>(), n, dz);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                recon_map(recon_t.data_ptr<float>(), n, dr);
            Eigen::MatrixXf Z     = z_map;
            Eigen::MatrixXf recon = recon_map;
            Eigen::VectorXf sA = (X_mm - recon).array().square().rowwise().mean();
            run_mahal(Z, mu, cov_inv);
            // Combined arithmetic (O(n_test), same as Python)
            volatile float sink = sA.sum();
            (void)sink;
        });
        record("KAN-AE-Combined-F", ms / n_test);
    }

    // ── 19. KAN-AE-SymRecon (SR) — pure symbolic forward, NO MSE ─────────────
    // Times only the 4-layer symbolic pass: identical role to a neural forward pass.
    // MSE reconstruction error is excluded per paper design (pure 1:1 comparison).
    {
        double ms = time_batch([&]() {
            Eigen::MatrixXd h1 = apply_sym_layer(sym_l0, X_mm_f64);
            Eigen::MatrixXd h2 = apply_sym_layer(sym_l1, h1);
            Eigen::MatrixXd h3 = apply_sym_layer(sym_l2, h2);
            Eigen::MatrixXd h4 = apply_sym_layer(sym_l3, h3);
            sink_matrix(h4);  // prevent dead-code elimination
        });
        record("KAN-AE-SymRecon", ms / n_test);
    }

    // ── Save results CSV ─────────────────────────────────────────────────────
    {
        std::ofstream csv(OUT_CSV);
        csv << "Method,ms_per_sample_cpp\n";
        for (auto& [name, ms] : results)
            csv << name << "," << std::fixed << std::setprecision(7) << ms << "\n";
        std::printf("\nSaved → %s\n", OUT_CSV.c_str());
    }

    // ── Summary table ────────────────────────────────────────────────────────
    std::printf("\n%-30s  %s\n", "Method", "ms/sample (C++)");
    std::printf("%s\n", std::string(50, '-').c_str());
    for (auto& [name, ms] : results)
        std::printf("  %-28s  %.6f\n", name.c_str(), ms);

    return 0;
}
