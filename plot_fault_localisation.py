"""
Generate a publication-quality figure for per-equation fault-violation ratio
with dominant-sensor attribution, for paper_ICIEA2026.tex.

Output: results_kan_symbolic/fault_localisation_figure.pdf
        results_kan_symbolic/fault_localisation_figure.png
"""

import csv, ast
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── 1. Load fault-ratio data ────────────────────────────────────────────────
fl_rows = list(csv.DictReader(
    open("results_kan_symbolic/fault_localization.csv")))
eq_ratio = {}
for r in fl_rows:
    eq_idx = int(r["equation"].replace("z_", ""))
    eq_ratio[eq_idx] = float(r["fault_ratio"])

# ── 2. Load symbolic regression results → per-sensor weight per equation ────
sym_rows = list(csv.DictReader(
    open("results_kan_symbolic/symbolic_regression_results.csv")))

eq_sensor_weight = defaultdict(lambda: defaultdict(float))
eq_top_feats = defaultdict(list)

for r in sym_rows:
    eq = int(r["out_neuron"])
    feat = r["in_feature"]
    sensor = feat.split("_")[0]
    params = ast.literal_eval(r["params"])
    weight = sum(abs(p) for p in params[:2])  # |a| + |b| for quadratic
    eq_sensor_weight[eq][sensor] += weight
    eq_top_feats[eq].append((weight, feat))

# Normalise sensor weights to fractions
eq_sensor_frac = {}
for eq, sw in eq_sensor_weight.items():
    total = sum(sw.values())
    eq_sensor_frac[eq] = {s: sw[s] / total for s in ["S1", "S2", "S3", "S4"]}

# Top feature per equation (by coefficient magnitude)
for eq in eq_top_feats:
    eq_top_feats[eq].sort(reverse=True)

# ── 3. Sort equations by fault ratio ────────────────────────────────────────
eqs_sorted = sorted(eq_ratio.keys(), key=lambda e: eq_ratio[e])  # ascending for horizontal bar
ratios_sorted = [eq_ratio[e] for e in eqs_sorted]
labels_sorted = [f"$z_{{\\!{e}}}$" for e in eqs_sorted]

sensors = ["S1", "S2", "S3", "S4"]
COLORS = {"S1": "#d62728", "S2": "#1f77b4", "S3": "#2ca02c", "S4": "#9467bd"}
LIGHT  = {"S1": "#f5a5a5", "S2": "#a5c8f0", "S3": "#a5dba5", "S4": "#d0bae8"}

# Sensor fraction stacks (sorted same order)
fracs = {s: [eq_sensor_frac[e].get(s, 0.0) for e in eqs_sorted] for s in sensors}

# ── 4. Build figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.8),
                          gridspec_kw={"width_ratios": [1.55, 1]})
fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.12, wspace=0.38)

# ── Panel A: fault ratio bars coloured by dominant sensor ───────────────────
ax = axes[0]
y_pos = np.arange(len(eqs_sorted))
bar_h = 0.72

# Dominant sensor colour for each bar
dom_colors = [COLORS[max(eq_sensor_frac[e], key=eq_sensor_frac[e].get)]
              for e in eqs_sorted]

bars = ax.barh(y_pos, ratios_sorted, height=bar_h,
               color=dom_colors, alpha=0.82, edgecolor="none")
ax.axvline(1.0, color="black", linestyle="--", linewidth=0.9, zorder=3)

# Annotate the three highest-ratio equations with top feature label
top3_pos = sorted(range(len(eqs_sorted)),
                  key=lambda i: ratios_sorted[i], reverse=True)[:3]
for pos in top3_pos:
    eq = eqs_sorted[pos]
    ratio = ratios_sorted[pos]
    top_feat = eq_top_feats[eq][0][1]
    ax.text(ratio + 0.15, pos, f"{top_feat}  ({ratio:.1f}\u00d7)",
            va="center", ha="left", fontsize=5.5, color="#333333")

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_sorted, fontsize=6.5)
ax.set_xlabel("Fault ratio  $\\rho_j$  (broken / healthy)", fontsize=8)
ax.set_title("(a) Per-equation violation ratio", fontsize=8.5, fontweight="bold")
ax.tick_params(axis="x", labelsize=7)
ax.set_xlim(left=0, right=max(ratios_sorted) * 1.22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend for dominant-sensor colour
legend_handles = [mpatches.Patch(color=COLORS[s], alpha=0.82, label=s)
                  for s in sensors]
ax.legend(handles=legend_handles, title="Dom. sensor", fontsize=6,
          title_fontsize=6.5, loc="lower right",
          framealpha=0.85, edgecolor="lightgrey")

# ── Panel B: stacked sensor-weight breakdown for top-8 violated equations ───
ax2 = axes[1]
top_n = 8
top_idxs = sorted(range(len(eqs_sorted)),
                  key=lambda i: ratios_sorted[i], reverse=True)[:top_n]
top_idxs_plot = top_idxs[::-1]  # ascending for horizontal bars

top_eqs   = [eqs_sorted[i] for i in top_idxs_plot]
top_labels = [f"$z_{{\\!{e}}}$ ({eq_ratio[e]:.1f}\u00d7)" for e in top_eqs]
y2 = np.arange(top_n)

left_accum = np.zeros(top_n)
for s in sensors:
    vals = [eq_sensor_frac[top_eqs[k]].get(s, 0.0) for k in range(top_n)]
    ax2.barh(y2, vals, left=left_accum, height=bar_h,
             color=COLORS[s], alpha=0.82, edgecolor="white", linewidth=0.4,
             label=s)
    left_accum += np.array(vals)

ax2.set_yticks(y2)
ax2.set_yticklabels(top_labels, fontsize=6.5)
ax2.set_xlabel("Sensor-weight fraction", fontsize=8)
ax2.set_title("(b) Sensor attribution\n(top-8 violated)", fontsize=8.5, fontweight="bold")
ax2.set_xlim(0, 1.0)
ax2.tick_params(axis="x", labelsize=7)
ax2.legend(fontsize=6, loc="lower right", framealpha=0.85, edgecolor="lightgrey")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Add vertical line at 0.25 (equal share) for reference
ax2.axvline(0.25, color="grey", linestyle=":", linewidth=0.7, alpha=0.6)

# ── 5. Save ──────────────────────────────────────────────────────────────────
out_base = "results_kan_symbolic/fault_localisation_figure"
fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
fig.savefig(out_base + ".png", dpi=200, bbox_inches="tight")
print(f"Saved {out_base}.pdf  and  {out_base}.png")
