"""Render Fig.1-style dynamic comparison figures (N=10 and N=20).

For BC (Ours) we use the DYNAMIC-TRAINED BC (teacher collected in a moving
env at speed=10), which removes the high-speed degradation seen with the
static-trained BC. All other algorithms are unchanged from the original
Fig.1 (random, dist_coloring, periodic_greedy from the centralized sweep,
and CARLTON as the CTDE RL baseline).

Usage:
    python render_fig1.py 10      # render N=10 (fig1_dynamic_comparison.pdf)
    python render_fig1.py 20      # render N=20 (fig1_dynamic_comparison_N20.pdf)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "results_data")
FIG_DIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def gs(dyn, speed, field):
    """Read a field from a dynamic result dict by speed (handles speed_X / speed_X.0)."""
    if not isinstance(dyn, dict) or "results" not in dyn:
        return np.nan
    res = dyn["results"]
    for k in (f"speed_{speed}", f"speed_{speed}.0", f"speed_{int(speed)}"):
        if k in res:
            return res[k].get(field, np.nan)
    return np.nan


def render(N):
    speeds = [0, 10, 20, 30, 40]
    methods_order = ["random", "dist_coloring", "periodic_greedy_k10", "periodic_greedy"]
    method_labels = {
        "random": "Random",
        "dist_coloring": "Dist-Color",
        "periodic_greedy_k10": "Periodic Greedy (k=10)",
        "periodic_greedy": "Periodic Greedy (k=30)",
        "iql": "IQL (no expert)",
    }
    method_colors = {
        "random": "#888888",
        "dist_coloring": "#4daf4a",
        "periodic_greedy_k10": "#9ecae1",
        "periodic_greedy": "#377eb8",
        "iql": "#e41a1c",
    }

    # Distributed baselines (random, dist_coloring, dist_greedy) from dist files
    baseline_data = {}
    for speed in speeds:
        data = load_json(f"{DATA_DIR}/dist_N{N}_speed{speed}.json")
        for s in data["summary"]:
            m = s["method"]
            if m not in baseline_data:
                baseline_data[m] = {"probs": [], "stds": []}
            baseline_data[m]["probs"].append(s["cum_interf_prob"])
            baseline_data[m]["stds"].append(s["std"])

    # Centralized periodic greedy (k=30 and k=10) from the interval sweep
    cg = load_json(f"{DATA_DIR}/centralized_interval_sweep.json")
    for REPLAN_K, key in [(30, "periodic_greedy"), (10, "periodic_greedy_k10")]:
        periodic = {"probs": [], "stds": []}
        for speed in speeds:
            sp = cg[str(N)][str(speed)][str(REPLAN_K)]
            periodic["probs"].append(sp["p_int"])
            periodic["stds"].append(sp["std"])
        baseline_data[key] = periodic

    # Dynamic-trained BC (Ours) -- teacher collected in a moving env.
    # Prefer the multi-speed domain-randomized BC (bc_dynamic_ms) when available,
    # fall back to the single-speed (s10) version otherwise.
    bc_path_ms = f"{DATA_DIR}/dynamic_bc_bc_dynamic_ms_N{N}.json"
    bc_path_s10 = f"{DATA_DIR}/dynamic_bc_bc_dynamic_s10_N{N}.json"
    if os.path.exists(bc_path_ms):
        bc_path = bc_path_ms
    elif os.path.exists(bc_path_s10):
        bc_path = bc_path_s10
    else:
        raise FileNotFoundError(
            f"Missing dynamic-trained BC result for N={N}. Looked for:\n"
            f"  {bc_path_ms}\n  {bc_path_s10}\n"
            f"Run eval_dynamic_marl.py on the corresponding checkpoint first."
        )
    bc = load_json(bc_path)
    bc_probs = [gs(bc, speed, "cum_interf_prob_mean") for speed in speeds]
    bc_stds = [gs(bc, speed, "cum_interf_prob_std") for speed in speeds]

    # CARLTON (CTDE value-based RL baseline, Cohen et al. 2024)
    carlton = load_json(f"{DATA_DIR}/dynamic_carlton_N{N}.json")
    carlton_probs = [gs(carlton, speed, "cum_interf_prob_mean") for speed in speeds]
    carlton_stds = [gs(carlton, speed, "cum_interf_prob_std") for speed in speeds]

    # IQL (no expert): Independent Q-Learning trained in the SAME dynamic multi-speed env
    # as our BC, using the IDENTICAL QNetwork architecture, but learning purely from reward
    # (no expert/teacher labels). Cleanest ablation: same env + same net, only the supervision
    # source differs (reward vs expert action). Demonstrates the value of expert data.
    iql = load_json(f"{DATA_DIR}/dynamic_iql_iql_dynamic_ms_N{N}.json")
    iql_probs = [gs(iql, speed, "cum_interf_prob_mean") for speed in speeds]
    iql_stds = [gs(iql, speed, "cum_interf_prob_std") for speed in speeds]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(speeds))
    width = 0.12

    all_methods = methods_order + ["carlton", "iql", "bc"]
    all_labels = [method_labels[m] for m in methods_order] + ["CARLTON", "IQL (no expert)", "BC (Ours)"]
    all_colors = [method_colors[m] for m in methods_order] + ["#a65628", "#e41a1c", "#984ea3"]
    all_probs = [baseline_data[m]["probs"] for m in methods_order] + [carlton_probs, iql_probs, bc_probs]
    all_stds = [baseline_data[m]["stds"] for m in methods_order] + [carlton_stds, iql_stds, bc_stds]

    for i, (label, color, probs, stds) in enumerate(zip(all_labels, all_colors, all_probs, all_stds)):
        offset = (i - len(all_methods) / 2 + 0.5) * width
        ax.bar(x + offset, probs, width, label=label, color=color, yerr=stds,
               capsize=2, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("UAV Speed (m/s)", fontsize=12)
    ax.set_ylabel("Cumulative Interference Probability", fontsize=12)
    ax.set_title(f"Dynamic Scenario Comparison (N={N}, 50 slots)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in speeds])
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    plt.tight_layout()

    suffix = "" if N == 10 else f"_N{N}"
    out_pdf = os.path.join(FIG_DIR, f"fig1_dynamic_comparison{suffix}.pdf")
    out_png = os.path.join(FIG_DIR, f"fig1_dynamic_comparison{suffix}.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"[N={N}] saved -> {out_pdf}")
    print(f"  BC (dynamic) probs: " + ", ".join(f"{p:.3f}" for p in bc_probs))


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    render(N)
