#!/usr/bin/env python3
"""Post-hoc analysis and figure generation for Experiment 1.

Usage:
    python analyze.py --results results/debug/results.json
    python analyze.py --results results/full/results.json --output figures/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(path):
    """Load results JSON."""
    with open(path) as f:
        data = json.load(f)
    meta = data.pop("_meta", {})
    return data, meta


def parse_condition(condition_key):
    """Parse 'gaussian_eps0.01' -> ('gaussian', 0.01)."""
    parts = condition_key.split("_eps")
    ptype = parts[0]
    eps = float(parts[1])
    return ptype, eps


def plot_snr_ratio_heatmap(results, meta, output_dir):
    """Heatmap: perturbation_type x epsilon, colored by SNR ratio.

    This is the key figure — shows where Jacobian monitoring wins.
    """
    # Collect unique perturbation types and epsilons
    ptypes = set()
    epsilons = set()
    for cond in results:
        pt, eps = parse_condition(cond)
        ptypes.add(pt)
        epsilons.add(eps)

    ptypes = sorted(ptypes)
    epsilons = sorted(epsilons)

    # For each condition, average ratio across layers
    ratio_matrix = np.zeros((len(ptypes), len(epsilons)))

    for i, pt in enumerate(ptypes):
        for j, eps in enumerate(epsilons):
            cond_key = f"{pt}_eps{eps}"
            if cond_key in results:
                layer_results = results[cond_key]
                ratios = [m["ratio"] for m in layer_results.values()
                          if np.isfinite(m["ratio"])]
                ratio_matrix[i, j] = np.mean(ratios) if ratios else 0
            else:
                ratio_matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    # Custom colormap: red (<1) -> yellow (1-3) -> green (>3)
    cmap = sns.diverging_palette(10, 130, s=80, l=55, n=256, as_cmap=True)

    sns.heatmap(
        ratio_matrix,
        xticklabels=[f"{e:.4f}" for e in epsilons],
        yticklabels=ptypes,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=1.0,
        vmin=0,
        vmax=max(6, np.nanmax(ratio_matrix)),
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Perturbation magnitude (ε = ||ΔW||/||W||)")
    ax.set_ylabel("Perturbation type")
    ax.set_title(f"SNR_J / SNR_act ratio — {meta.get('model', 'unknown')}\n"
                 f"(>3 = GO, 1-3 = MARGINAL, <1 = NO-GO)")

    fig.tight_layout()
    fig.savefig(output_dir / "snr_ratio_heatmap.pdf", dpi=150)
    fig.savefig(output_dir / "snr_ratio_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved: snr_ratio_heatmap.pdf/png")


def plot_snr_by_layer(results, meta, output_dir):
    """Line plot: SNR_J and SNR_act across layers for each condition."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for cond_key, layer_results in results.items():
        ptype, eps = parse_condition(cond_key)
        layers = sorted([int(l) for l in layer_results.keys()])
        snr_acts = [layer_results[str(l)]["snr_act"] for l in layers]
        snr_js = [layer_results[str(l)]["snr_j"] for l in layers]

        label = f"{ptype} ε={eps}"
        axes[0].plot(layers, snr_acts, marker="o", markersize=3, label=label, alpha=0.7)
        axes[1].plot(layers, snr_js, marker="o", markersize=3, label=label, alpha=0.7)

    axes[0].set_ylabel("SNR_act")
    axes[0].set_title(f"Activation SNR by layer — {meta.get('model', 'unknown')}")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("SNR_J")
    axes[1].set_xlabel("Layer index")
    axes[1].set_title("Jacobian SNR by layer")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "snr_by_layer.pdf", dpi=150)
    fig.savefig(output_dir / "snr_by_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved: snr_by_layer.pdf/png")


def plot_ratio_by_layer(results, meta, output_dir):
    """Line plot: ratio = SNR_J / SNR_act across layers."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for cond_key, layer_results in results.items():
        ptype, eps = parse_condition(cond_key)
        layers = sorted([int(l) for l in layer_results.keys()])
        ratios = [layer_results[str(l)]["ratio"] for l in layers]
        # Clip infinite values for plotting
        ratios = [min(r, 50) for r in ratios]

        label = f"{ptype} ε={eps}"
        ax.plot(layers, ratios, marker="o", markersize=3, label=label, alpha=0.7)

    # Decision boundaries
    ax.axhline(y=3, color="green", linestyle="--", alpha=0.5, label="GO threshold (3)")
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="NO-GO threshold (1)")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("SNR_J / SNR_act ratio")
    ax.set_title(f"Jacobian advantage ratio by layer — {meta.get('model', 'unknown')}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "ratio_by_layer.pdf", dpi=150)
    fig.savefig(output_dir / "ratio_by_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved: ratio_by_layer.pdf/png")


def plot_epsilon_sensitivity(results, meta, output_dir):
    """For each perturbation type, plot ratio vs epsilon (averaged across layers)."""
    # Group by perturbation type
    by_type = {}
    for cond_key, layer_results in results.items():
        ptype, eps = parse_condition(cond_key)
        if ptype not in by_type:
            by_type[ptype] = []
        ratios = [m["ratio"] for m in layer_results.values() if np.isfinite(m["ratio"])]
        mean_ratio = np.mean(ratios) if ratios else 0
        by_type[ptype].append((eps, mean_ratio))

    fig, ax = plt.subplots(figsize=(10, 6))

    for ptype, points in by_type.items():
        points.sort()
        eps_vals, ratio_vals = zip(*points)
        ax.plot(eps_vals, ratio_vals, marker="s", markersize=6, label=ptype, linewidth=2)

    ax.axhline(y=3, color="green", linestyle="--", alpha=0.5, label="GO threshold")
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="NO-GO threshold")

    ax.set_xscale("log")
    ax.set_xlabel("Perturbation magnitude (ε)")
    ax.set_ylabel("Mean SNR_J / SNR_act ratio")
    ax.set_title(f"Jacobian advantage vs perturbation size — {meta.get('model', 'unknown')}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "epsilon_sensitivity.pdf", dpi=150)
    fig.savefig(output_dir / "epsilon_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"Saved: epsilon_sensitivity.pdf/png")


def print_summary_table(results, meta):
    """Print a compact summary table to stdout."""
    print("\n" + "=" * 80)
    print(f"EXPERIMENT 1 SUMMARY — {meta.get('model', 'unknown')}")
    print(f"Prompts: {meta.get('num_prompts', '?')}, "
          f"Directions: {meta.get('num_directions', '?')}")
    print("=" * 80)
    print(f"{'Condition':<30} | {'Mean SNR_act':>12} | {'Mean SNR_J':>12} | "
          f"{'Mean Ratio':>12} | {'Decision':>10}")
    print("-" * 80)

    for cond_key in sorted(results.keys()):
        layer_results = results[cond_key]
        snr_acts = [m["snr_act"] for m in layer_results.values()]
        snr_js = [m["snr_j"] for m in layer_results.values()]
        ratios = [m["ratio"] for m in layer_results.values() if np.isfinite(m["ratio"])]

        mean_act = np.mean(snr_acts) if snr_acts else 0
        mean_j = np.mean(snr_js) if snr_js else 0
        mean_ratio = np.mean(ratios) if ratios else 0

        if mean_ratio > 3:
            decision = "GO"
        elif mean_ratio > 1:
            decision = "MARGINAL"
        else:
            decision = "NO-GO"

        print(f"{cond_key:<30} | {mean_act:>12.3f} | {mean_j:>12.3f} | "
              f"{mean_ratio:>12.3f} | {decision:>10}")

    overall = meta.get("overall_decision", "?")
    print("-" * 80)
    print(f"{'OVERALL':>30} |{' '*12} |{' '*12} |{' '*12} | {overall:>10}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze Experiment 1 results")
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--output", default=None, help="Output directory for figures")
    args = parser.parse_args()

    results, meta = load_results(args.results)

    # Default output dir: same as results file
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.results).parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print summary
    print_summary_table(results, meta)

    # Generate figures
    print(f"\nGenerating figures in {output_dir}/")
    plot_snr_ratio_heatmap(results, meta, output_dir)
    plot_snr_by_layer(results, meta, output_dir)
    plot_ratio_by_layer(results, meta, output_dir)
    plot_epsilon_sensitivity(results, meta, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
