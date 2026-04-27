#!/usr/bin/env python3
"""Plot Geometry/RepInd cosine-profile comparison results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "geometry_repind"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Geometry/RepInd results.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def load_results(results_dir: Path) -> dict:
    return json.loads((results_dir / "geometry_repind_results.json").read_text())


def plot_change_heatmap(results: dict, results_dir: Path) -> None:
    names = [d["name"] for d in results["directions"]]
    grid = np.array([[results["summary"][intervention][measured]["mean_abs_change"] for measured in names] for intervention in names])

    fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.1), max(4, len(names) * 0.9)))
    image = ax.imshow(grid, cmap="magma", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Mean absolute profile change")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Measured direction")
    ax.set_ylabel("Ablated direction")
    ax.set_title("RepInd Profile Change Matrix")

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white" if grid[i, j] > grid.max() / 2 else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(results_dir / "repind_change_heatmap.png", dpi=180)
    plt.close(fig)


def plot_dim_pair_profiles(results: dict, results_dir: Path) -> None:
    names = [d["name"] for d in results["directions"]]
    if "DIM" not in names:
        return
    x = np.arange(len(results["baseline_profiles"]["DIM"]))

    other_names = [name for name in names if name != "DIM"]
    fig, axes = plt.subplots(len(other_names), 2, figsize=(10, max(3, 2.6 * len(other_names))), squeeze=False, sharex=True)

    for row, other in enumerate(other_names):
        ax = axes[row, 0]
        ax.plot(x, results["baseline_profiles"]["DIM"], label="DIM baseline", color="#3066be")
        ax.plot(x, results["ablated_profiles"][other]["DIM"], label=f"DIM after {other} ablation", color="#3066be", linestyle="--")
        ax.set_title(f"DIM profile under {other} ablation")
        ax.set_ylabel("Cosine")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        ax.plot(x, results["baseline_profiles"][other], label=f"{other} baseline", color="#c33c54")
        ax.plot(x, results["ablated_profiles"]["DIM"][other], label=f"{other} after DIM ablation", color="#c33c54", linestyle="--")
        ax.set_title(f"{other} profile under DIM ablation")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Layer")

    fig.tight_layout()
    fig.savefig(results_dir / "repind_dim_pair_profiles.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = load_results(args.results_dir)
    plot_change_heatmap(results, args.results_dir)
    plot_dim_pair_profiles(results, args.results_dir)
    print(f"Saved Geometry/RepInd plots to {args.results_dir}")


if __name__ == "__main__":
    main()
