#!/usr/bin/env python3
"""Plot safety-vs-utility overlap figures from a saved JSON result file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "safety_utility_overlap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot safety-vs-utility overlap figures.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--primary_rank", type=int, default=8)
    return parser.parse_args()


def load_results(results_dir: Path) -> dict:
    return json.loads((results_dir / "safety_utility_overlap_results.json").read_text())


def sorted_ranks(results: dict) -> list[int]:
    return sorted(int(rank) for rank in results["rank_results"])


def plot_per_layer(results: dict, ranks: list[int], output_dir: Path, primary_rank: int, named_directions: dict = {}) -> None:
    rank_key = str(primary_rank)
    if rank_key not in results["rank_results"]:
        primary_rank = ranks[min(len(ranks) - 1, len(ranks) // 2)]
        rank_key = str(primary_rank)

    scores = np.array(results["rank_results"][rank_key]["mso"])
    baseline = results["rank_results"][rank_key]["random_baseline"]
    layers = np.arange(len(scores))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(layers, scores, color="#3066be", label=f"safety vs utility MSO, k={primary_rank}")
    ax.axhline(baseline, color="#c33c54", linestyle="--", linewidth=1.4, label="random baseline")

    _nd_colors = ["#e6550d", "#31a354", "#756bb1", "#636363"]
    for idx, (name, nd) in enumerate(named_directions.items()):
        if rank_key not in nd:
            continue
        color = _nd_colors[idx % len(_nd_colors)]
        ax.plot(
            layers,
            np.array(nd[rank_key]["mso"]),
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=color,
            label=f"{name} (k={primary_rank})",
        )

    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Projection fraction / MSO")
    ax.set_title("Safety Direction Overlap With Utility Activation Subspace")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(x) for x in layers], fontsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "safety_utility_overlap_per_layer.png", dpi=180)
    plt.close(fig)


def plot_heatmap(results: dict, ranks: list[int], output_dir: Path) -> None:
    grid = np.array([results["rank_results"][str(rank)]["mso"] for rank in ranks])
    baselines = np.array([results["rank_results"][str(rank)]["random_baseline"] for rank in ranks])[:, None]
    ratio_grid = grid / baselines

    fig, ax = plt.subplots(figsize=(11, 3.8))
    image = ax.imshow(ratio_grid, aspect="auto", cmap="viridis", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="MSO / random baseline")
    ax.set_yticks(np.arange(len(ranks)))
    ax.set_yticklabels([str(rank) for rank in ranks])
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_xticklabels([str(x) for x in range(grid.shape[1])], fontsize=7)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Utility PCA rank k")
    ax.set_title("Safety-Utility Overlap Relative to Random Baseline")
    fig.tight_layout()
    fig.savefig(output_dir / "safety_utility_overlap_heatmap.png", dpi=180)
    plt.close(fig)


def plot_rank_summary(results: dict, ranks: list[int], output_dir: Path) -> None:
    means = [results["rank_results"][str(rank)]["mean_mso"] for rank in ranks]
    baselines = [results["rank_results"][str(rank)]["random_baseline"] for rank in ranks]

    x = np.arange(len(ranks))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, means, marker="o", color="#3066be", label="mean safety-utility MSO")
    ax.plot(x, baselines, marker="s", color="#c33c54", linestyle="--", label="random baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([str(rank) for rank in ranks])
    ax.set_xlabel("Utility PCA rank k")
    ax.set_ylabel("Mean MSO across layers")
    ax.set_title("Safety-Utility Overlap by Utility Subspace Rank")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "safety_utility_overlap_by_rank.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results = load_results(args.results_dir)
    ranks = sorted_ranks(results)
    named_directions = results.get("named_directions", {})
    plot_per_layer(results, ranks, args.results_dir, args.primary_rank, named_directions)
    plot_heatmap(results, ranks, args.results_dir)
    plot_rank_summary(results, ranks, args.results_dir)
    print(f"Saved safety-vs-utility overlap plots to {args.results_dir}")


if __name__ == "__main__":
    main()
