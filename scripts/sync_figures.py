#!/usr/bin/env python3
"""Copy generated experiment figures into docs/figures with report-friendly names."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_FIGURES = ROOT / "docs" / "figures"

FIGURE_MAP = {
    ROOT / "code/results/benchmark/jailbreak_asr.png": DOC_FIGURES / "benchmark_jailbreak_asr_overall.png",
    ROOT / "code/results/benchmark/jailbreak_asr_per_category.png": DOC_FIGURES / "benchmark_jailbreak_asr_per_category.png",
    ROOT / "code/results/benchmark/perplexity_comparison.png": DOC_FIGURES / "benchmark_perplexity_pile_alpaca.png",
    ROOT / "code/results/benchmark/safety_utility_tradeoff.png": DOC_FIGURES / "benchmark_safety_utility_tradeoff.png",
    ROOT / "code/results/method_overlap/mso_heatmap.png": DOC_FIGURES / "subspace_mso_heatmap_layer_by_weight.png",
    ROOT / "code/results/method_overlap/mso_per_layer.png": DOC_FIGURES / "subspace_mso_per_layer_avg.png",
    ROOT / "code/results/method_overlap/cross_model_cosine.png": DOC_FIGURES / "subspace_cross_model_dim_cosine.png",
    ROOT / "code/results/safety_utility_overlap/safety_utility_overlap_per_layer.png": DOC_FIGURES / "safety_utility_overlap_per_layer.png",
    ROOT / "code/results/safety_utility_overlap/safety_utility_overlap_heatmap.png": DOC_FIGURES / "safety_utility_overlap_heatmap.png",
    ROOT / "code/results/safety_utility_overlap/safety_utility_overlap_by_rank.png": DOC_FIGURES / "safety_utility_overlap_by_rank.png",
    ROOT / "code/results/geometry_repind/repind_change_heatmap.png": DOC_FIGURES / "repind_change_heatmap.png",
    ROOT / "code/results/geometry_repind/repind_dim_pair_profiles.png": DOC_FIGURES / "repind_dim_pair_profiles.png",
}


def main() -> None:
    DOC_FIGURES.mkdir(parents=True, exist_ok=True)
    copied = 0

    for src, dst in FIGURE_MAP.items():
        if not src.exists():
            print(f"missing: {src.relative_to(ROOT)}")
            continue
        shutil.copy2(src, dst)
        copied += 1
        print(f"copied: {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    print(f"done: copied {copied} figure(s)")


if __name__ == "__main__":
    main()
