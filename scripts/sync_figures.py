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
    ROOT / "code/results/benchmark/jailbreak_asr_judges.png": DOC_FIGURES / "benchmark_jailbreak_asr_judges.png",
    ROOT / "code/results/benchmark/truthfulqa.png": DOC_FIGURES / "benchmark_truthfulqa.png",
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
    ROOT / "code/results/probe_attack_types/scatter_projection_vs_outcome.png": DOC_FIGURES / "probe_scatter_projection_vs_outcome.png",
    ROOT / "code/results/probe_attack_types/boxplot_projection_by_attack_type.png": DOC_FIGURES / "probe_boxplot_projection_by_attack_type.png",
    ROOT / "code/results/probe_attack_types/asr_and_projection_by_attack_type.png": DOC_FIGURES / "probe_asr_and_projection_by_attack_type.png",
    ROOT / "code/results/probe_attack_types/correlation_projection_vs_asr.png": DOC_FIGURES / "probe_correlation_projection_vs_asr.png",
    ROOT / "code/results/probe_attack_types/dim_vs_rco_projection_scatter.png": DOC_FIGURES / "probe_dim_vs_rco_projection_scatter.png",
    ROOT / "code/results/probe_attack_types/layer_sweep_projection.png": DOC_FIGURES / "probe_layer_sweep_projection.png",
    ROOT / "code/results/probe_attack_types/ablation_cross_test.png": DOC_FIGURES / "probe_ablation_cross_test.png",
}

# RCO repind results override the DIM-derived ones if available
FIGURE_MAP_RCO_OVERRIDE = {
    ROOT / "code/results/geometry_repind_rco/repind_change_heatmap.png": DOC_FIGURES / "repind_change_heatmap.png",
    ROOT / "code/results/geometry_repind_rco/repind_dim_pair_profiles.png": DOC_FIGURES / "repind_dim_pair_profiles.png",
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

    # Override with RCO repind figures if they exist
    for src, dst in FIGURE_MAP_RCO_OVERRIDE.items():
        if not src.exists():
            continue
        shutil.copy2(src, dst)
        copied += 1
        print(f"copied (rco override): {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    print(f"done: copied {copied} figure(s)")


if __name__ == "__main__":
    main()
