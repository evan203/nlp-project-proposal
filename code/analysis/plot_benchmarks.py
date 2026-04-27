#!/usr/bin/env python3
"""Regenerate benchmark plots from code/results/benchmark/benchmark_results.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "benchmark"
RESULTS = OUT_DIR / "benchmark_results.json"

# Base is always first; remaining methods are appended in insertion order.
BASE_NAME = "Base (Llama-3.1-8B-Instruct)"
PREFERRED_ORDER = ["Base (Llama-3.1-8B-Instruct)", "DIM-Ablated", "ActSVD-Modified"]
# Extended palette — first three match original colors, rest use matplotlib tab10.
_FIXED_COLORS = ["#4c78a8", "#f58518", "#54a24b"]
_EXTRA_COLORS = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]


def _build_order(data: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (names, short_labels, colors) for every method present in data."""
    seen = set()
    names: list[str] = []
    for name in PREFERRED_ORDER:
        if name in data:
            names.append(name)
            seen.add(name)
    for name in data:
        if name not in seen:
            names.append(name)

    labels = []
    for name in names:
        label = name.replace("(Llama-3.1-8B-Instruct)", "").replace("-Ablated", "").replace("-Modified", "").strip()
        labels.append(label or name)

    colors = []
    extra_idx = 0
    for i in range(len(names)):
        if i < len(_FIXED_COLORS):
            colors.append(_FIXED_COLORS[i])
        else:
            colors.append(_EXTRA_COLORS[extra_idx % len(_EXTRA_COLORS)])
            extra_idx += 1

    return names, labels, colors


def load_results() -> dict:
    return json.loads(RESULTS.read_text())


def add_bar_labels(ax, bars, fmt="{:.2f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_jailbreak_asr(data: dict, names, labels, colors) -> None:
    values = [data[name]["jailbreakbench"]["asr"] for name in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Attack success rate")
    ax.set_title("JailbreakBench ASR")
    add_bar_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "jailbreak_asr.png", dpi=180)
    plt.close(fig)


def plot_harmless_compliance(data: dict, names, labels, colors) -> None:
    values = [data[name]["harmless_compliance"]["rate"] for name in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 4))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Compliance rate")
    ax.set_title("Harmless Prompt Compliance")
    add_bar_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "harmless_compliance.png", dpi=180)
    plt.close(fig)


def plot_perplexity(data: dict, names, labels, colors) -> None:
    datasets = ["pile", "alpaca"]
    x = np.arange(len(datasets))
    n = len(names)
    width = min(0.24, 0.8 / n)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, name in enumerate(names):
        values = [data[name]["perplexity"][dataset]["perplexity"] for dataset in datasets]
        bars = ax.bar(x + offsets[idx], values, width, label=labels[idx], color=colors[idx])
        add_bar_labels(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(["Pile", "Alpaca"])
    ax.set_ylabel("Perplexity")
    ax.set_title("Utility Perplexity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "perplexity_comparison.png", dpi=180)
    plt.close(fig)


def plot_safety_utility_tradeoff(data: dict, names, labels, colors) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, name in enumerate(names):
        asr = data[name]["jailbreakbench"]["asr"]
        ppl = data[name]["perplexity"]["pile"]["perplexity"]
        ax.scatter(asr, ppl, s=90, color=colors[idx], label=labels[idx])
        ax.annotate(labels[idx], (asr, ppl), xytext=(6, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("JailbreakBench ASR")
    ax.set_ylabel("Pile perplexity")
    ax.set_title("Safety-Utility Tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "safety_utility_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_jailbreak_asr_per_category(data: dict, names, labels, colors) -> None:
    categories = list(data[names[0]]["jailbreakbench"]["per_category"].keys())
    x = np.arange(len(categories))
    n = len(names)
    width = min(0.24, 0.8 / n)
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for idx, name in enumerate(names):
        values = [data[name]["jailbreakbench"]["per_category"][category] for category in categories]
        ax.bar(x + offsets[idx], values, width, label=labels[idx], color=colors[idx])

    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Attack success rate")
    ax.set_title("JailbreakBench ASR by Harm Category")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "jailbreak_asr_per_category.png", dpi=180)
    plt.close(fig)


def main() -> None:
    data = load_results()
    names, labels, colors = _build_order(data)
    print(f"Methods in benchmark: {names}")
    plot_jailbreak_asr(data, names, labels, colors)
    plot_harmless_compliance(data, names, labels, colors)
    plot_perplexity(data, names, labels, colors)
    plot_safety_utility_tradeoff(data, names, labels, colors)
    plot_jailbreak_asr_per_category(data, names, labels, colors)
    print(f"Saved benchmark plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
