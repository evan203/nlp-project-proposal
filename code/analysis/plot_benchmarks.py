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
MODEL_ORDER = ["Base (Llama-3.1-8B-Instruct)", "DIM-Ablated", "ActSVD-Modified"]
MODEL_LABELS = ["Base", "DIM-Ablated", "ActSVD"]
COLORS = ["#4c78a8", "#f58518", "#54a24b"]


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


def plot_jailbreak_asr(data: dict) -> None:
    values = [data[name]["jailbreakbench"]["asr"] for name in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(MODEL_LABELS, values, color=COLORS)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Attack success rate")
    ax.set_title("JailbreakBench ASR")
    add_bar_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "jailbreak_asr.png", dpi=180)
    plt.close(fig)


def plot_harmless_compliance(data: dict) -> None:
    values = [data[name]["harmless_compliance"]["rate"] for name in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(MODEL_LABELS, values, color=COLORS)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Compliance rate")
    ax.set_title("Harmless Prompt Compliance")
    add_bar_labels(ax, bars)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "harmless_compliance.png", dpi=180)
    plt.close(fig)


def plot_perplexity(data: dict) -> None:
    datasets = ["pile", "alpaca"]
    x = np.arange(len(datasets))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, name in enumerate(MODEL_ORDER):
        values = [data[name]["perplexity"][dataset]["perplexity"] for dataset in datasets]
        bars = ax.bar(x + (idx - 1) * width, values, width, label=MODEL_LABELS[idx], color=COLORS[idx])
        add_bar_labels(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(["Pile", "Alpaca"])
    ax.set_ylabel("Perplexity")
    ax.set_title("Utility Perplexity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "perplexity_comparison.png", dpi=180)
    plt.close(fig)


def plot_safety_utility_tradeoff(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, name in enumerate(MODEL_ORDER):
        asr = data[name]["jailbreakbench"]["asr"]
        ppl = data[name]["perplexity"]["pile"]["perplexity"]
        ax.scatter(asr, ppl, s=90, color=COLORS[idx], label=MODEL_LABELS[idx])
        ax.annotate(MODEL_LABELS[idx], (asr, ppl), xytext=(6, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("JailbreakBench ASR")
    ax.set_ylabel("Pile perplexity")
    ax.set_title("Safety-Utility Tradeoff")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "safety_utility_tradeoff.png", dpi=180)
    plt.close(fig)


def plot_jailbreak_asr_per_category(data: dict) -> None:
    categories = list(data[MODEL_ORDER[0]]["jailbreakbench"]["per_category"].keys())
    x = np.arange(len(categories))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for idx, name in enumerate(MODEL_ORDER):
        values = [data[name]["jailbreakbench"]["per_category"][category] for category in categories]
        ax.bar(x + (idx - 1) * width, values, width, label=MODEL_LABELS[idx], color=COLORS[idx])

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
    plot_jailbreak_asr(data)
    plot_harmless_compliance(data)
    plot_perplexity(data)
    plot_safety_utility_tradeoff(data)
    plot_jailbreak_asr_per_category(data)
    print(f"Saved benchmark plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
