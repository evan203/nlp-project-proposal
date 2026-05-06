#!/usr/bin/env python3
"""
Subspace comparison visualisations.

Reads the already-computed results from code/results/method_overlap/comparison_results.json
and produces:
  1. <pair>_mso_heatmap.png    – layers (rows) × weight-type (cols) heatmap of MSO values
  2. <pair>_mso_per_layer.png  – per-layer average MSO bar chart with random baseline
  3. cross_model_cosine.png    – cross-model DIM direction cosine similarity bar chart
  4. direction_cosine.png      – direction-pair cosine similarity bar chart (if present)
"""

from __future__ import annotations
import json, re, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results" / "method_overlap"
RESULTS = OUT_DIR / "comparison_results.json"

# ── helpers ─────────────────────────────────────────────────────────────
_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)\.weight")

WEIGHT_TYPE_ORDER = ["self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj"]
WEIGHT_TYPE_LABELS = {
    "self_attn.q_proj": "Q proj",
    "self_attn.o_proj": "O proj",
    "mlp.down_proj": "MLP down",
}


def _parse_layer_and_type(key: str):
    m = _LAYER_RE.match(key)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name)


# ── 1. MSO Heatmap ─────────────────────────────────────────────────────
def plot_mso_heatmap(mso_data: dict, pair_name: str, out_dir: pathlib.Path) -> None:
    layers_set: set[int] = set()
    types_set: set[str] = set()
    vals: dict[tuple[int, str], float] = {}
    baselines: dict[tuple[int, str], float] = {}

    for key, info in mso_data.items():
        layer, wtype = _parse_layer_and_type(key)
        if layer is None:
            continue
        layers_set.add(layer)
        types_set.add(wtype)
        vals[(layer, wtype)] = info["mso"]
        baselines[(layer, wtype)] = info["random_baseline"]

    layers = sorted(layers_set)
    wtypes = [t for t in WEIGHT_TYPE_ORDER if t in types_set]
    wtypes += sorted(types_set - set(WEIGHT_TYPE_ORDER))

    n_layers = len(layers)
    n_types = len(wtypes)

    if n_layers == 0 or n_types == 0:
        print(f"Warning: no valid layers/types for heatmap '{pair_name}', skipping.")
        return

    grid = np.full((n_layers, n_types), np.nan)
    baseline_grid = np.full((n_layers, n_types), np.nan)
    for i, l in enumerate(layers):
        for j, t in enumerate(wtypes):
            if (l, t) in vals:
                grid[i, j] = vals[(l, t)]
                baseline_grid[i, j] = baselines[(l, t)]

    ratio_grid = grid / np.where(baseline_grid > 0, baseline_grid, 1.0)

    fig, ax = plt.subplots(figsize=(5, 10))
    im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.6, label="MSO")

    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([str(l) for l in layers], fontsize=7)
    ax.set_xticks(range(n_types))
    ax.set_xticklabels([WEIGHT_TYPE_LABELS.get(t, t) for t in wtypes], fontsize=9)
    ax.set_ylabel("Transformer Layer")
    ax.set_xlabel("Weight Matrix")
    ax.set_title(f"MSO: {pair_name}\n(per layer × weight type)")

    for i in range(n_layers):
        for j in range(n_types):
            v = grid[i, j]
            r = ratio_grid[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.035 else "black"
                ax.text(j, i, f"{v:.3f}\n({r:.1f}×)", ha="center", va="center",
                        fontsize=5.5, color=color)

    plt.tight_layout()
    fname = _safe_filename(pair_name) + "_mso_heatmap.png"
    out = out_dir / fname
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


# ── 2. Per-layer average MSO bar chart ──────────────────────────────────
def plot_mso_per_layer(mso_data: dict, pair_name: str, out_dir: pathlib.Path) -> None:
    layer_mso: dict[int, list[float]] = defaultdict(list)
    layer_baseline: dict[int, list[float]] = defaultdict(list)
    for key, info in mso_data.items():
        layer, _ = _parse_layer_and_type(key)
        if layer is None:
            continue
        layer_mso[layer].append(info["mso"])
        layer_baseline[layer].append(info["random_baseline"])

    layers = sorted(layer_mso)
    if not layers:
        print(f"Warning: no valid layers for per-layer plot '{pair_name}', skipping.")
        return

    avg_mso = [np.mean(layer_mso[l]) for l in layers]
    avg_base = [np.mean(layer_baseline[l]) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(layers))
    ax.bar(x - 0.15, avg_mso, 0.3, label=f"{pair_name} MSO", color="#d62728")
    ax.bar(x + 0.15, avg_base, 0.3, label="Random baseline (k_A·k_B / d)",
           color="#aec7e8", edgecolor="#1f77b4", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Subspace Overlap")
    ax.set_title(f"Per-layer average MSO: {pair_name}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fname = _safe_filename(pair_name) + "_mso_per_layer.png"
    out = out_dir / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── 3. Cross-model DIM direction cosine similarity ─────────────────────
def plot_cross_model_cosine(cosine_data: dict) -> None:
    pairs = sorted(cosine_data.keys(), key=lambda k: abs(cosine_data[k]), reverse=True)
    vals = [cosine_data[p] for p in pairs]
    short = [p.replace("Llama-3.1-8B-Instruct", "Llama-3.1")
              .replace("meta-llama-3-8b-instruct", "Llama-3")
              .replace("llama-2-7b-chat-hf", "Llama-2")
              .replace("yi-6b-chat", "Yi-6B")
              .replace("gemma-2b-it", "Gemma-2B")
              .replace("qwen-1_8b-chat", "Qwen-1.8B")
             for p in pairs]

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#2ca02c" if v > 0.1 else "#1f77b4" for v in vals]
    ax.barh(range(len(vals)), vals, color=colours)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Cosine similarity")
    ax.set_title("Cross-model DIM refusal-direction cosine similarity")
    ax.axvline(0, color="grey", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.01 * (1 if v >= 0 else -1), i, f"{v:.3f}",
                va="center", fontsize=7)
    plt.tight_layout()
    out = OUT_DIR / "cross_model_cosine.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── 4. Direction-pair cosine similarity ─────────────────────────────────
def plot_direction_cosine(cosine_data: dict) -> None:
    pairs = sorted(cosine_data.keys(), key=lambda k: abs(cosine_data[k]), reverse=True)
    vals = [cosine_data[p] for p in pairs]

    fig, ax = plt.subplots(figsize=(max(6, len(pairs) * 1.5), 4))
    colours = ["#2ca02c" if v > 0.1 else "#1f77b4" for v in vals]
    ax.bar(range(len(vals)), vals, color=colours)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(pairs, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Direction-pair cosine similarities")
    ax.axhline(0, color="grey", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01 * (1 if v >= 0 else -1), f"{v:.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / "direction_cosine.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# ── main ────────────────────────────────────────────────────────────────
def main():
    if not RESULTS.exists():
        sys.exit(f"Results file not found: {RESULTS}")

    data = json.loads(RESULTS.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mso_keys = [k for k in data.get("mso", {}).keys() if k.endswith("_per_layer")]
    for pair_key in mso_keys:
        mso_data = data["mso"][pair_key]
        pair_name = pair_key.replace("_per_layer", "").replace("_", " ")
        plot_mso_heatmap(mso_data, pair_name, OUT_DIR)
        plot_mso_per_layer(mso_data, pair_name, OUT_DIR)

    if "cross_model_dim_cosine" in data:
        plot_cross_model_cosine(data["cross_model_dim_cosine"])

    direction_cosine = data.get("direction_cosine", {})
    if direction_cosine:
        plot_direction_cosine(direction_cosine)

    print("Done — all plots saved to", OUT_DIR)


if __name__ == "__main__":
    main()
